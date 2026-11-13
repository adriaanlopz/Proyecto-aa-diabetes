"""Modelado supervisado: construcción, evaluación y configuración final de los modelos.

Todos los modelos son un `Pipeline` que recibe las columnas **crudas** del CSV
(`pp.columnas_crudas`) y empieza por el preprocesador de `preprocesamiento.py`, de
modo que el mismo objeto sirve para validar, para evaluar y para la API.

Convención de importación
-------------------------
Este módulo importa `preprocesamiento` como módulo de primer nivel, con `src/` en
`sys.path`, que es exactamente lo que hace el servicio (`src/servicio.py`). Así los
artefactos serializados referencian `preprocesamiento.*` y `modelado.*`, y el servicio
los carga sin depender del directorio desde el que se lance.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    FixedThresholdClassifier,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer

import preprocesamiento as pp

VERSION_MODELO = "1.0.0"
N_PLIEGUES = 5

# Métricas de validación cruzada: nombre legible → scorer de sklearn. Las de error
# (Brier, log-loss) se devuelven negadas por sklearn y se corrigen en `evaluar_cv`.
METRICAS_CV = {
    "ROC-AUC": "roc_auc",
    "PR-AUC": "average_precision",
    "Brier": "neg_brier_score",
    "Log-loss": "neg_log_loss",
}
METRICAS_ERROR = {"Brier", "Log-loss"}

# ---------------------------------------------------------------------------
# Pasos auxiliares (funciones de módulo para que el artefacto sea serializable)
# ---------------------------------------------------------------------------

COLS_CLINICAS = ["presion_alta", "colesterol_alto", "imc", "grupo_edad"]
COLS_SPLINE_IMC = ["imc"]
COLS_SPLINE_RESTO = [
    "dias_salud_mental_mala",
    "grupo_edad",
    "salud_general",
    "nivel_educativo",
    "nivel_ingresos",
]


def seleccionar_clinicas(X: pd.DataFrame) -> pd.DataFrame:
    """Solo los cuatro indicadores clínicos clásicos (baseline de referencia)."""
    return X[COLS_CLINICAS]


def quitar_derivadas(X: pd.DataFrame) -> pd.DataFrame:
    """Quita `imc_categoria` y `n_factores_riesgo_cv` en modelos lineales.

    Son combinaciones de variables que ya entran en el modelo (decisión #9 del
    README: en modelos lineales, la derivada *o* sus componentes, no las dos).
    """
    return X.drop(columns=pp.COLS_FEATURES_NUEVAS_ORD)


# ---------------------------------------------------------------------------
# Catálogo de modelos
# ---------------------------------------------------------------------------


def construir_modelo(linea: str, estimador: Any, umbral: float | None = None) -> Pipeline:
    """Pipeline completo: datos crudos → preprocesador → estimador.

    Si se pasa `umbral`, el estimador se envuelve en `FixedThresholdClassifier`
    *dentro* del pipeline: `predict` aplica ese umbral y el pipeline sigue
    exponiendo `feature_names_in_`, que el servicio usa para filtrar columnas.
    """
    if umbral is not None:
        estimador = FixedThresholdClassifier(estimador, threshold=umbral, response_method="predict_proba")
    return Pipeline([("pre", pp.construir_preprocesador(linea)), ("modelo", estimador)])


def logistica(C: float = 1.0) -> Pipeline:
    return make_pipeline(
        FunctionTransformer(quitar_derivadas),
        LogisticRegression(C=C, max_iter=5000),
    )


def logistica_clinica() -> Pipeline:
    return make_pipeline(
        FunctionTransformer(seleccionar_clinicas),
        LogisticRegression(max_iter=5000),
    )


def logistica_splines(C: float = 0.1, nodos_imc: int = 5, nodos_resto: int = 5) -> Pipeline:
    """Regresión logística con B-splines cúbicos en las variables no binarias.

    Permite relaciones no lineales (p. ej. el riesgo crece con el IMC pero se aplana
    en obesidad extrema) sin perder la interpretabilidad de un modelo aditivo. En el
    IMC los nodos van por cuantiles para que los outliers no los desplacen; en el
    resto, uniformes (son ordinales con pocos valores, los cuantiles se repetirían).
    `extrapolation="constant"` hace que un valor fuera del rango visto en
    entrenamiento no dispare la predicción.
    """
    splines = ColumnTransformer(
        [
            ("imc", SplineTransformer(n_knots=nodos_imc, knots="quantile", extrapolation="constant"), COLS_SPLINE_IMC),
            ("resto", SplineTransformer(n_knots=nodos_resto, extrapolation="constant"), COLS_SPLINE_RESTO),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return make_pipeline(
        FunctionTransformer(quitar_derivadas),
        splines,
        LogisticRegression(C=C, max_iter=5000),
    )


def gradient_boosting(**params: Any) -> HistGradientBoostingClassifier:
    base = dict(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=8,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=pp.SEMILLA,
    )
    return HistGradientBoostingClassifier(**(base | params))


def ensamble(params_logistica: dict | None = None, params_boosting: dict | None = None) -> VotingClassifier:
    """Media de probabilidades (voto suave) de la logística con splines y el boosting.

    Combina un modelo aditivo suave con otro que capta interacciones: sus errores
    están poco correlacionados, así que promediarlos reduce varianza.
    """
    return VotingClassifier(
        [
            ("logistica", logistica_splines(**(params_logistica or {}))),
            ("boosting", gradient_boosting(**(params_boosting or {}))),
        ],
        voting="soft",
    )


def candidatos() -> dict[str, Any]:
    """Familias comparadas en la primera ronda, con hiperparámetros razonables por defecto."""
    return {
        "Dummy (prevalencia)": DummyClassifier(strategy="prior"),
        "Logística clínica (4 var.)": logistica_clinica(),
        "Logística": logistica(),
        "Logística + splines": logistica_splines(),
        "KNN (k=50)": KNeighborsClassifier(n_neighbors=50),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=pp.SEMILLA
        ),
        "Gradient Boosting": gradient_boosting(),
    }


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


def validacion_cruzada(repeticiones: int = 1) -> StratifiedKFold | RepeatedStratifiedKFold:
    """Particiones estratificadas fijas (misma semilla para todos los modelos → comparación pareada)."""
    if repeticiones == 1:
        return StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=pp.SEMILLA)
    return RepeatedStratifiedKFold(n_splits=N_PLIEGUES, n_repeats=repeticiones, random_state=pp.SEMILLA)


def evaluar_cv(modelo: Pipeline, X: pd.DataFrame, y: pd.Series, cv=None) -> pd.DataFrame:
    """Métricas por pliegue (filas) en validación cruzada. Brier y log-loss en positivo."""
    cv = validacion_cruzada() if cv is None else cv
    res = cross_validate(modelo, X, y, cv=cv, scoring=METRICAS_CV, n_jobs=-1)
    tabla = pd.DataFrame({nombre: res[f"test_{nombre}"] for nombre in METRICAS_CV})
    for nombre in METRICAS_ERROR:
        tabla[nombre] = -tabla[nombre]
    tabla["Tiempo ajuste (s)"] = res["fit_time"]
    return tabla


def metricas(y: np.ndarray, proba: np.ndarray, umbral: float) -> dict[str, float]:
    """Métricas de ranking, calibración y de la decisión con un umbral dado."""
    pred = (proba >= umbral).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "Brier": brier_score_loss(y, proba),
        "Log-loss": log_loss(y, proba),
        "Sensibilidad": recall_score(y, pred),
        "Especificidad": recall_score(y, pred, pos_label=0),
        "Precisión (VPP)": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred),
        "Exactitud": float((pred == y).mean()),
    }


def metricas_con_ic(
    y: pd.Series, proba: np.ndarray, umbral: float, n_boot: int = 1000, nivel: float = 0.95
) -> pd.DataFrame:
    """Métricas con intervalo de confianza por bootstrap percentil (remuestreo de pacientes)."""
    y = np.asarray(y)
    proba = np.asarray(proba)
    rng = np.random.default_rng(pp.SEMILLA)
    puntual = metricas(y, proba, umbral)
    muestras = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        muestras.append(metricas(y[idx], proba[idx], umbral))
    muestras = pd.DataFrame(muestras)
    alfa = (1 - nivel) / 2
    return pd.DataFrame(
        {
            "valor": puntual,
            "IC inf.": muestras.quantile(alfa),
            "IC sup.": muestras.quantile(1 - alfa),
        }
    )


def dependencia_parcial(modelo: Pipeline, X: pd.DataFrame, columna: str, valores: list) -> pd.Series:
    """Probabilidad media predicha fijando `columna` a cada valor (sobre datos crudos).

    Se calcula a mano en lugar de con `PartialDependenceDisplay` porque la entrada son
    columnas crudas de texto (p. ej. `grupo_edad = "60-64"`), que sklearn no sabe
    rejillar.
    """
    salida = {}
    for valor in valores:
        X_mod = X.copy()
        X_mod[columna] = valor
        salida[valor] = modelo.predict_proba(X_mod)[:, 1].mean()
    return pd.Series(salida, name=columna)


DIAS_MIN, DIAS_MAX = pp.RANGO_DIAS


class RegresorAcotado(BaseEstimator, RegressorMixin):
    def __init__(self, estimador: Any):
        self.estimador = estimador

    def fit(self, X, y):
        self.estimador_ = clone(self.estimador).fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.estimador_.predict(X), DIAS_MIN, DIAS_MAX)


class DosPartes(BaseEstimator, RegressorMixin):
    def __init__(self, clasificador: Any, regresor: Any, umbral: float | None = None):
        self.clasificador = clasificador
        self.regresor = regresor
        self.umbral = umbral

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        positivos = y > 0
        self.clasificador_ = clone(self.clasificador).fit(X, positivos.astype(int))
        self.regresor_ = clone(self.regresor).fit(X[positivos], y[positivos])
        return self

    def predict(self, X):
        prob = self.clasificador_.predict_proba(X)[:, 1]
        dias = self.regresor_.predict(X)
        if self.umbral is None:
            return prob * dias
        return np.where(prob >= self.umbral, dias, 0.0)


def boosting_regresion(perdida: str = "poisson", **params: Any) -> HistGradientBoostingRegressor:
    base = dict(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        random_state=pp.SEMILLA,
    )
    return HistGradientBoostingRegressor(loss=perdida, **(base | params))


def dos_partes(
    umbral: float | None = None,
    params_clasificador: dict | None = None,
    params_regresor: dict | None = None,
) -> DosPartes:
    perdida = "squared_error" if umbral is None else "absolute_error"
    return DosPartes(
        gradient_boosting(**(params_clasificador or {})),
        boosting_regresion(perdida, **(params_regresor or {})),
        umbral,
    )


def construir_regresion(estimador: Any) -> Pipeline:
    return construir_modelo("regresion", RegresorAcotado(estimador))


def candidatos_regresion() -> dict[str, Any]:
    return {
        "Dummy (media)": DummyRegressor(strategy="mean"),
        "Dummy (mediana = 0)": DummyRegressor(strategy="median"),
        "Ridge": make_pipeline(FunctionTransformer(quitar_derivadas), Ridge(alpha=1.0)),
        "GLM Poisson": make_pipeline(
            FunctionTransformer(quitar_derivadas), PoissonRegressor(alpha=1e-3, max_iter=1000)
        ),
        "Boosting (error cuadrático)": boosting_regresion("squared_error"),
        "Boosting (Poisson)": boosting_regresion("poisson"),
        "Boosting (error absoluto)": boosting_regresion("absolute_error"),
        "Dos partes (media)": dos_partes(),
        "Dos partes (mediana)": dos_partes(umbral=0.5),
    }


def validacion_cruzada_regresion(y: pd.Series, repeticiones: int = 1) -> list:
    if repeticiones == 1:
        cv = StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=pp.SEMILLA)
    else:
        cv = RepeatedStratifiedKFold(n_splits=N_PLIEGUES, n_repeats=repeticiones, random_state=pp.SEMILLA)
    return list(cv.split(np.zeros(len(y)), pp.tramos_regresion(y)))


def metricas_regresion(y, pred) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    intermedio = (y > 0) & (y < DIAS_MAX)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R²": r2_score(y, pred),
        "MAE (0 días)": mean_absolute_error(y[y == 0], pred[y == 0]),
        "MAE (1-29 días)": mean_absolute_error(y[intermedio], pred[intermedio]),
        "MAE (30 días)": mean_absolute_error(y[y == DIAS_MAX], pred[y == DIAS_MAX]),
        "Media predicha": float(pred.mean()),
    }


def evaluar_cv_regresion(modelo: Pipeline, X: pd.DataFrame, y: pd.Series, pliegues: list) -> pd.DataFrame:
    filas = []
    for entreno, valida in pliegues:
        inicio = time.perf_counter()
        ajustado = clone(modelo).fit(X.iloc[entreno], y.iloc[entreno])
        segundos = time.perf_counter() - inicio
        fila = metricas_regresion(y.iloc[valida], ajustado.predict(X.iloc[valida]))
        filas.append(fila | {"Tiempo ajuste (s)": segundos})
    return pd.DataFrame(filas)


def metricas_regresion_con_ic(y, pred, n_boot: int = 1000, nivel: float = 0.95) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rng = np.random.default_rng(pp.SEMILLA)
    muestras = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        muestras.append(metricas_regresion(y[idx], pred[idx]))
    muestras = pd.DataFrame(muestras)
    alfa = (1 - nivel) / 2
    return pd.DataFrame(
        {
            "valor": metricas_regresion(y, pred),
            "IC inf.": muestras.quantile(alfa),
            "IC sup.": muestras.quantile(1 - alfa),
        }
    )


def kmeans(k: int, semilla: int = pp.SEMILLA) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=semilla)


def construir_clustering(estimador: Any) -> Pipeline:
    return construir_modelo("clustering", estimador)


def metricas_clustering(X: np.ndarray, etiquetas: np.ndarray, n_silueta: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(pp.SEMILLA)
    muestra = rng.choice(len(X), min(n_silueta, len(X)), replace=False)
    tamanos = np.bincount(etiquetas)
    return {
        "Silueta": silhouette_score(X[muestra], etiquetas[muestra]),
        "Davies-Bouldin": davies_bouldin_score(X, etiquetas),
        "Calinski-Harabasz": calinski_harabasz_score(X, etiquetas),
        "Grupo más pequeño (%)": 100 * tamanos.min() / len(etiquetas),
    }


def estabilidad_clustering(construir, X: pd.DataFrame, k: int, repeticiones: int = 5) -> float:
    aris = []
    for r in range(repeticiones):
        orden = np.random.default_rng(r).permutation(len(X))
        mitad_a, mitad_b = orden[: len(X) // 2], orden[len(X) // 2 :]
        modelo_a = construir(k, r).fit(X.iloc[mitad_a])
        modelo_b = construir(k, r + 100).fit(X.iloc[mitad_b])
        aris.append(adjusted_rand_score(modelo_a.predict(X), modelo_b.predict(X)))
    return float(np.mean(aris))


# ---------------------------------------------------------------------------
# Configuración final (fijada a partir de notebooks/03_modelado_clasificacion.ipynb)
# ---------------------------------------------------------------------------

CONFIG_FINAL: dict[str, dict[str, Any]] = {
    "clasificacion": {
        "estimador": "ensamble",
        # Rejilla (sección 5): C × nodos del spline del IMC × nodos del resto
        "params_logistica": {"C": 1.0, "nodos_imc": 4, "nodos_resto": 4},
        # Búsqueda aleatoria de 40 combinaciones (sección 5), redondeada a 4 decimales
        "params_boosting": {
            "l2_regularization": 7.076,
            "learning_rate": 0.0213,
            "max_iter": 485,
            "max_leaf_nodes": 6,
            "min_samples_leaf": 122,
        },
        # Máximo F1 sobre probabilidades fuera de pliegue (sección 7.2)
        "umbral": 0.35,
    },
    "regresion": {
        "estimador": "dos_partes",
        "umbral": 0.5,
        "params_clasificador": {},
        "params_regresor": {},
    },
    "clustering": {
        "estimador": "kmeans",
        "k": 5,
    },
}


def construir_final(linea: str) -> Pipeline:
    """Modelo final sin ajustar de una línea, según `CONFIG_FINAL`."""
    if linea not in CONFIG_FINAL:
        raise ValueError(f"La línea {linea!r} aún no tiene modelo final. Disponibles: {list(CONFIG_FINAL)}")
    cfg = CONFIG_FINAL[linea]
    if linea == "clustering":
        if cfg["estimador"] != "kmeans":
            raise ValueError(f"Estimador desconocido en CONFIG_FINAL: {cfg['estimador']!r}")
        return construir_clustering(kmeans(cfg["k"]))
    if linea == "regresion":
        if cfg["estimador"] == "dos_partes":
            estimador = dos_partes(cfg["umbral"], cfg["params_clasificador"], cfg["params_regresor"])
        elif cfg["estimador"] == "boosting_poisson":
            estimador = boosting_regresion("poisson", **cfg["params_regresor"])
        else:
            raise ValueError(f"Estimador desconocido en CONFIG_FINAL: {cfg['estimador']!r}")
        return construir_regresion(estimador)
    if cfg["estimador"] == "ensamble":
        estimador = ensamble(cfg["params_logistica"], cfg["params_boosting"])
    elif cfg["estimador"] == "logistica_splines":
        estimador = logistica_splines(**cfg["params_logistica"])
    elif cfg["estimador"] == "gradient_boosting":
        estimador = gradient_boosting(**cfg["params_boosting"])
    else:
        raise ValueError(f"Estimador desconocido en CONFIG_FINAL: {cfg['estimador']!r}")
    return construir_modelo(linea, estimador, umbral=cfg["umbral"])
