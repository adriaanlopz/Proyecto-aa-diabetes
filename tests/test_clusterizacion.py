import numpy as np
import pytest
from conftest import RAIZ, a_instancias

import modelado as md
import preprocesamiento as pp

ARTEFACTO_CLUSTERING = RAIZ / "artifacts" / "proy_modelo_clusterizacion.joblib"
K = md.CONFIG_FINAL["clustering"]["k"]


@pytest.fixture(scope="module")
def modelo_clustering():
    if not ARTEFACTO_CLUSTERING.is_file():
        pytest.skip("Falta el artefacto: ejecutar antes `python src/entrenar.py clusterizacion`")
    import joblib

    return joblib.load(ARTEFACTO_CLUSTERING)


def pedir(cliente, instancias):
    return cliente.post("/predict", json={"linea": "clusterizacion", "instancias": instancias})


def test_un_grupo_entero_por_instancia(cliente, modelo_clustering, crudo):
    instancias = a_instancias(crudo.sample(200, random_state=pp.SEMILLA))
    r = pedir(cliente, instancias)
    assert r.status_code == 200, r.text
    cuerpo = r.json()
    assert cuerpo["linea"] == "clusterizacion"
    assert len(cuerpo["predicciones"]) == len(instancias)
    assert all(isinstance(g, int) and 0 <= g < K for g in cuerpo["predicciones"])


def test_la_api_devuelve_lo_mismo_que_el_modelo(cliente, modelo_clustering, crudo):
    muestra = crudo.sample(100, random_state=1)
    r = pedir(cliente, a_instancias(muestra))
    esperado = modelo_clustering.predict(muestra[list(modelo_clustering.feature_names_in_)])
    assert r.json()["predicciones"] == [int(g) for g in esperado]


def test_no_exige_los_objetivos(cliente, modelo_clustering, crudo):
    instancias = a_instancias(crudo.head(10).drop(columns=[pp.TARGET_CLASIF, pp.TARGET_REGR]))
    r = pedir(cliente, instancias)
    assert r.status_code == 200, r.text
    assert len(r.json()["predicciones"]) == 10


def test_nulos_y_codificaciones_mezcladas(cliente, modelo_clustering, crudo):
    muestra = crudo.head(5).copy()
    muestra["imc"] = None
    muestra["presion_alta"] = ["SI", "True", "1.0", "no", None]
    muestra["salud_general"] = ["Mala", "EXCELENTE", None, "buena", "regular"]
    r = pedir(cliente, a_instancias(muestra))
    assert r.status_code == 200, r.text
    assert len(r.json()["predicciones"]) == 5


def test_artefacto_pide_solo_columnas_crudas_sin_pii_ni_objetivos(modelo_clustering):
    prohibidas = set(pp.COLS_PII + pp.COLS_GESTION + [pp.COL_ID, pp.TARGET_CLASIF, pp.TARGET_REGR])
    assert not set(modelo_clustering.feature_names_in_) & prohibidas
    assert list(modelo_clustering.feature_names_in_) == pp.columnas_crudas("clustering")


def test_los_grupos_no_cambian_al_reentrenar(modelo_clustering, crudo):
    X = crudo[pp.columnas_crudas("clustering")]
    nuevo = md.construir_final("clustering").fit(X)
    assert np.array_equal(nuevo.predict(X), modelo_clustering.predict(X))
