import numpy as np
import pytest
from conftest import RAIZ, a_instancias

import preprocesamiento as pp

ARTEFACTO_REGRESION = RAIZ / "artifacts" / "proy_modelo_regresion.joblib"
DIAS_MIN, DIAS_MAX = pp.RANGO_DIAS


@pytest.fixture(scope="module")
def modelo_regresion():
    if not ARTEFACTO_REGRESION.is_file():
        pytest.skip("Falta el artefacto: ejecutar antes `python src/entrenar.py regresion`")
    import joblib

    return joblib.load(ARTEFACTO_REGRESION)


@pytest.fixture(scope="module")
def test_regresion(crudo):
    _, X_test, _, _ = pp.dividir(crudo, "regresion")
    return X_test


def pedir(cliente, instancias):
    return cliente.post("/predict", json={"linea": "regresion", "instancias": instancias})


def test_una_prediccion_en_rango_por_instancia(cliente, modelo_regresion, test_regresion):
    instancias = a_instancias(test_regresion.head(200))
    r = pedir(cliente, instancias)
    assert r.status_code == 200, r.text
    cuerpo = r.json()
    assert cuerpo["linea"] == "regresion"
    assert len(cuerpo["predicciones"]) == len(instancias)
    assert all(isinstance(p, float) and DIAS_MIN <= p <= DIAS_MAX for p in cuerpo["predicciones"])


def test_la_api_devuelve_lo_mismo_que_el_modelo(cliente, modelo_regresion, test_regresion):
    muestra = test_regresion.head(100)
    r = pedir(cliente, a_instancias(muestra))
    esperado = modelo_regresion.predict(muestra[list(modelo_regresion.feature_names_in_)])
    assert np.allclose(r.json()["predicciones"], esperado)


def test_no_exige_diabetes_ni_el_objetivo(cliente, modelo_regresion, test_regresion):
    instancias = a_instancias(test_regresion.head(10).drop(columns=[pp.TARGET_CLASIF]))
    r = pedir(cliente, instancias)
    assert r.status_code == 200, r.text
    assert len(r.json()["predicciones"]) == 10


def test_nulos_y_codificaciones_mezcladas(cliente, modelo_regresion, test_regresion):
    muestra = test_regresion.head(5).copy()
    muestra["imc"] = None
    muestra["fumador"] = ["SI", "True", "1.0", "no", None]
    muestra["salud_general"] = ["Mala", "EXCELENTE", None, "buena", "regular"]
    r = pedir(cliente, a_instancias(muestra))
    assert r.status_code == 200, r.text
    assert len(r.json()["predicciones"]) == 5


def test_artefacto_pide_solo_columnas_crudas_sin_pii_ni_objetivos(modelo_regresion):
    columnas = set(modelo_regresion.feature_names_in_)
    prohibidas = set(pp.COLS_PII + pp.COLS_GESTION + [pp.COL_ID, pp.TARGET_CLASIF, pp.TARGET_REGR])
    assert not columnas & prohibidas
    assert list(modelo_regresion.feature_names_in_) == pp.columnas_crudas("regresion")
