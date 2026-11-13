import json

import pytest
from conftest import RAIZ

EJEMPLOS = RAIZ / "ejemplos_api"


@pytest.mark.parametrize("linea", ["clasificacion", "regresion", "clusterizacion"])
def test_las_peticiones_de_ejemplo_funcionan(cliente, linea):
    peticion = json.loads((EJEMPLOS / f"peticion_{linea}.json").read_text(encoding="utf-8"))
    r = cliente.post("/predict", json=peticion)
    assert r.status_code == 200, r.text
    assert len(r.json()["predicciones"]) == len(peticion["instancias"])
