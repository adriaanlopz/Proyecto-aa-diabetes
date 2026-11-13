# Proyecto de Aprendizaje Automático: encuesta de salud y diabetes

Proyecto de la asignatura de Aprendizaje Automático (UIE). A partir de la encuesta anual de salud de un consorcio de atención primaria (`proy_diabetes_dev.csv`) se abordan tres líneas de trabajo:

| Línea | Tarea | Objetivo | Estado |
|---|---|---|---|
| Clusterización | Perfiles de paciente interpretables | — | **Modelada y servida en la API** |
| Clasificación | Cribado de diabetes | `diabetes` (sí/no) | **Modelada y servida en la API** |
| Regresión | Carga de enfermedad | `dias_salud_fisica_mala` (0–30) | **Modelada y servida en la API** |

## Estado

**Punto de control 1 (S09)**
- [x] Análisis exploratorio completo → `notebooks/01_eda.ipynb`
- [x] Código de preprocesado → `src/preprocesamiento.py`
- [x] Versión preprocesada de los datos → `data/processed/` y `models/preprocesadores/`

**Punto de control 2 (S12)**
- [x] Revisión pedida en el PC1: objetivos cruzados como predictores → descartados con evidencia (decisión #13)
- [x] Clasificación modelada → `notebooks/03_modelado_clasificacion.ipynb` y `src/modelado.py`
- [x] Artefacto final reproducible → `python src/entrenar.py clasificacion` → `artifacts/`
- [x] API común respondiendo para `clasificacion` → `src/servicio.py`, con 16 pruebas en `tests/`
- [x] ZIP de entrega con `data/processed/` y artefactos → `python src/empaquetar.py`
- [x] Regresión modelada y servida → `notebooks/04_modelado_regresion.ipynb` y `python src/entrenar.py regresion`
- [x] Clusterización modelada y servida → `notebooks/05_clusterizacion.ipynb` y `python src/entrenar.py clusterizacion`

## Resultados: clasificación (`diabetes`)

Modelo: **ensamble por voto suave** de una regresión logística con *splines* y un *gradient boosting* poco profundo. Umbral de decisión: **0,35**.

| | CV en train¹ | Test (n = 2 550) · IC 95 % | Dummy |
|---|---|---|---|
| ROC-AUC | 0,828 | **0,826** [0,809–0,842] | 0,500 |
| PR-AUC | 0,680 | 0,683 [0,646–0,718] | 0,333 |
| Brier | 0,157 | 0,158 [0,150–0,166] | 0,222 |
| Sensibilidad (umbral 0,35) | 0,773 | **0,759** [0,727–0,789] | 0 |
| Especificidad (umbral 0,35) | 0,729 | 0,726 [0,705–0,746] | 1 |

¹ ROC-AUC, PR-AUC y Brier: media de la CV repetida 5 × 5. Sensibilidad y especificidad: predicciones fuera de pliegue de la CV de 5 pliegues, con las que se eligió el umbral.

Detecta 644 de los 848 diabéticos de test. Uno de cada 1,7 pacientes citados es diabético, frente a uno de cada 3 si se citara al azar. Las variables más influyentes son salud percibida, IMC, edad, hipertensión y colesterol. Detalle completo y figuras 21–28 en el notebook 03.

## Resultados: regresión (`dias_salud_fisica_mala`)

Modelo: **dos partes** (*hurdle*). Un *gradient boosting* estima la probabilidad de tener algún día de mala salud física; si es de 0,5 o más, otro *gradient boosting* predice la mediana de los días; si no, predice 0. Predicciones recortadas a [0, 30].

| | CV repetida 5 × 5 | Test (n = 2 550) · IC 95 % | Dummy mediana (0) | Dummy media |
|---|---|---|---|---|
| MAE (días) | 3,70 | **3,72** [3,44–3,97] | 5,00 | 6,86 |
| RMSE (días) | 7,48 | 7,64 [7,20–8,07] | 10,69 | 9,45 |
| R² | 0,37 | 0,35 [0,28–0,41] | −0,28 | 0,00 |

Mejora un 26 % el MAE de la mejor referencia. Las variables que más pesan son la salud percibida, la dificultad para caminar, los días de mala salud mental y la edad. Detalle completo y figuras 29–31 en el notebook 04.

**Decisión del equipo: dos partes (mediana).** El MAE premia predecir 0 (60 % de ceros). El modelo de la media (boosting de Poisson) tiene peor MAE (4,28) pero mejor RMSE (6,96) y R² (0,46), y acierta el total de días (el de dos partes lo subestima un 12 %). Se preguntará al profesor la métrica de la fase competitiva; si fuera el RMSE, cambiar de modelo es una línea en `modelado.CONFIG_FINAL`.

## Resultados: clusterización (perfiles de paciente)

Modelo: **K-means con 5 grupos** sobre 10 variables de salud y hábitos, imputadas y estandarizadas. Los objetivos y las sociodemográficas no se usan para formar los grupos, solo para describirlos. Estabilidad entre mitades de los datos: ARI = 0,85.

| Grupo | Pacientes | Perfil en una frase | Diabetes¹ | Días salud física¹ |
|---|---|---|---|---|
| 1 · Sanos y activos | 39 % | Los más jóvenes, sin hipertensión, activos y con buena salud percibida | 10 % | 1,5 |
| 2 · Hipertensos sin síntomas | 30 % | Mayores con hipertensión y colesterol, pero que se encuentran bien | 39 % | 2,8 |
| 4 · Cardiópatas | 11 % | Los de más edad, todos con cardiopatía; más hombres | 60 % | 8,3 |
| 0 · Movilidad reducida y obesidad | 12 % | Dificultad para caminar (99 %), IMC medio 32,6 y salud regular o mala | 55 % | 11,1 |
| 3 · Malestar psicológico | 8 % | 27 días al mes de mala salud mental y la peor salud percibida | 51 % | 16,3 |

¹ No se usaron para formar los grupos: que los separen tanto valida que los perfiles tienen sentido clínico. Detalle completo y figuras 32–34 en el notebook 05.

## Estructura

```
Proyecto/
├── proy_diabetes_dev.csv           datos crudos (no se modifican)
├── requirements.txt
├── src/
│   ├── preprocesamiento.py         limpieza, variables derivadas, pipelines y particiones
│   ├── modelado.py                 catálogo de modelos, evaluación (CV, IC bootstrap) y CONFIG_FINAL
│   ├── entrenar.py                 reentrena la configuración final y guarda el artefacto de la API
│   ├── servicio.py                 API común (plantilla del profesor, sin modificar)
│   ├── empaquetar.py               genera el ZIP de entrega y comprueba que no falte nada
│   └── eda_utils.py                estilo gráfico, estadísticos y gráficos
├── notebooks/
│   ├── 01_eda.ipynb                EDA (resumen ejecutivo, 10 secciones, figuras 01–20)
│   ├── 02_preprocesado.ipynb       ejecución del preprocesado y comprobaciones
│   ├── 03_modelado_clasificacion.ipynb   selección, ajuste, umbral, test e interpretación (figuras 21–28)
│   ├── 04_modelado_regresion.ipynb       métrica, comparación, ajuste, test e interpretación (figuras 29–31)
│   └── 05_clusterizacion.ipynb           variables, algoritmo, número de grupos y perfiles (figuras 32–34)
├── data/processed/                 datos limpios y particiones transformadas (generado por el notebook 02)
├── models/preprocesadores/         preprocesadores ajustados solo con train (notebook 02)
├── artifacts/
│   ├── proy_modelo_clasificacion.joblib   modelo que carga la API (datos crudos → predicción)
│   ├── proy_modelo_clasificacion.json     metadatos: versión, columnas, configuración y métricas
│   ├── proy_modelo_regresion.joblib       modelo de regresión que carga la API
│   ├── proy_modelo_regresion.json         sus metadatos
│   ├── proy_modelo_clusterizacion.joblib  modelo de clusterización que carga la API
│   └── proy_modelo_clusterizacion.json    sus metadatos
├── reports/
│   ├── figures/                    figuras para informe y slides (PNG, 150 dpi)
│   ├── metricas_clasificacion.json métricas de CV y test (las genera el notebook 03)
│   ├── metricas_regresion.json     métricas de CV y test (las genera el notebook 04)
│   └── metricas_clustering.json    métricas y perfiles de los grupos (los genera el notebook 05)
├── tests/                          pruebas del contrato de la API (las tres líneas)
└── ejemplos_api/peticion_<linea>.json     peticiones de demostración de las tres líneas (5 pacientes ficticios)
```

## Cómo reproducir

```bash
pip install -r requirements.txt

# 1. Análisis, preprocesado y modelado (en este orden)
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocesado.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_modelado_clasificacion.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_modelado_regresion.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/05_clusterizacion.ipynb

# 2. Artefacto final que sirve la API (reentrena con los 12 750 pacientes)
python src/entrenar.py clasificacion
python src/entrenar.py regresion
python src/entrenar.py clusterizacion

# 3. Pruebas del servicio
python -m pytest tests -q

# 4. ZIP de entrega (incluye data/processed, models y artifacts)
python src/empaquetar.py --nombre PC2
```

Todo usa una semilla fija (`SEMILLA = 42`), así que el resultado es reproducible. El notebook 03 termina con 10 comprobaciones (`assert`) sobre el modelo que se va a servir. Además verifica que `modelado.CONFIG_FINAL` coincide con lo que ha decidido, para que notebook y artefacto no se desincronicen.

## API

El servicio implementa el contrato común con la plantilla del profesor, sin cambios. Carga los artefactos de `artifacts/` y sirve las líneas cuyo artefacto existe; ahora mismo, las tres.

```bash
uvicorn src.servicio:app --host 0.0.0.0 --port 8000
```

| Endpoint | Respuesta |
|---|---|
| `GET /health` | `{"estado": "ok", "lineas": ["clasificacion", "clusterizacion", "regresion"], "modelo_version": "1.0.0"}` |
| `POST /predict` | `{"linea", "modelo_version", "predicciones": [...]}`, una por instancia: `{"clase": 0/1, "probabilidad": p}` en clasificación, los días estimados (0 a 30) en regresión y el número de grupo (0 a 4) en clusterización |
| `GET /docs` | Documentación interactiva (Swagger) para probarla desde el navegador |

Demostración con la petición de ejemplo:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary @ejemplos_api/peticion_clasificacion.json
```

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/predict -Method Post -ContentType "application/json; charset=utf-8" -InFile ejemplos_api/peticion_clasificacion.json
```

Cambiando `clasificacion` por `regresion` o `clusterizacion` en el nombre del fichero se prueban las otras dos líneas. Los pacientes son ficticios: traen columnas personales inventadas, codificaciones mezcladas y valores ausentes, para enseñar que el servicio los tolera.

**Qué acepta.** Instancias **crudas**, tal y como vienen en el CSV:
- Las columnas sobrantes se ignoran: PII, id, `usuario_alta_datos` y los objetivos.
- Valores en cualquiera de las codificaciones vistas (`si/SI/True/1.0`…) o no vistas (`Sí`, `YES`).
- Nulos y valores imposibles, que se tratan como ausentes y se imputan. **Nunca se descarta una instancia.**
- Solo devuelve 422 si una de las 18 columnas que usa el modelo no aparece en *ninguna* instancia, o si la línea no está servida.

**Detalle técnico.** Los artefactos referencian `preprocesamiento` y `modelado` como módulos de primer nivel. Es lo mismo que resuelve el servicio al añadir `src/` a `sys.path`, así que el artefacto carga desde cualquier directorio de trabajo; hay una prueba que lo verifica.

## Registro de decisiones

| # | Hallazgo | Decisión |
|---|---|---|
| 1 | 7 columnas PII ofuscadas (`dni`, `nombre`, …) | Eliminar: sin valor predictivo y sujetas a protección de datos |
| 2 | `usuario_alta_datos` independiente de ambos objetivos (χ² p = 0,48; V ≈ 0) | Eliminar: columna de gestión sin señal ni riesgo de fuga |
| 3 | `id_paciente_proy` | Índice, nunca feature |
| 4 | 4 binarias con hasta 8 codificaciones (`si/SI/True/1.0`…), ~25 % de registros | Unificar a 0/1; un valor irreconocible pasa a nulo (la API nunca falla) |
| 5 | Ordinales con gradientes monótonos con los objetivos | Codificación ordinal (enteros) |
| 6 | 8 % de filas con nulos: bloque de hábitos (376) + patrón anidado de abandono (267 ⊂ 397 ⊂ 657); compatibles con MCAR | Imputar mediana/moda dentro del pipeline (ajustado solo en train) + indicadores `falta_habitos`, `falta_imc` |
| 7 | IMC asimétrico con outliers clínicamente reales (obesidad III) | Conservar; añadir `imc_categoria` (OMS) |
| 8 | Factores de riesgo CV muy asociados a la diabetes | Variable `n_factores_riesgo_cv` (V = 0,38 con diabetes) |
| 9 | Sin multicolinealidad entre los predictores originales (VIF ≤ 1,7) | Se mantienen todos; en modelos lineales, usar variable derivada **o** sus componentes |
| 10 | `diabetes` remuestreada al 33 % | Partición estratificada; probabilidades no poblacionales (limitación) |
| 11 | `dias_salud_fisica_mala`: 60 % ceros, 9 % en 30, sobredispersión ×18, *heaping* | Estratificar por tramos 0 / 1–29 / 30; métrica MAE; valorar modelos *hurdle*/Tweedie |
| 12 | Clustering: continuo dominado por "carga de enfermedad" y "edad cardiometabólica vs malestar" | Variables de salud y hábitos estandarizadas (10 desde la decisión #25); sociodemográficas solo para describir |
| 13 | **Revisión del PC1.** El objetivo de una línea como predictor de la otra no aporta nada: ΔROC-AUC = 0,0000 con `dias_salud_fisica_mala` en clasificación, ΔMAE = −0,002 días con `diabetes` en regresión (CV pareada) | **Descartados en ambas líneas.** En cribado el diagnóstico es justo lo desconocido, y exigirlos en la API arriesgaba un 422. Clustering los conserva (describen el perfil) |
| 14 | El modelo solo necesita 18 columnas crudas | Se ajusta con `pp.columnas_crudas`: `feature_names_in_` pide solo eso y el servicio descarta el resto |
| 15 | Ranking de pacientes para citar → la métrica relevante es de orden; la API devuelve probabilidad | Selección por **ROC-AUC**; calidad de la probabilidad con Brier/log-loss; exactitud nunca (el 67 % es trivial) |
| 16 | 7 familias en CV: todas las razonables entre 0,816 y 0,825, diferencias < 1 de entre pliegues | KNN y RF descartados; se ajustan logística + splines (rejilla) y boosting (búsqueda aleatoria) |
| 17 | En CV repetida 5 × 5, el ensamble supera a la logística simple en 24/25 particiones (+0,004 AUC). Regla fijada de antemano: ≥ 80 % | **Ensamble por voto suave** (logística + splines y boosting) |
| 18 | Probabilidades ya calibradas; la isotónica no mejora el Brier | No se recalibra |
| 19 | Con umbral 0,5 solo se detecta el 58 % de los diabéticos; máximo F1 y Youden coinciden en 0,35 | **Umbral 0,35** (sensibilidad 0,77), fijado dentro del pipeline con `FixedThresholdClassifier` |
| 20 | Configuración cerrada con CV y validada una vez en test | El artefacto final se reentrena con los 12 750 pacientes (train + test) |
| 21 | En el PC1 el ZIP llegó sin `data/processed/` ni artefactos | `src/empaquetar.py` los incluye explícitamente y falla si falta algún bloque |
| 22 | Regresión: el boosting con error absoluto predice 0 a todos; los modelos de la media empatan en MAE ≈ 4,27 | Para la mediana, **modelo de dos partes** (¿algún día? y ¿cuántos?) |
| 23 | El MAE del modelo de dos partes baja al subir el umbral, pero solo por predecir más ceros (media predicha de 4,4 a 3,4 días) | **Umbral 0,5 fijo**: es el que corresponde a la mediana por definición, no se ajusta a la métrica |
| 24 | En CV 5 × 5, dos partes gana al boosting de Poisson en MAE en 25/25 particiones (−0,58 días); Poisson gana en RMSE y R² | **Dos partes con mediana**, por la métrica principal (MAE). Pendiente de confirmar la métrica de la fase competitiva |
| 25 | Clustering sin los objetivos: mejor silueta (0,185 frente a 0,143) y Davies-Bouldin (1,76 frente a 2,07) con 5 grupos, estabilidad parecida | **Se forman sin `diabetes` ni `dias_salud_fisica_mala`**: quedan como validación externa y la API no los exige |
| 26 | Siluetas bajas en todo (0,16–0,24): un continuo. GMM inestable desde 3 grupos (ARI ≤ 0,83); K-means estable (0,81–0,97) | **K-means con 5 grupos**: mejor separación que con 4 y un perfil nuevo con sentido clínico (malestar psicológico) |
| 27 | `anadir_features` exigía `ictus`, que el clustering no usa, y el modelo no podía entrenarse solo con sus columnas | Cada variable derivada se calcula solo si están sus columnas de origen |

## Limitaciones conocidas

- La prevalencia de diabetes (33 %) es artificial por el remuestreo: no admite lecturas epidemiológicas y las probabilidades del clasificador no son poblacionales. Para el uso real habría que corregir el *prior* y reajustar el umbral.
- Todos los datos son autorreportados (sesgo de recuerdo, deseabilidad social, redondeo de días). Los cuatro tipos de modelo se quedan en ROC-AUC ≈ 0,82–0,83: el techo parece estar en la información disponible, no en el algoritmo.
- Diseño transversal: las asociaciones no son causales (p. ej. la mala salud percibida es en parte consecuencia de la diabetes).
- El servicio sigue la plantilla del profesor (`servicio.py`). Falta confirmar con el enunciado ampliado (contrato publicado en S07) qué columnas incluye cada petición y la métrica de la fase competitiva. Si la métrica dependiera de la `clase`, el umbral se ajusta en `modelado.CONFIG_FINAL` sin tocar el modelo.
