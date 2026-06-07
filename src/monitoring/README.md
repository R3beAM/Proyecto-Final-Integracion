# Monitoreo del modelo y datos

Este módulo separa el monitoreo de **drift de datos** del monitoreo de
**desempeño del modelo**. La comparación se realiza entre un dataset histórico de
referencia y un dataset nuevo o de producción para identificar cambios que puedan
afectar la confiabilidad del modelo de proyección de caída de ventas basado en
reseñas de productos Amazon.

## Estructura

```text
src/monitoring/
├── evidently_reports.py
├── data_drift.py
├── model_performance.py
└── README.md

reports/monitoring/
├── data_drift/
├── model_performance/
└── summary/
```

## Qué variables deberían monitorearse

Se deberían monitorear las variables disponibles en el dataset final y en los
datos productivos, priorizando:

- Variables de reseñas: calificación/rating, texto de la reseña, longitud del
  texto, número de palabras, indicador de texto vacío y fecha de la reseña.
- Variables de interacción: votos útiles, cantidad de reseñas por producto,
  frecuencia mensual o temporal de reseñas.
- Variables de producto: identificador o categoría del producto, agregados por
  producto/mes y señales de riesgo o caída de ventas si están disponibles.
- Variables de inferencia: predicción del modelo, probabilidad o score, etiqueta
  real cuando se consiga retroalimentación, y errores de clasificación.

## Qué tipo de drift se pretende detectar

El monitoreo de datos busca detectar principalmente:

- **Data drift univariado:** cambios en la distribución de variables numéricas o
  categóricas, por ejemplo cambios en ratings, votos útiles o categorías.
- **Drift en variables derivadas de texto:** cambios en longitud de reseñas,
  número de palabras o proporción de textos vacíos.
- **Target/prediction drift:** cambios en la distribución de etiquetas reales o
  predicciones cuando esas columnas estén disponibles.
- **Cambios de calidad de datos:** aumento de valores faltantes, columnas
  inesperadas, tipos inconsistentes o pérdida de variables relevantes.

El monitoreo de desempeño se revisa por separado para identificar degradación en
métricas como accuracy, precision, recall, F1 y matriz de confusión.

## Dataset de referencia

El dataset de referencia debería ser el conjunto histórico usado para entrenar o
validar la versión vigente del modelo. En este proyecto, una opción razonable es
usar `data/processed/final_dataset_with_risk.csv` o el dataset procesado que se
haya utilizado para entrenar el modelo registrado en MLflow.

## Dataset nuevo o de producción

El dataset nuevo o de producción debería contener las reseñas e inferencias más
recientes, preparadas con el mismo pipeline de variables que el dataset de
referencia. Puede corresponder a un lote diario, semanal o mensual de reseñas,
predicciones y, cuando estén disponibles, etiquetas reales posteriores.

## Reportes que se generarían

- `reports/monitoring/data_drift/`: reportes HTML de drift y calidad de datos
  generados por `data_drift.py`.
- `reports/monitoring/model_performance/`: reportes HTML de desempeño del modelo
  generados por `model_performance.py` cuando existan etiqueta real y predicción.
- `reports/monitoring/summary/`: resúmenes Markdown para documentar la corrida,
  datasets comparados, reportes generados y notas de revisión.

## Frecuencia sugerida

- **Drift de datos:** semanal si entran reseñas con frecuencia, o mensual si los
  datos se actualizan por lotes. Esta frecuencia permite detectar cambios de
  comportamiento antes de que se acumulen errores.
- **Desempeño del modelo:** mensual o cada vez que se reciban suficientes
  etiquetas reales nuevas. No conviene medirlo sin volumen mínimo porque las
  métricas podrían ser inestables.
- **Resumen ejecutivo:** después de cada corrida de monitoreo para conservar la
  evidencia de decisiones de revisión o reentrenamiento.

## Uso para decidir revisión o reentrenamiento

Los reportes se usarían como señales de alerta. El modelo debería revisarse si:

- Evidently marca drift significativo en variables críticas como rating, votos
  útiles, variables temporales, señales de riesgo o variables derivadas del texto.
- Aumentan valores faltantes o aparecen cambios de esquema en columnas usadas por
  el modelo.
- La distribución de predicciones cambia de forma abrupta sin explicación de
  negocio.
- Las métricas de desempeño caen por debajo del umbral aceptado, por ejemplo una
  reducción sostenida de F1, precision o recall frente al periodo de referencia.

Si las alertas se repiten durante varios periodos o afectan variables clave, se
debería auditar el pipeline, analizar ejemplos recientes, actualizar el dataset
de entrenamiento y evaluar el reentrenamiento del modelo.

## Ejemplos de ejecución

Reporte de drift de datos:

```bash
python src/monitoring/data_drift.py \
  --reference data/processed/final_dataset_with_risk.csv \
  --current data/processed/final_dataset_with_risk.csv \
  --output reports/monitoring/data_drift/data_drift_example.html \
  --summary reports/monitoring/summary/data_drift_example.md
```

Reporte de desempeño del modelo, si existen columnas de etiqueta y predicción:

```bash
python src/monitoring/model_performance.py \
  --reference data/reference_scored.csv \
  --current data/production_scored.csv \
  --target-column target \
  --prediction-column prediction \
  --output reports/monitoring/model_performance/model_performance_example.html \
  --summary reports/monitoring/summary/model_performance_example.md
```
