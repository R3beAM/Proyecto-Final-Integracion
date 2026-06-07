# Evidencia de elementos candidatos a versionar con DVC

Este documento deja evidencia, dentro de la ruta `.dvc/`, de los archivos y carpetas del proyecto que deberían administrarse con **DVC** por ser datos, artefactos reproducibles o binarios pesados que no conviene versionar directamente en Git.

## Criterio de versionado

Se propone usar Git para código fuente, notebooks livianos, documentación y metadatos de DVC (`*.dvc`, `dvc.yaml`, `params.yaml`, `dvc.lock`). DVC debería administrar los archivos que cumplan al menos una de estas condiciones:

- Datasets originales o transformados necesarios para reproducir entrenamiento, evaluación o monitoreo.
- Artefactos de modelos entrenados en formatos binarios como `.pkl`.
- Reportes o salidas reproducibles que puedan ser pesadas, como PDF, HTML, imágenes o resultados de monitoreo.
- Archivos generados por pipelines que deban conservar trazabilidad entre versiones.

## Elementos identificados en este repositorio

| Categoría | Ruta candidata | Motivo | Comando DVC sugerido |
| --- | --- | --- | --- |
| Datos crudos | `data/raw/` | Conserva insumos originales sin modificar para trazabilidad. | `dvc add data/raw/<archivo>` |
| Datos externos | `data/external/` | Registra fuentes recibidas o descargadas de terceros. | `dvc add data/external/<archivo>` |
| Datos intermedios | `data/interim/monthly_product.csv` | Resultado de transformaciones previas al dataset final. | `dvc add data/interim/monthly_product.csv` |
| Datos procesados | `data/processed/final_dataset_ohe.csv` | Dataset final transformado para experimentación. | `dvc add data/processed/final_dataset_ohe.csv` |
| Datos procesados | `data/processed/final_dataset_remove_duplicates.csv` | Dataset depurado con duplicados removidos. | `dvc add data/processed/final_dataset_remove_duplicates.csv` |
| Datos procesados | `data/processed/final_dataset_robust.csv` | Dataset robusto usado como entrada conceptual del pipeline DVC. | `dvc add data/processed/final_dataset_robust.csv` |
| Datos procesados | `data/processed/final_dataset_with_risk.csv` | Dataset enriquecido con variable o indicador de riesgo. | `dvc add data/processed/final_dataset_with_risk.csv` |
| Features | `data/features/product_features.csv` | Tabla de características derivadas para modelado. | `dvc add data/features/product_features.csv` |
| Features | `data/features/product_product_month.csv` | Agregación producto-mes generada para análisis/modelado. | `dvc add data/features/product_product_month.csv` |
| Modelos entrenados | `mlflow_tracking/models_pickle/best_random_search_model.pkl` | Modelo binario seleccionado como mejor corrida. | `dvc add mlflow_tracking/models_pickle/best_random_search_model.pkl` |
| Modelos entrenados | `mlflow_tracking/models_pickle/modelo_base_pipeline.pkl` | Modelo base serializado en pickle. | `dvc add mlflow_tracking/models_pickle/modelo_base_pipeline.pkl` |
| Tracking de experimentos | `mlflow_tracking/mlruns/` | Artefactos y metadatos locales de corridas MLflow que pueden crecer rápidamente. | `dvc add mlflow_tracking/mlruns` |
| Reportes de evaluación | `reports/Interpretabilidad_Modelo_Entregable_3.pdf` | Reporte final reproducible en formato binario. | `dvc add reports/Interpretabilidad_Modelo_Entregable_3.pdf` |
| Figuras de reportes | `reports/Figures/` | Imágenes generadas como evidencia de análisis y evaluación. | `dvc add reports/Figures` |
| Reportes de monitoreo | `reports/monitoring/` | Salidas de monitoreo y drift generadas por scripts. | `dvc add reports/monitoring` |

## Evidencia de pipeline ya declarada

El archivo `.dvc/dvc.yaml` declara una etapa conceptual `train_sentiment_models` que depende de `data.processed_path` y escribe modelos en `training.pickle_dir`. Con los parámetros actuales de `.dvc/params.yaml`, esto evidencia que los siguientes elementos son parte del flujo reproducible:

- Entrada principal del entrenamiento: `data/processed/final_dataset_robust.csv`.
- Script reproducible: `mlflow_tracking/train_and_log_models.py`.
- Salida versionable con DVC: `mlflow_tracking/models_pickle/`.
- Tracking local asociado: `mlflow_tracking/mlruns/`.

## Comandos de evidencia sugeridos

Cuando se quiera materializar el versionado con DVC, ejecutar desde la raíz del repositorio:

```bash
dvc add data/interim/monthly_product.csv
dvc add data/processed/final_dataset_ohe.csv
dvc add data/processed/final_dataset_remove_duplicates.csv
dvc add data/processed/final_dataset_robust.csv
dvc add data/processed/final_dataset_with_risk.csv
dvc add data/features/product_features.csv
dvc add data/features/product_product_month.csv
dvc add mlflow_tracking/models_pickle/best_random_search_model.pkl
dvc add mlflow_tracking/models_pickle/modelo_base_pipeline.pkl
dvc add reports/Interpretabilidad_Modelo_Entregable_3.pdf
dvc add reports/Figures
```

Después de ejecutar `dvc add`, versionar en Git solo los metadatos generados por DVC:

```bash
git add .dvc data/**/*.dvc mlflow_tracking/**/*.dvc reports/**/*.dvc
git commit -m "Track project data and artifacts with DVC"
```

> Nota: este archivo es evidencia documental. No reemplaza la ejecución de `dvc add`; sirve para dejar explícito qué elementos del proyecto deberían pasar a ser administrados por DVC.
