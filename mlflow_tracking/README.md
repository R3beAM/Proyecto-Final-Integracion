# Tracking de modelos con MLflow

Esta carpeta agrega una estructura mínima para registrar corridas de modelos con **MLflow** y guardar los modelos entrenados en formato **pickle** (`.pkl`) con nombres distintos por corrida.

## Estructura

- `train_and_log_models.py`: script para entrenar y registrar varios modelos de clasificación de sentimiento.
- `REGISTRO_MLFLOW.md`: documento que explica cómo registrar nombre de experimento, parámetros, métricas, artefactos y modelo final en MLflow.
- `mlruns/`: carpeta local de tracking de MLflow. Sus artefactos generados se ignoran en Git.
- `models_pickle/`: carpeta donde se escriben los modelos `.pkl` generados. Sus archivos pickle se ignoran en Git y pueden versionarse con DVC si se desean conservar.

## Instalación sugerida

```bash
pip install mlflow pandas scikit-learn
```

## Ejecución con datos de ejemplo

```bash
python mlflow_tracking/train_and_log_models.py
```

El comando ejecuta dos modelos con nombres diferenciados por tipo de modelo y timestamp UTC con microsegundos:

- `tfidf_logistic_regression_<timestamp>.pkl`
- `tfidf_multinomial_nb_<timestamp>.pkl`

Cada archivo se guarda en `mlflow_tracking/models_pickle/` y también se registra como artefacto del run en MLflow bajo `pickle_models/`. El mismo nombre único se usa como `run_name` en MLflow.

## Ejecución con un CSV propio

```bash
python mlflow_tracking/train_and_log_models.py \
  --data data/processed/reviews.csv \
  --text-column review_text \
  --target-column sentiment
```

El CSV debe contener una columna de texto y una columna objetivo. Si los nombres son distintos, ajústelos con `--text-column` y `--target-column`.

## Visualizar las corridas

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlflow_tracking/mlruns
```

Luego abra la URL local que informe MLflow para revisar parámetros, métricas y artefactos pickle.
