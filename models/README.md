# Modelos versionados con DVC

Esta carpeta está destinada a guardar modelos entrenados, vectorizadores y otros artefactos binarios del sistema de análisis de sentimiento.

Los artefactos reales se excluyen de Git y deben registrarse con DVC, por ejemplo:

```bash
dvc add models/sentiment_model.pkl
git add models/sentiment_model.pkl.dvc models/.gitignore
git commit -m "Track sentiment model with DVC"
```

## Relación con MLflow

Las corridas experimentales pueden generarse desde `mlflow_tracking/train_and_log_models.py`. Ese flujo guarda modelos pickle con nombres únicos por corrida en `mlflow_tracking/models_pickle/` y los registra como artefactos MLflow; cuando un modelo quede seleccionado para producción, copie o versione el `.pkl` correspondiente mediante DVC.
