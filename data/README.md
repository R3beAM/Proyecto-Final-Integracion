# Datos versionados con DVC

Esta carpeta reserva la estructura de datos del proyecto para ser administrada con **DVC**:

- `raw/`: datos originales sin modificar.
- `external/`: datos externos descargados o recibidos de terceros.
- `interim/`: datos intermedios generados durante limpieza o preparación.
- `processed/`: datasets finales listos para entrenamiento, evaluación o monitoreo.
- `features/`: matrices o tablas de características derivadas.

Los archivos de datos se excluyen de Git mediante `.gitignore`. Para versionarlos, usa DVC, por ejemplo:

```bash
dvc add data/raw/amazon_reviews.csv
git add data/raw/amazon_reviews.csv.dvc data/.gitignore
git commit -m "Track raw Amazon reviews with DVC"
```
