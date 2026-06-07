# Guía DVC del proyecto

Esta guía describe la estructura inicial creada para administrar datos y artefactos del proyecto con **Data Version Control (DVC)**.

## Estructura

```text
.dvc/                 Configuración local del repositorio DVC.
.dvcignore            Patrones que DVC debe ignorar.
data/                 Datos del proyecto, organizados por etapa.
models/               Modelos entrenados y artefactos binarios.
```

## Flujo sugerido

1. Instalar DVC en el entorno de trabajo:

   ```bash
   pip install dvc
   ```

2. Agregar datos o artefactos pesados:

   ```bash
   dvc add data/raw/amazon_reviews.csv
   dvc add models/sentiment_model.pkl
   ```

3. Versionar en Git únicamente los metadatos generados por DVC:

   ```bash
   git add .dvc .dvcignore data models
   git commit -m "Add DVC project structure"
   ```

4. Configurar un remoto DVC cuando exista almacenamiento compartido:

   ```bash
   dvc remote add -d storage <url-del-remoto>
   dvc push
   ```

## Archivos de pipeline conceptual

La carpeta también incluye evidencia de inicialización conceptual del flujo DVC:

- [`dvc.yaml`](dvc.yaml) define una etapa `train_sentiment_models` para reproducir el entrenamiento y registrar artefactos pickle con MLflow cuando los datos versionados estén disponibles.
- [`params.yaml`](params.yaml) centraliza rutas, columnas y parámetros de entrenamiento para mantener el flujo reproducible.

> Nota: las rutas usan referencias relativas desde `docs/dvc/`. Antes de ejecutar `dvc repro docs/dvc/dvc.yaml`, agrega los datasets reales con `dvc add` y verifica que existan las columnas configuradas.

## Recomendación

No subas datasets, modelos entrenados ni reportes pesados directamente a Git. Usa Git para código y metadatos, y DVC para los archivos grandes o reproducibles.
