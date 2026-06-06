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

## Recomendación

No subas datasets, modelos entrenados ni reportes pesados directamente a Git. Usa Git para código y metadatos, y DVC para los archivos grandes o reproducibles.
