# Proyecto-Final-Integracion

Sistema de proyección de caída de ventas aplicado a reseñas históricas de productos Amazon.

## Objetivo general del proyecto

El objetivo general es construir una solución de Machine Learning que permita analizar reseñas históricas de productos de Amazon y estimar señales de riesgo asociadas a una posible caída de ventas. El proyecto busca integrar preparación de datos, entrenamiento de modelos, seguimiento de experimentos, versionado de artefactos y monitoreo para dejar una base reproducible y escalable. Se busca predecir riesgo de caída de ventas de un producto con base en reseñas negativas.

## Descripción breve del problema de Machine Learning

El problema se aborda como una tarea supervisada sobre información histórica de reseñas, productos y variables derivadas. A partir de atributos como calificaciones, texto de reseñas, frecuencia temporal, votos útiles y variables agregadas por producto o mes, el modelo debería aprender patrones que indiquen riesgo o probabilidad de caída de ventas. En términos prácticos, el proyecto puede plantearse como clasificación de riesgo, predicción de sentimiento o generación de una señal preventiva para apoyar decisiones comerciales.

## Estructura de carpetas

```text
.
├── data/                  # Datos del proyecto organizados por etapa.
│   ├── raw/               # Datos originales sin modificar.
│   ├── external/          # Datos externos o de terceros.
│   ├── interim/           # Datos intermedios de limpieza y transformación.
│   ├── processed/         # Datasets finales para entrenamiento/evaluación.
│   └── features/          # Tablas o matrices de características derivadas.
├── docs/dvc/              # Guía y archivos conceptuales para uso de DVC.
├── mlflow_tracking/       # Scripts, documentación y tracking local de MLflow.
├── models/                # Espacio previsto para modelos entrenados y artefactos.
├── Notebooks/             # Notebooks exploratorios y entregables del proyecto.
├── reports/               # Figuras, reportes y salidas de análisis/monitoreo.
└── src/                   # Código fuente reutilizable del proyecto.
```

## Flujo general del proyecto

1. **Ingesta de datos:** recopilar reseñas históricas, datos de productos y fuentes externas relevantes.
2. **Limpieza y preparación:** normalizar columnas, eliminar duplicados, tratar valores faltantes y construir datasets intermedios.
3. **Ingeniería de características:** generar variables de texto, agregaciones temporales, métricas por producto y señales de riesgo.
4. **Entrenamiento y evaluación:** entrenar modelos, comparar métricas y seleccionar la mejor alternativa según el objetivo del negocio.
5. **Tracking de experimentos:** registrar parámetros, métricas, artefactos y modelos con MLflow.
6. **Versionado:** controlar datasets, modelos y artefactos pesados con DVC, manteniendo en Git el código y los metadatos.
7. **Monitoreo:** comparar datos históricos contra datos nuevos con Evidently para detectar drift, problemas de calidad y degradación de desempeño.
8. **Preparación para despliegue:** organizar el modelo final, su pipeline de transformación, dependencias y documentación para una futura puesta en producción.

## Uso previsto de DVC

DVC se utilizaría para versionar datasets, modelos entrenados y otros artefactos pesados que no deben subirse directamente a Git. El repositorio ya incluye una estructura inicial en `.dvc/`, `.dvcignore`, `data/`, `models/` y `docs/dvc/`.

El flujo previsto sería:

```bash
dvc add data/raw/amazon_reviews.csv
dvc add data/processed/final_dataset_with_risk.csv
dvc add models/modelo_final.pkl
git add data/raw/amazon_reviews.csv.dvc data/processed/final_dataset_with_risk.csv.dvc models/modelo_final.pkl.dvc
git commit -m "Version data and model artifacts with DVC"
dvc push
```

De esta forma, Git conserva el historial del código y de los archivos `.dvc`, mientras que DVC administra el almacenamiento real de los artefactos en un remoto compartido.

## Uso previsto de MLflow

MLflow se utilizaría para registrar experimentos de entrenamiento, comparar corridas y conservar evidencia de los modelos evaluados. Cada corrida debería incluir:

- Nombre del experimento y del run.
- Parámetros de entrenamiento y configuración de features.
- Métricas de evaluación, por ejemplo accuracy, precision, recall, F1 o métricas específicas del problema.
- Artefactos como modelos pickle, reportes, matrices de confusión o gráficos.
- Referencia al dataset versionado con DVC utilizado en la corrida.

La carpeta `mlflow_tracking/` contiene el script `train_and_log_models.py`, documentación de registro y el espacio local `mlruns/` para tracking. Los modelos pickle generados se guardan en `mlflow_tracking/models_pickle/` y podrían versionarse con DVC si se desean conservar.

## Uso previsto de Evidently para monitoreo

Evidently se utilizaría para monitorear la estabilidad de datos y del modelo cuando existan datos nuevos o de producción. El monitoreo previsto incluye:

- **Data drift:** cambios en distribuciones de variables numéricas, categóricas y derivadas del texto.
- **Data quality:** valores faltantes, tipos inesperados, columnas nuevas o columnas ausentes.
- **Prediction drift:** cambios en la distribución de predicciones o probabilidades.
- **Model performance:** degradación de métricas cuando estén disponibles las etiquetas reales.

Los módulos de monitoreo se encuentran en `src/monitoring/` y los reportes se organizarían en `reports/monitoring/`, separados por drift de datos, desempeño del modelo y resúmenes ejecutivos.

## Organización de los datos

Los datos deberían organizarse por etapa para mantener trazabilidad y evitar mezclar archivos originales con archivos transformados:

- `data/raw/`: archivos originales descargados o recibidos, sin modificaciones.
- `data/external/`: fuentes externas que complementen las reseñas o productos.
- `data/interim/`: resultados parciales de limpieza, unión o transformación.
- `data/features/`: variables derivadas, agregaciones por producto/mes y matrices de características.
- `data/processed/`: datasets finales listos para entrenamiento, validación, evaluación o monitoreo.

Los archivos pesados o sensibles no deberían subirse directamente a Git. En su lugar, se recomienda versionarlos con DVC y documentar su origen, fecha de generación, esquema de columnas y etapa del pipeline.

## Organización de los modelos

Los modelos deberían almacenarse en `models/` o en una subcarpeta equivalente, separados por versión, fecha o tipo de experimento. Una organización sugerida es:

```text
models/
├── baseline/              # Modelos base para comparación.
├── candidates/            # Modelos candidatos de experimentos.
└── production/            # Modelo seleccionado para uso futuro.
```

Cada modelo debería acompañarse de metadatos mínimos: fecha de entrenamiento, versión del dataset DVC, parámetros principales, métricas de validación, ruta del run de MLflow y dependencias necesarias para reproducirlo. Los binarios de modelos deben versionarse con DVC o registrarse como artefactos de MLflow, evitando almacenarlos directamente en Git.

## Preparación para un despliegue futuro

Para preparar el proyecto para despliegue, se recomienda avanzar hacia una arquitectura reproducible y modular:

1. Convertir notebooks en scripts o módulos reutilizables dentro de `src/`.
2. Separar el pipeline de preprocesamiento, entrenamiento, evaluación e inferencia.
3. Guardar el modelo final junto con su pipeline de transformación para asegurar inferencias consistentes.
4. Definir un archivo de dependencias y comandos reproducibles para ejecución local o en contenedores.
5. Crear una interfaz de inferencia, por ejemplo una API REST con FastAPI o un servicio batch programado.
6. Configurar un flujo de CI/CD que ejecute pruebas, valide datos mínimos y publique artefactos.
7. Integrar MLflow Model Registry o una estrategia equivalente para promover modelos entre etapas.
8. Programar monitoreo con Evidently sobre datos recientes y definir umbrales de alerta para revisión o reentrenamiento.

Con estos pasos, el proyecto quedaría preparado para evolucionar desde un análisis académico hacia una solución mantenible, monitoreable y lista para producción.
