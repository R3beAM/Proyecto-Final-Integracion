# Registro de experimentos con MLflow

Este documento describe cómo se usa MLflow en este proyecto para registrar y consultar las corridas de entrenamiento de modelos de clasificación de sentimiento. La implementación de referencia está en `train_and_log_models.py`.

## 1. Nombre del experimento

El nombre del experimento se define con el argumento `--experiment-name`. Si no se entrega un valor, el script usa por defecto:

```text
amazon_reviews_sentiment
```

En el código, MLflow recibe este nombre con:

```python
mlflow.set_experiment(args.experiment_name)
```

Ejemplo de ejecución con un nombre de experimento personalizado:

```bash
python mlflow_tracking/train_and_log_models.py \
  --experiment-name amazon_reviews_sentiment_v2
```

Además, cada corrida se identifica con un `run_name` único compuesto por el tipo de modelo y un timestamp UTC, por ejemplo `tfidf_logistic_regression_20260607T103015123456Z`.

## 2. Parámetros del modelo

Los parámetros permiten reconstruir el contexto de cada corrida. En este proyecto se registran con `mlflow.log_param` valores como:

- `run_name`: nombre único de la corrida y del archivo pickle generado.
- `model_name`: familia del modelo entrenado, por ejemplo `tfidf_logistic_regression` o `tfidf_multinomial_nb`.
- `text_column`: columna de texto utilizada como variable de entrada.
- `target_column`: columna objetivo usada como etiqueta.

Ejemplo conceptual:

```python
mlflow.log_param("run_name", run_name)
mlflow.log_param("model_name", config.name)
mlflow.log_param("text_column", args.text_column)
mlflow.log_param("target_column", args.target_column)
```

Si se agregan nuevos hiperparámetros, por ejemplo `max_iter`, `ngram_range`, `min_df`, `test_size` o `random_state`, deben registrarse de la misma manera para facilitar la trazabilidad:

```python
mlflow.log_param("test_size", args.test_size)
mlflow.log_param("random_state", args.random_state)
mlflow.log_param("tfidf_ngram_range", "1,2")
mlflow.log_param("classifier_max_iter", 1000)
```

## 3. Métricas de evaluación

Las métricas muestran el desempeño del modelo en el conjunto de evaluación. En el script se calculan predicciones sobre `x_test` y se registran métricas con `mlflow.log_metric`.

Actualmente se registran:

- `accuracy`: proporción de predicciones correctas.
- `f1_macro`: promedio macro del F1-score, útil cuando se quiere comparar el desempeño entre clases.

Ejemplo conceptual:

```python
predictions = config.pipeline.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
f1_macro = f1_score(y_test, predictions, average="macro")

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("f1_macro", f1_macro)
```

Si se requiere una evaluación más completa, también pueden agregarse métricas como `precision_macro`, `recall_macro`, matriz de confusión o métricas por clase.

## 4. Artefactos

Los artefactos son archivos asociados a una corrida. En este proyecto, cada modelo entrenado se guarda primero como archivo `.pkl` dentro de `mlflow_tracking/models_pickle/` y luego se registra como artefacto de MLflow.

Ejemplo conceptual:

```python
pickle_path = save_pickle_model(
    model=config.pipeline,
    output_dir=args.pickle_dir,
    run_name=run_name,
)
mlflow.log_artifact(str(pickle_path), artifact_path="pickle_models")
```

Con esta configuración, MLflow guarda los archivos en la sección de artefactos del run bajo la carpeta `pickle_models/`.

También podrían registrarse artefactos adicionales, por ejemplo:

- Matriz de confusión en formato `.png` o `.csv`.
- Reporte de clasificación en `.txt`, `.json` o `.html`.
- Archivo de configuración del entrenamiento.
- Ejemplos de predicciones.
- Versiones procesadas de los datos, si aplica la política del proyecto.

## 5. Modelo final

El modelo final corresponde al pipeline ya entrenado que se selecciona después de comparar las corridas en MLflow. En este proyecto, el modelo queda disponible de dos formas:

1. Como archivo pickle local en `mlflow_tracking/models_pickle/`.
2. Como artefacto del run en MLflow dentro de `pickle_models/`.

Para elegir el modelo final se recomienda:

1. Ejecutar los entrenamientos.
2. Abrir la interfaz de MLflow.
3. Comparar los runs por métricas como `accuracy` y `f1_macro`.
4. Seleccionar el run con mejor desempeño y validar que sus parámetros sean adecuados.
5. Usar el archivo `.pkl` asociado a ese run como modelo final reproducible.

Comando para abrir la interfaz local de MLflow:

```bash
mlflow ui --backend-store-uri file://$(pwd)/mlflow_tracking/mlruns
```

Desde la interfaz se puede revisar el experimento, filtrar corridas, comparar parámetros y métricas, y descargar los artefactos registrados.

## Ejemplo completo de flujo

```bash
python mlflow_tracking/train_and_log_models.py \
  --data data/processed/reviews.csv \
  --text-column review_text \
  --target-column sentiment \
  --experiment-name amazon_reviews_sentiment

mlflow ui --backend-store-uri file://$(pwd)/mlflow_tracking/mlruns
```

Después de ejecutar el flujo, MLflow permitirá consultar:

- El experimento `amazon_reviews_sentiment`.
- Los parámetros de cada corrida.
- Las métricas de evaluación.
- Los artefactos guardados en `pickle_models/`.
- El archivo pickle del modelo candidato a modelo final.
