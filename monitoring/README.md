# Monitoreo con Evidently

Esta carpeta contiene un módulo para monitorear el proyecto de análisis de sentimiento de reseñas de Amazon con **Evidently**.

## Objetivo

El monitoreo compara un dataset de referencia contra un dataset actual para detectar:

- **Data drift**: cambios en la distribución de variables como calificación, votos útiles o longitud de reseñas.
- **Calidad de datos**: valores faltantes, columnas con tipos inesperados y cambios generales en el dataset.
- **Desempeño de clasificación**: métricas del modelo cuando existen columnas de etiqueta real y predicción.

## Instalación sugerida

```bash
pip install pandas evidently
```

## Uso básico

```bash
python monitoring/evidently_monitoring.py \
  --reference data/reference.csv \
  --current data/current.csv \
  --output monitoring/reports/amazon_reviews_monitoring.html
```

## Uso con target y predicción

Si los archivos incluyen una columna real de sentimiento y una columna con la predicción del modelo:

```bash
python monitoring/evidently_monitoring.py \
  --reference data/reference.csv \
  --current data/current.csv \
  --target-column sentiment \
  --prediction-column prediction \
  --output monitoring/reports/amazon_reviews_monitoring.html
```

## Variables derivadas del texto

El módulo crea automáticamente variables auxiliares cuando encuentra columnas como `reviews.text`, `review_text`, `text` o `content`:

- conteo de caracteres;
- conteo de palabras;
- indicador de texto vacío.

Estas variables facilitan detectar drift en el comportamiento de las reseñas sin alterar el dataset original.

## Salida

El archivo HTML generado queda por defecto en:

```text
monitoring/reports/amazon_reviews_monitoring.html
```

La carpeta `monitoring/reports/` está destinada a almacenar reportes locales de monitoreo.
