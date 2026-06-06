# Proyecto-Final-Integracion

Sistema de proyección de caída de ventas aplicado a reseñas históricas de productos Amazon.

## Monitoreo

El repositorio incluye la carpeta [`monitoring/`](monitoring/) con un módulo en Python para generar reportes de monitoreo con Evidently y detectar drift/calidad de datos en reseñas o predicciones del modelo.

## Versionado de datos y modelos con DVC

El repositorio incluye una estructura inicial para **DVC**:

- [`.dvc/`](.dvc/) contiene la configuración base del repositorio DVC.
- [`.dvcignore`](.dvcignore) define patrones que DVC no debe recorrer.
- [`data/`](data/) reserva carpetas para datos `raw`, `external`, `interim`, `processed` y `features`.
- [`models/`](models/) reserva el espacio para modelos entrenados y artefactos binarios.
- [`docs/dvc/`](docs/dvc/) documenta el flujo sugerido para agregar datos, modelos y remotos DVC.

Los datasets y modelos reales deben agregarse con `dvc add` para evitar subir archivos pesados directamente a Git.
