"""
Monitoreo de datos y desempeño con Evidently para reseñas de Amazon.

Este módulo genera reportes HTML comparando un dataset de referencia
(por ejemplo, datos históricos de entrenamiento) contra un dataset actual
(por ejemplo, reseñas recientes o inferencias en producción).

Ejemplo de uso:
    python monitoring/evidently_monitoring.py \
        --reference data/reference.csv \
        --current data/current.csv \
        --target-column sentiment \
        --prediction-column prediction \
        --output monitoring/reports/amazon_reviews_monitoring.html

Si el dataset no cuenta con columnas de target/predicción, omita esos
argumentos y el reporte incluirá únicamente drift/calidad de datos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_TEXT_COLUMNS = ("reviews.text", "review_text", "text", "content")
DEFAULT_OUTPUT = Path("monitoring/reports/amazon_reviews_monitoring.html")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carga un CSV y normaliza nombres de columnas quitando espacios externos."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {dataset_path}")

    import pandas as pd

    df = pd.read_csv(dataset_path)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def add_text_monitoring_features(
    df: pd.DataFrame,
    text_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS,
) -> pd.DataFrame:
    """
    Agrega variables simples para monitorear cambios en reseñas de texto.

    Evidently trabaja muy bien con variables tabulares; estas variables ayudan
    a detectar cambios en longitud y densidad de texto sin modificar la columna
    original de la reseña.
    """
    monitored = df.copy()
    for column in text_columns:
        if column not in monitored.columns:
            continue

        text = monitored[column].fillna("").astype(str)
        safe_name = column.replace(".", "_")
        monitored[f"{safe_name}_char_count"] = text.str.len()
        monitored[f"{safe_name}_word_count"] = text.str.split().str.len()
        monitored[f"{safe_name}_empty"] = text.str.strip().eq("").astype(int)

    return monitored


def _build_legacy_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str | None,
    prediction_column: str | None,
):
    """Construye reportes para Evidently 0.3/0.4/0.6 con presets clásicos."""
    from evidently import ColumnMapping
    from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset
    from evidently.report import Report

    column_mapping = ColumnMapping()
    column_mapping.target = target_column
    column_mapping.prediction = prediction_column

    metrics = [DataDriftPreset(), DataQualityPreset()]
    if target_column and prediction_column:
        metrics.append(ClassificationPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    return report


def _build_modern_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str | None,
    prediction_column: str | None,
):
    """
    Construye reportes para versiones recientes de Evidently cuando estén
    disponibles las APIs nuevas. Mantiene compatibilidad sin forzar una versión.
    """
    from evidently import Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    reference_dataset = Dataset.from_pandas(reference)
    current_dataset = Dataset.from_pandas(current)

    metrics = [DataSummaryPreset(), DataDriftPreset()]

    # Las métricas de clasificación han cambiado entre versiones; si el usuario
    # incluye target/predicción, intentamos agregarlas sin romper el monitoreo base.
    if target_column and prediction_column:
        try:
            from evidently.presets import ClassificationPreset

            metrics.append(ClassificationPreset())
        except ImportError:
            pass

    report = Report(metrics)
    report.run(reference_dataset, current_dataset)
    return report


def create_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str | None = None,
    prediction_column: str | None = None,
):
    """
    Crea un reporte de Evidently priorizando compatibilidad entre versiones.

    El reporte siempre intenta incluir drift y calidad/resumen de datos; si se
    reciben columnas de target y predicción, también intenta agregar desempeño
    de clasificación.
    """
    try:
        return _build_legacy_evidently_report(
            reference=reference,
            current=current,
            target_column=target_column,
            prediction_column=prediction_column,
        )
    except (ImportError, ModuleNotFoundError, TypeError):
        return _build_modern_evidently_report(
            reference=reference,
            current=current,
            target_column=target_column,
            prediction_column=prediction_column,
        )


def save_report(report, output_path: str | Path) -> Path:
    """Guarda el reporte HTML en disco y retorna la ruta generada."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(report, "save_html"):
        report.save_html(str(output))
    else:
        report.save(str(output))

    return output


def run_monitoring(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    target_column: str | None = None,
    prediction_column: str | None = None,
) -> Path:
    """Ejecuta el flujo completo de monitoreo y devuelve el HTML generado."""
    reference = add_text_monitoring_features(load_dataset(reference_path))
    current = add_text_monitoring_features(load_dataset(current_path))

    missing_columns = [
        column
        for column in (target_column, prediction_column)
        if column and (column not in reference.columns or column not in current.columns)
    ]
    if missing_columns:
        raise ValueError(
            "Las siguientes columnas no existen en ambos datasets: "
            + ", ".join(missing_columns)
        )

    report = create_evidently_report(
        reference=reference,
        current=current,
        target_column=target_column,
        prediction_column=prediction_column,
    )
    return save_report(report, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un reporte de monitoreo con Evidently para reseñas de Amazon."
    )
    parser.add_argument("--reference", required=True, help="CSV de referencia/histórico.")
    parser.add_argument("--current", required=True, help="CSV actual o de producción.")
    parser.add_argument(
        "--target-column",
        default=None,
        help="Columna real de sentimiento/clase, si está disponible.",
    )
    parser.add_argument(
        "--prediction-column",
        default=None,
        help="Columna de predicción del modelo, si está disponible.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Ruta del reporte HTML generado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = run_monitoring(
        reference_path=args.reference,
        current_path=args.current,
        output_path=args.output,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
    )
    print(f"Reporte de monitoreo generado en: {report_path}")


if __name__ == "__main__":
    main()
