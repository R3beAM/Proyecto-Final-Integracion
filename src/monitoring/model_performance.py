"""Monitoreo de desempeño del modelo con datos etiquetados recientes."""

from __future__ import annotations

import argparse
from pathlib import Path

from evidently_reports import (
    add_text_monitoring_features,
    load_dataset,
    save_report,
    timestamped_report_path,
    validate_shared_columns,
    write_summary_markdown,
)


def build_model_performance_report(reference, current, target_column: str, prediction_column: str):
    """Construye un reporte de desempeño de clasificación con Evidently."""
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import ClassificationPreset
        from evidently.report import Report

        column_mapping = ColumnMapping()
        column_mapping.target = target_column
        column_mapping.prediction = prediction_column

        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
        return report
    except (ImportError, ModuleNotFoundError, TypeError):
        from evidently import Dataset, Report
        from evidently.presets import ClassificationPreset

        report = Report([ClassificationPreset()])
        report.run(Dataset.from_pandas(reference), Dataset.from_pandas(current))
        return report


def run_model_performance_monitoring(
    reference_path: str | Path,
    current_path: str | Path,
    target_column: str,
    prediction_column: str,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Ejecuta el monitoreo de desempeño cuando hay etiquetas y predicciones."""
    reference = add_text_monitoring_features(load_dataset(reference_path))
    current = add_text_monitoring_features(load_dataset(current_path))
    validate_shared_columns(reference, current, (target_column, prediction_column))

    report_path = Path(output_path) if output_path else timestamped_report_path("model_performance")
    report = build_model_performance_report(reference, current, target_column, prediction_column)
    saved_report = save_report(report, report_path)

    saved_summary = None
    if summary_path:
        saved_summary = write_summary_markdown(
            output_path=summary_path,
            title="Resumen de monitoreo de desempeño del modelo",
            reference_path=reference_path,
            current_path=current_path,
            generated_reports=[saved_report],
            notes=[
                "Comparar accuracy, precision, recall, F1 y matriz de confusión contra la línea base.",
                "Revisar el modelo si el desempeño cae por debajo del umbral definido por negocio.",
            ],
        )

    return saved_report, saved_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera reporte de desempeño del modelo con Evidently.")
    parser.add_argument("--reference", required=True, help="CSV histórico etiquetado de referencia.")
    parser.add_argument("--current", required=True, help="CSV nuevo o de producción etiquetado.")
    parser.add_argument("--target-column", required=True, help="Columna con la etiqueta real.")
    parser.add_argument("--prediction-column", required=True, help="Columna con la predicción del modelo.")
    parser.add_argument("--output", default=None, help="Ruta del reporte HTML.")
    parser.add_argument("--summary", default=None, help="Ruta opcional de resumen Markdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, summary_path = run_model_performance_monitoring(
        reference_path=args.reference,
        current_path=args.current,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
        output_path=args.output,
        summary_path=args.summary,
    )
    print(f"Reporte de desempeño del modelo generado en: {report_path}")
    if summary_path:
        print(f"Resumen generado en: {summary_path}")


if __name__ == "__main__":
    main()
