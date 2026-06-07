"""Monitoreo de drift de datos entre datasets de referencia y producción."""

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


def build_data_drift_report(reference, current):
    """Construye un reporte de drift/calidad de datos con Evidently."""
    try:
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference, current_data=current)
        return report
    except (ImportError, ModuleNotFoundError, TypeError):
        from evidently import Dataset, Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset

        report = Report([DataSummaryPreset(), DataDriftPreset()])
        report.run(Dataset.from_pandas(reference), Dataset.from_pandas(current))
        return report


def run_data_drift_monitoring(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Ejecuta la comparación de drift y devuelve el reporte HTML generado."""
    reference = add_text_monitoring_features(load_dataset(reference_path))
    current = add_text_monitoring_features(load_dataset(current_path))
    validate_shared_columns(reference, current)

    report_path = Path(output_path) if output_path else timestamped_report_path("data_drift")
    report = build_data_drift_report(reference, current)
    saved_report = save_report(report, report_path)

    saved_summary = None
    if summary_path:
        saved_summary = write_summary_markdown(
            output_path=summary_path,
            title="Resumen de monitoreo de drift de datos",
            reference_path=reference_path,
            current_path=current_path,
            generated_reports=[saved_report],
            notes=[
                "Revisar variables con drift estadísticamente significativo o cambios fuertes de distribución.",
                "Priorizar variables de reseñas, calificaciones, votos útiles y señales derivadas del texto.",
            ],
        )

    return saved_report, saved_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera reporte de drift de datos con Evidently.")
    parser.add_argument("--reference", required=True, help="CSV histórico usado como referencia.")
    parser.add_argument("--current", required=True, help="CSV nuevo, actual o de producción.")
    parser.add_argument("--output", default=None, help="Ruta del reporte HTML.")
    parser.add_argument("--summary", default=None, help="Ruta opcional de resumen Markdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path, summary_path = run_data_drift_monitoring(
        reference_path=args.reference,
        current_path=args.current,
        output_path=args.output,
        summary_path=args.summary,
    )
    print(f"Reporte de drift de datos generado en: {report_path}")
    if summary_path:
        print(f"Resumen generado en: {summary_path}")


if __name__ == "__main__":
    main()
