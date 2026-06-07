"""Generador de reportes de monitoreo de desempeño del modelo.

Este script crea un paquete de monitoreo dentro de
``reports/monitoring/model_performance`` a partir de dos archivos CSV con
etiquetas reales y predicciones: un dataset de referencia y uno nuevo o de
producción. Genera tres salidas complementarias:

* Reporte HTML visual con métricas globales, métricas por clase y matrices de
  confusión.
* Resumen Markdown ejecutivo para documentar hallazgos y recomendaciones.
* Archivo JSON con métricas estructuradas para auditoría o automatización.

Ejemplo de uso:
    python reports/monitoring/model_performance/generate_model_monitoring_reports.py \
        --reference data/processed/reference_scored.csv \
        --current data/processed/current_scored.csv \
        --target-column target \
        --prediction-column prediction
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_performance_comparison import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SUMMARY_DIR,
    PerformanceComparison,
    compare_model_performance,
    format_delta,
    format_percent,
    load_dataset,
    timestamped_path,
    write_html_report,
    write_summary_markdown,
)


def performance_to_dict(comparison: PerformanceComparison) -> dict[str, Any]:
    """Convierte la comparación de desempeño a un diccionario serializable."""

    def dataset_to_dict(dataset) -> dict[str, Any]:
        return {
            "name": dataset.name,
            "row_count": dataset.row_count,
            "evaluated_rows": dataset.evaluated_rows,
            "dropped_rows": dataset.dropped_rows,
            "metrics": {
                "accuracy": dataset.accuracy,
                "macro_precision": dataset.macro_precision,
                "macro_recall": dataset.macro_recall,
                "macro_f1": dataset.macro_f1,
                "weighted_precision": dataset.weighted_precision,
                "weighted_recall": dataset.weighted_recall,
                "weighted_f1": dataset.weighted_f1,
            },
            "class_metrics": [
                {
                    "label": metric.label,
                    "support": metric.support,
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "f1_score": metric.f1_score,
                }
                for metric in dataset.class_metrics
            ],
            "confusion_matrix": {
                "labels": [str(label) for label in dataset.confusion_matrix.index],
                "values": dataset.confusion_matrix.values.astype(int).tolist(),
            },
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alert_detected": comparison.alert_detected,
        "deltas": {
            "accuracy": comparison.accuracy_delta,
            "macro_f1": comparison.macro_f1_delta,
            "weighted_f1": comparison.weighted_f1_delta,
        },
        "classes_with_f1_drop": list(comparison.classes_with_f1_drop),
        "reference": dataset_to_dict(comparison.reference),
        "current": dataset_to_dict(comparison.current),
    }


def write_metrics_json(output_path: str | Path, comparison: PerformanceComparison) -> Path:
    """Guarda métricas de desempeño en JSON para trazabilidad del monitoreo."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(performance_to_dict(comparison), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def write_monitoring_summary(
    output_path: str | Path,
    reference_path: str | Path,
    current_path: str | Path,
    report_path: Path,
    metrics_path: Path,
    comparison: PerformanceComparison,
) -> Path:
    """Genera un resumen Markdown con enlaces a todas las salidas creadas."""
    summary_path = write_summary_markdown(
        output_path=output_path,
        reference_path=reference_path,
        current_path=current_path,
        report_path=report_path,
        comparison=comparison,
    )

    status = "ALERTA" if comparison.alert_detected else "OK"
    extra_lines = [
        "## Paquete de monitoreo",
        f"- Estado general: **{status}**",
        f"- Métricas JSON: `{metrics_path}`",
        f"- Diferencia accuracy: {format_delta(comparison.accuracy_delta)}",
        f"- Diferencia F1 macro: {format_delta(comparison.macro_f1_delta)}",
        f"- Accuracy referencia: {format_percent(comparison.reference.accuracy)}",
        f"- Accuracy nuevo: {format_percent(comparison.current.accuracy)}",
        "",
    ]
    with summary_path.open("a", encoding="utf-8") as summary_file:
        summary_file.write("\n" + "\n".join(extra_lines))

    return summary_path


def generate_model_monitoring_reports(
    reference_path: str | Path,
    current_path: str | Path,
    target_column: str,
    prediction_column: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    html_output: str | Path | None = None,
    summary_output: str | Path | None = None,
    metrics_output: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Ejecuta el flujo completo y devuelve HTML, Markdown y JSON generados."""
    reference = load_dataset(reference_path)
    current = load_dataset(current_path)
    comparison = compare_model_performance(reference, current, target_column, prediction_column)

    base_output_dir = Path(output_dir)
    report_path = Path(html_output) if html_output else timestamped_path(
        base_output_dir, "model_performance_monitoring", "html"
    )
    summary_path = Path(summary_output) if summary_output else timestamped_path(
        DEFAULT_SUMMARY_DIR, "model_performance_monitoring", "md"
    )
    metrics_path = Path(metrics_output) if metrics_output else timestamped_path(
        base_output_dir / "metrics", "model_performance_metrics", "json"
    )

    saved_report = write_html_report(
        output_path=report_path,
        reference_path=reference_path,
        current_path=current_path,
        target_column=target_column,
        prediction_column=prediction_column,
        comparison=comparison,
    )
    saved_metrics = write_metrics_json(metrics_path, comparison)
    saved_summary = write_monitoring_summary(
        output_path=summary_path,
        reference_path=reference_path,
        current_path=current_path,
        report_path=saved_report,
        metrics_path=saved_metrics,
        comparison=comparison,
    )

    return saved_report, saved_summary, saved_metrics


def parse_args() -> argparse.Namespace:
    """Define argumentos de línea de comandos para generar reportes."""
    parser = argparse.ArgumentParser(
        description="Genera reportes de monitoreo de desempeño del modelo."
    )
    parser.add_argument(
        "--reference", required=True, help="CSV histórico etiquetado de referencia."
    )
    parser.add_argument("--current", required=True, help="CSV nuevo o de producción etiquetado.")
    parser.add_argument("--target-column", required=True, help="Columna con la etiqueta real.")
    parser.add_argument(
        "--prediction-column",
        required=True,
        help="Columna con la predicción del modelo.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta donde se guardan el HTML y el JSON si no se pasan rutas explícitas.",
    )
    parser.add_argument("--html-output", default=None, help="Ruta opcional del reporte HTML.")
    parser.add_argument(
        "--summary-output", default=None, help="Ruta opcional del resumen Markdown."
    )
    parser.add_argument(
        "--metrics-output", default=None, help="Ruta opcional del archivo JSON de métricas."
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del generador de reportes de monitoreo."""
    args = parse_args()
    report_path, summary_path, metrics_path = generate_model_monitoring_reports(
        reference_path=args.reference,
        current_path=args.current,
        target_column=args.target_column,
        prediction_column=args.prediction_column,
        output_dir=args.output_dir,
        html_output=args.html_output,
        summary_output=args.summary_output,
        metrics_output=args.metrics_output,
    )
    print(f"Reporte HTML generado en: {report_path}")
    print(f"Resumen Markdown generado en: {summary_path}")
    print(f"Métricas JSON generadas en: {metrics_path}")


if __name__ == "__main__":
    main()
