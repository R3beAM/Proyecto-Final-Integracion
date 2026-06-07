"""Comparación de desempeño del modelo entre datos de referencia y datos nuevos.

Este script compara un CSV histórico contra un CSV nuevo/actual con etiquetas
reales y predicciones. Genera un reporte HTML en
``reports/monitoring/model_performance`` y, opcionalmente, un resumen Markdown
con métricas de clasificación calculadas sin depender de Evidently ni sklearn.

Ejemplo de uso:
    python reports/monitoring/model_performance/model_performance_comparison.py \
        --reference data/processed/reference_scored.csv \
        --current data/processed/current_scored.csv \
        --target-column target \
        --prediction-column prediction \
        --output reports/monitoring/model_performance/model_performance_report.html \
        --summary reports/monitoring/summary/model_performance_report.md
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("reports/monitoring/model_performance")
DEFAULT_SUMMARY_DIR = Path("reports/monitoring/summary")
ACCURACY_DROP_ALERT_THRESHOLD = 0.05
MACRO_F1_DROP_ALERT_THRESHOLD = 0.05
CLASS_F1_DROP_ALERT_THRESHOLD = 0.10


@dataclass(frozen=True)
class ClassMetrics:
    """Métricas de clasificación para una clase específica."""

    label: str
    support: int
    precision: float
    recall: float
    f1_score: float


@dataclass(frozen=True)
class DatasetPerformance:
    """Resumen de desempeño de un dataset etiquetado y puntuado."""

    name: str
    row_count: int
    evaluated_rows: int
    dropped_rows: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    class_metrics: tuple[ClassMetrics, ...]
    confusion_matrix: pd.DataFrame


@dataclass(frozen=True)
class PerformanceComparison:
    """Comparación entre desempeño de referencia y desempeño nuevo."""

    reference: DatasetPerformance
    current: DatasetPerformance
    accuracy_delta: float
    macro_f1_delta: float
    weighted_f1_delta: float
    classes_with_f1_drop: tuple[str, ...]
    alert_detected: bool


def timestamped_path(directory: Path, prefix: str, suffix: str) -> Path:
    """Construye una ruta con timestamp UTC para evitar sobrescribir reportes."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return directory / f"{prefix}_{timestamp}.{suffix}"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carga un CSV y normaliza nombres de columnas eliminando espacios externos."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    dataset.columns = [str(column).strip() for column in dataset.columns]
    return dataset


def validate_required_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """Valida que las columnas requeridas existan en ambos datasets."""
    missing: list[str] = []
    for column in required_columns:
        if column not in reference.columns:
            missing.append(f"{column} (referencia)")
        if column not in current.columns:
            missing.append(f"{column} (nuevo)")

    if missing:
        raise ValueError("Columnas requeridas faltantes: " + ", ".join(missing))


def clean_scored_dataset(
    dataset: pd.DataFrame,
    target_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    """Conserva filas con etiqueta y predicción disponibles para evaluar."""
    scored = dataset[[target_column, prediction_column]].copy()
    scored = scored.dropna(subset=[target_column, prediction_column])
    scored[target_column] = scored[target_column].astype(str)
    scored[prediction_column] = scored[prediction_column].astype(str)
    return scored


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide retornando cero cuando el denominador es cero."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_class_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: Sequence[str],
) -> tuple[ClassMetrics, ...]:
    """Calcula precision, recall y F1 por clase."""
    results: list[ClassMetrics] = []
    for label in labels:
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())
        support = int((y_true == label).sum())

        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1_score = safe_divide(2 * precision * recall, precision + recall)
        results.append(
            ClassMetrics(
                label=label,
                support=support,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )
        )

    return tuple(results)


def weighted_average(metrics: Sequence[ClassMetrics], attribute: str) -> float:
    """Calcula promedio ponderado por soporte para una métrica por clase."""
    total_support = sum(metric.support for metric in metrics)
    if total_support == 0:
        return 0.0
    return float(
        sum(getattr(metric, attribute) * metric.support for metric in metrics) / total_support
    )


def macro_average(metrics: Sequence[ClassMetrics], attribute: str) -> float:
    """Calcula promedio macro para una métrica por clase."""
    if not metrics:
        return 0.0
    return float(sum(getattr(metric, attribute) for metric in metrics) / len(metrics))


def compute_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: Sequence[str],
) -> pd.DataFrame:
    """Construye matriz de confusión con etiquetas reales en filas."""
    matrix = pd.crosstab(y_true, y_pred, rownames=["real"], colnames=["predicho"], dropna=False)
    return matrix.reindex(index=labels, columns=labels, fill_value=0).astype(int)


def compute_dataset_performance(
    dataset: pd.DataFrame,
    name: str,
    target_column: str,
    prediction_column: str,
    labels: Sequence[str],
) -> DatasetPerformance:
    """Calcula métricas de desempeño para un dataset."""
    scored = clean_scored_dataset(dataset, target_column, prediction_column)
    if scored.empty:
        raise ValueError(f"El dataset {name} no tiene filas evaluables con target y predicción.")

    y_true = scored[target_column]
    y_pred = scored[prediction_column]
    class_metrics = compute_class_metrics(y_true, y_pred, labels)
    accuracy = float((y_true == y_pred).mean())

    return DatasetPerformance(
        name=name,
        row_count=len(dataset),
        evaluated_rows=len(scored),
        dropped_rows=len(dataset) - len(scored),
        accuracy=accuracy,
        macro_precision=macro_average(class_metrics, "precision"),
        macro_recall=macro_average(class_metrics, "recall"),
        macro_f1=macro_average(class_metrics, "f1_score"),
        weighted_precision=weighted_average(class_metrics, "precision"),
        weighted_recall=weighted_average(class_metrics, "recall"),
        weighted_f1=weighted_average(class_metrics, "f1_score"),
        class_metrics=class_metrics,
        confusion_matrix=compute_confusion_matrix(y_true, y_pred, labels),
    )


def collect_labels(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str,
    prediction_column: str,
) -> list[str]:
    """Obtiene etiquetas presentes en targets o predicciones de ambos datasets."""
    labels = pd.concat(
        [
            reference[target_column],
            reference[prediction_column],
            current[target_column],
            current[prediction_column],
        ],
        ignore_index=True,
    ).dropna()
    return sorted(labels.astype(str).unique().tolist())


def compare_model_performance(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str,
    prediction_column: str,
) -> PerformanceComparison:
    """Compara métricas de desempeño entre referencia y datos nuevos."""
    validate_required_columns(reference, current, (target_column, prediction_column))
    labels = collect_labels(reference, current, target_column, prediction_column)
    if not labels:
        raise ValueError("No hay etiquetas disponibles para comparar desempeño.")

    reference_performance = compute_dataset_performance(
        reference, "Referencia", target_column, prediction_column, labels
    )
    current_performance = compute_dataset_performance(
        current, "Datos nuevos", target_column, prediction_column, labels
    )

    reference_f1_by_class = {
        metric.label: metric.f1_score for metric in reference_performance.class_metrics
    }
    classes_with_f1_drop = tuple(
        metric.label
        for metric in current_performance.class_metrics
        if reference_f1_by_class.get(metric.label, 0.0) - metric.f1_score
        >= CLASS_F1_DROP_ALERT_THRESHOLD
    )
    accuracy_delta = current_performance.accuracy - reference_performance.accuracy
    macro_f1_delta = current_performance.macro_f1 - reference_performance.macro_f1
    weighted_f1_delta = current_performance.weighted_f1 - reference_performance.weighted_f1

    return PerformanceComparison(
        reference=reference_performance,
        current=current_performance,
        accuracy_delta=accuracy_delta,
        macro_f1_delta=macro_f1_delta,
        weighted_f1_delta=weighted_f1_delta,
        classes_with_f1_drop=classes_with_f1_drop,
        alert_detected=(
            accuracy_delta <= -ACCURACY_DROP_ALERT_THRESHOLD
            or macro_f1_delta <= -MACRO_F1_DROP_ALERT_THRESHOLD
            or bool(classes_with_f1_drop)
        ),
    )


def format_percent(value: float) -> str:
    """Formatea proporciones como porcentaje con dos decimales."""
    return f"{value:.2%}"


def format_delta(value: float) -> str:
    """Formatea diferencias en puntos porcentuales."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f} pp"


def render_metric_row(metric_name: str, reference_value: float, current_value: float) -> str:
    """Renderiza una fila de comparación de métricas globales."""
    delta = current_value - reference_value
    status_class = "alert" if delta < 0 else "ok"
    status_text = "Baja" if delta < 0 else "Mejora/estable"
    return f"""
      <tr>
        <td>{html.escape(metric_name)}</td>
        <td>{format_percent(reference_value)}</td>
        <td>{format_percent(current_value)}</td>
        <td class=\"{status_class}\">{format_delta(delta)}</td>
        <td class=\"{status_class}\">{status_text}</td>
      </tr>
    """


def render_class_rows(comparison: PerformanceComparison) -> str:
    """Renderiza tabla de métricas por clase."""
    reference_by_label = {
        metric.label: metric for metric in comparison.reference.class_metrics
    }
    rows: list[str] = []
    for current_metric in comparison.current.class_metrics:
        reference_metric = reference_by_label[current_metric.label]
        f1_delta = current_metric.f1_score - reference_metric.f1_score
        status_class = "alert" if current_metric.label in comparison.classes_with_f1_drop else "ok"
        status_text = "Alerta" if status_class == "alert" else "OK"
        rows.append(
            f"""
      <tr>
        <td>{html.escape(current_metric.label)}</td>
        <td>{reference_metric.support}</td>
        <td>{current_metric.support}</td>
        <td>{format_percent(reference_metric.precision)}</td>
        <td>{format_percent(current_metric.precision)}</td>
        <td>{format_percent(reference_metric.recall)}</td>
        <td>{format_percent(current_metric.recall)}</td>
        <td>{format_percent(reference_metric.f1_score)}</td>
        <td>{format_percent(current_metric.f1_score)}</td>
        <td class=\"{status_class}\">{format_delta(f1_delta)}</td>
        <td class=\"{status_class}\">{status_text}</td>
      </tr>
            """
        )
    return "\n".join(rows)


def render_confusion_matrix(matrix: pd.DataFrame) -> str:
    """Renderiza una matriz de confusión en HTML."""
    header_cells = "".join(f"<th>{html.escape(str(column))}</th>" for column in matrix.columns)
    body_rows: list[str] = []
    for index, row in matrix.iterrows():
        cells = "".join(f"<td>{int(value)}</td>" for value in row)
        body_rows.append(f"<tr><th>{html.escape(str(index))}</th>{cells}</tr>")
    return f"""
      <table>
        <thead><tr><th>Real \\ Predicho</th>{header_cells}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    """


def write_html_report(
    output_path: str | Path,
    reference_path: str | Path,
    current_path: str | Path,
    target_column: str,
    prediction_column: str,
    comparison: PerformanceComparison,
) -> Path:
    """Escribe el reporte HTML de comparación de desempeño."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status_class = "alert" if comparison.alert_detected else "ok"
    status_text = "Revisar desempeño" if comparison.alert_detected else "Desempeño estable"
    class_alerts = ", ".join(comparison.classes_with_f1_drop) or "Sin alertas por clase"

    report = f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Comparación de desempeño del modelo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.55rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 1rem; background: #f9fafb; }}
    .ok {{ color: #047857; font-weight: 700; }}
    .alert {{ color: #b91c1c; font-weight: 700; }}
    .note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 1rem; margin: 1rem 0; }}
  </style>
</head>
<body>
  <h1>Comparación de desempeño del modelo</h1>
  <p>Generado: <strong>{generated_at}</strong></p>
  <div class="summary">
    <div class="card"><strong>Dataset referencia</strong><br>{html.escape(str(reference_path))}<br>{comparison.reference.evaluated_rows:,} filas evaluadas</div>
    <div class="card"><strong>Dataset nuevo</strong><br>{html.escape(str(current_path))}<br>{comparison.current.evaluated_rows:,} filas evaluadas</div>
    <div class="card"><strong>Columnas evaluadas</strong><br>Target: {html.escape(target_column)}<br>Predicción: {html.escape(prediction_column)}</div>
    <div class="card"><strong>Estado</strong><br><span class="{status_class}">{status_text}</span></div>
  </div>
  <div class="note">
    <strong>Criterios de alerta:</strong> caída de accuracy ≥ {ACCURACY_DROP_ALERT_THRESHOLD:.0%},
    caída de macro F1 ≥ {MACRO_F1_DROP_ALERT_THRESHOLD:.0%} o caída de F1 por clase ≥ {CLASS_F1_DROP_ALERT_THRESHOLD:.0%}.
    <br><strong>Clases con alerta:</strong> {html.escape(class_alerts)}.
  </div>
  <h2>Métricas globales</h2>
  <table>
    <thead><tr><th>Métrica</th><th>Referencia</th><th>Datos nuevos</th><th>Diferencia</th><th>Estado</th></tr></thead>
    <tbody>
      {render_metric_row('Accuracy', comparison.reference.accuracy, comparison.current.accuracy)}
      {render_metric_row('Precision macro', comparison.reference.macro_precision, comparison.current.macro_precision)}
      {render_metric_row('Recall macro', comparison.reference.macro_recall, comparison.current.macro_recall)}
      {render_metric_row('F1 macro', comparison.reference.macro_f1, comparison.current.macro_f1)}
      {render_metric_row('Precision ponderada', comparison.reference.weighted_precision, comparison.current.weighted_precision)}
      {render_metric_row('Recall ponderado', comparison.reference.weighted_recall, comparison.current.weighted_recall)}
      {render_metric_row('F1 ponderado', comparison.reference.weighted_f1, comparison.current.weighted_f1)}
    </tbody>
  </table>
  <h2>Métricas por clase</h2>
  <table>
    <thead>
      <tr>
        <th>Clase</th><th>Soporte ref.</th><th>Soporte nuevo</th>
        <th>Precisión ref.</th><th>Precisión nuevo</th>
        <th>Recall ref.</th><th>Recall nuevo</th>
        <th>F1 ref.</th><th>F1 nuevo</th><th>Δ F1</th><th>Estado</th>
      </tr>
    </thead>
    <tbody>{render_class_rows(comparison)}</tbody>
  </table>
  <h2>Matriz de confusión - Referencia</h2>
  {render_confusion_matrix(comparison.reference.confusion_matrix)}
  <h2>Matriz de confusión - Datos nuevos</h2>
  {render_confusion_matrix(comparison.current.confusion_matrix)}
</body>
</html>
"""
    output.write_text(report, encoding="utf-8")
    return output


def write_summary_markdown(
    output_path: str | Path,
    reference_path: str | Path,
    current_path: str | Path,
    report_path: Path,
    comparison: PerformanceComparison,
) -> Path:
    """Escribe un resumen Markdown ejecutivo del monitoreo de desempeño."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    alert_lines = "\n".join(f"- {label}" for label in comparison.classes_with_f1_drop)

    output.write_text(
        "\n".join(
            [
                "# Resumen de monitoreo de desempeño del modelo",
                "",
                f"Generado: {generated_at}",
                f"Dataset de referencia: `{reference_path}`",
                f"Dataset nuevo/producción: `{current_path}`",
                f"Reporte HTML: `{report_path}`",
                f"Accuracy referencia: {format_percent(comparison.reference.accuracy)}",
                f"Accuracy nuevo: {format_percent(comparison.current.accuracy)}",
                f"Diferencia accuracy: {format_delta(comparison.accuracy_delta)}",
                f"F1 macro referencia: {format_percent(comparison.reference.macro_f1)}",
                f"F1 macro nuevo: {format_percent(comparison.current.macro_f1)}",
                f"Diferencia F1 macro: {format_delta(comparison.macro_f1_delta)}",
                "",
                "## Clases con caída relevante de F1",
                alert_lines or "- No se detectaron alertas por clase con los umbrales configurados.",
                "",
                "## Recomendación",
                "- Revisar datos nuevos, matriz de confusión y clases alertadas antes de decidir reentrenamiento o ajuste de umbrales.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


def run_model_performance_comparison(
    reference_path: str | Path,
    current_path: str | Path,
    target_column: str,
    prediction_column: str,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Ejecuta el flujo completo de comparación y devuelve rutas generadas."""
    reference = load_dataset(reference_path)
    current = load_dataset(current_path)
    comparison = compare_model_performance(reference, current, target_column, prediction_column)

    report_path = (
        Path(output_path)
        if output_path
        else timestamped_path(DEFAULT_OUTPUT_DIR, "model_performance", "html")
    )
    saved_report = write_html_report(
        output_path=report_path,
        reference_path=reference_path,
        current_path=current_path,
        target_column=target_column,
        prediction_column=prediction_column,
        comparison=comparison,
    )

    saved_summary = None
    if summary_path:
        saved_summary = write_summary_markdown(
            output_path=summary_path,
            reference_path=reference_path,
            current_path=current_path,
            report_path=saved_report,
            comparison=comparison,
        )

    return saved_report, saved_summary


def parse_args() -> argparse.Namespace:
    """Define argumentos de línea de comandos para la comparación."""
    parser = argparse.ArgumentParser(
        description="Compara el desempeño del modelo entre datos de referencia y datos nuevos."
    )
    parser.add_argument("--reference", required=True, help="CSV histórico etiquetado de referencia.")
    parser.add_argument("--current", required=True, help="CSV nuevo, actual o de producción etiquetado.")
    parser.add_argument("--target-column", required=True, help="Columna con la etiqueta real.")
    parser.add_argument("--prediction-column", required=True, help="Columna con la predicción del modelo.")
    parser.add_argument("--output", default=None, help="Ruta opcional del reporte HTML.")
    parser.add_argument("--summary", default=None, help="Ruta opcional del resumen Markdown.")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del script."""
    args = parse_args()
    report_path, summary_path = run_model_performance_comparison(
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
