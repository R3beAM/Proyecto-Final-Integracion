"""Monitoreo de drift de datos para datasets de referencia y producción.

Este script compara un CSV histórico contra un CSV actual y genera un reporte
HTML en ``reports/monitoring/data_drift`` con señales de drift por columna. Está
pensado para ejecutarse por lotes y no depende de Evidently: calcula métricas
estadísticas simples y explicables para variables numéricas, categóricas y
variables derivadas de texto.

Ejemplo de uso:
    python reports/monitoring/data_drift/data_drift_monitoring.py \
        --reference data/processed/final_dataset_with_risk.csv \
        --current data/processed/final_dataset_with_risk.csv \
        --output reports/monitoring/data_drift/data_drift_report.html \
        --summary reports/monitoring/summary/data_drift_report.md
"""

from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("reports/monitoring/data_drift")
DEFAULT_SUMMARY_DIR = Path("reports/monitoring/summary")
DEFAULT_TEXT_COLUMNS = (
    "reviews.text",
    "review_text",
    "review_texto",
    "text",
    "content",
    "review",
)
NUMERIC_PSI_ALERT_THRESHOLD = 0.20
CATEGORICAL_PSI_ALERT_THRESHOLD = 0.20
KS_PVALUE_ALERT_THRESHOLD = 0.05
TVD_ALERT_THRESHOLD = 0.20
MISSING_RATE_DELTA_ALERT_THRESHOLD = 0.10


@dataclass(frozen=True)
class DriftResult:
    """Resultado resumido del monitoreo de drift para una columna."""

    column: str
    feature_type: str
    reference_missing_rate: float
    current_missing_rate: float
    primary_metric: str
    primary_value: float
    secondary_metric: str
    secondary_value: float
    drift_detected: bool
    details: str


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


def add_text_monitoring_features(
    dataset: pd.DataFrame,
    text_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS,
) -> pd.DataFrame:
    """Agrega variables derivadas para detectar drift en reseñas o texto libre."""
    monitored = dataset.copy()
    for column in text_columns:
        if column not in monitored.columns:
            continue

        text = monitored[column].fillna("").astype(str)
        safe_name = column.replace(".", "_").replace(" ", "_")
        monitored[f"{safe_name}_char_count"] = text.str.len()
        monitored[f"{safe_name}_word_count"] = text.str.split().str.len()
        monitored[f"{safe_name}_empty"] = text.str.strip().eq("").astype(int)

    return monitored


def select_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    requested_columns: Sequence[str] | None = None,
) -> list[str]:
    """Obtiene columnas comunes, validando columnas solicitadas si se entregan."""
    common_columns = set(reference.columns).intersection(current.columns)
    if requested_columns:
        missing = [column for column in requested_columns if column not in common_columns]
        if missing:
            raise ValueError(
                "Estas columnas no existen en ambos datasets: " + ", ".join(missing)
            )
        return list(requested_columns)

    if not common_columns:
        raise ValueError("Los datasets no tienen columnas en común para monitorear.")

    return sorted(common_columns)


def missing_rate(series: pd.Series) -> float:
    """Calcula la proporción de valores faltantes."""
    if series.empty:
        return 0.0
    return float(series.isna().mean())


def is_numeric_pair(reference: pd.Series, current: pd.Series) -> bool:
    """Determina si ambas columnas deben tratarse como numéricas."""
    return bool(pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(current))


def psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-6) -> float:
    """Calcula Population Stability Index entre dos distribuciones discretizadas."""
    expected = expected.astype(float) + epsilon
    actual = actual.astype(float) + epsilon
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calcula PSI para variables numéricas usando cortes por cuantiles de referencia."""
    ref_values = reference.dropna().astype(float).to_numpy()
    cur_values = current.dropna().astype(float).to_numpy()
    if len(ref_values) == 0 or len(cur_values) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref_values, quantiles))
    if len(edges) <= 2:
        minimum = float(np.nanmin([ref_values.min(), cur_values.min()]))
        maximum = float(np.nanmax([ref_values.max(), cur_values.max()]))
        if math.isclose(minimum, maximum):
            return 0.0
        edges = np.linspace(minimum, maximum, bins + 1)

    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist, _ = np.histogram(ref_values, bins=edges)
    cur_hist, _ = np.histogram(cur_values, bins=edges)
    return psi(ref_hist, cur_hist)


def ks_statistic_and_pvalue(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Calcula estadístico Kolmogorov-Smirnov y una aproximación de p-value."""
    ref_values = np.sort(reference.dropna().astype(float).to_numpy())
    cur_values = np.sort(current.dropna().astype(float).to_numpy())
    if len(ref_values) == 0 or len(cur_values) == 0:
        return 0.0, 1.0

    all_values = np.concatenate([ref_values, cur_values])
    ref_cdf = np.searchsorted(ref_values, all_values, side="right") / len(ref_values)
    cur_cdf = np.searchsorted(cur_values, all_values, side="right") / len(cur_values)
    statistic = float(np.max(np.abs(ref_cdf - cur_cdf)))

    effective_n = len(ref_values) * len(cur_values) / (len(ref_values) + len(cur_values))
    lambda_value = (math.sqrt(effective_n) + 0.12 + 0.11 / math.sqrt(effective_n)) * statistic
    p_value = min(1.0, max(0.0, 2.0 * math.exp(-2.0 * lambda_value * lambda_value)))
    return statistic, p_value


def categorical_distributions(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Obtiene distribuciones categóricas alineadas incluyendo valores faltantes."""
    missing_label = "__MISSING__"
    ref_values = reference.fillna(missing_label).astype(str)
    cur_values = current.fillna(missing_label).astype(str)
    categories = sorted(set(ref_values.unique()).union(cur_values.unique()))
    ref_distribution = ref_values.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    cur_distribution = cur_values.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    return ref_distribution, cur_distribution


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """Calcula PSI para variables categóricas."""
    ref_distribution, cur_distribution = categorical_distributions(reference, current)
    return psi(ref_distribution.to_numpy(), cur_distribution.to_numpy())


def total_variation_distance(reference: pd.Series, current: pd.Series) -> float:
    """Calcula distancia de variación total entre distribuciones categóricas."""
    ref_distribution, cur_distribution = categorical_distributions(reference, current)
    return float(0.5 * np.abs(ref_distribution - cur_distribution).sum())


def summarize_numeric_column(column: str, reference: pd.Series, current: pd.Series) -> DriftResult:
    """Resume drift para una columna numérica."""
    ref_missing = missing_rate(reference)
    cur_missing = missing_rate(current)
    ks_statistic, ks_pvalue = ks_statistic_and_pvalue(reference, current)
    psi_value = numeric_psi(reference, current)
    missing_delta = abs(cur_missing - ref_missing)
    drift_detected = (
        ks_pvalue < KS_PVALUE_ALERT_THRESHOLD
        or psi_value >= NUMERIC_PSI_ALERT_THRESHOLD
        or missing_delta >= MISSING_RATE_DELTA_ALERT_THRESHOLD
    )

    ref_non_null = reference.dropna().astype(float)
    cur_non_null = current.dropna().astype(float)
    details = (
        f"media ref={ref_non_null.mean():.4f}, media actual={cur_non_null.mean():.4f}; "
        f"p50 ref={ref_non_null.median():.4f}, p50 actual={cur_non_null.median():.4f}"
    )

    return DriftResult(
        column=column,
        feature_type="numérica",
        reference_missing_rate=ref_missing,
        current_missing_rate=cur_missing,
        primary_metric="PSI",
        primary_value=psi_value,
        secondary_metric="KS p-value",
        secondary_value=ks_pvalue,
        drift_detected=drift_detected,
        details=details,
    )


def summarize_categorical_column(column: str, reference: pd.Series, current: pd.Series) -> DriftResult:
    """Resume drift para una columna categórica o de texto."""
    ref_missing = missing_rate(reference)
    cur_missing = missing_rate(current)
    psi_value = categorical_psi(reference, current)
    tvd_value = total_variation_distance(reference, current)
    missing_delta = abs(cur_missing - ref_missing)
    drift_detected = (
        psi_value >= CATEGORICAL_PSI_ALERT_THRESHOLD
        or tvd_value >= TVD_ALERT_THRESHOLD
        or missing_delta >= MISSING_RATE_DELTA_ALERT_THRESHOLD
    )

    ref_values = reference.fillna("__MISSING__").astype(str)
    cur_values = current.fillna("__MISSING__").astype(str)
    new_categories = sorted(set(cur_values.unique()) - set(ref_values.unique()))
    top_ref = ref_values.value_counts(normalize=True).head(1)
    top_cur = cur_values.value_counts(normalize=True).head(1)
    top_ref_name = top_ref.index[0] if not top_ref.empty else "n/a"
    top_ref_share = f"{top_ref.iloc[0]:.2%}" if not top_ref.empty else "n/a"
    top_cur_name = top_cur.index[0] if not top_cur.empty else "n/a"
    top_cur_share = f"{top_cur.iloc[0]:.2%}" if not top_cur.empty else "n/a"
    details = (
        f"top ref={top_ref_name} ({top_ref_share}); "
        f"top actual={top_cur_name} ({top_cur_share}); "
        f"categorías nuevas={len(new_categories)}"
    )

    return DriftResult(
        column=column,
        feature_type="categórica/texto",
        reference_missing_rate=ref_missing,
        current_missing_rate=cur_missing,
        primary_metric="PSI",
        primary_value=psi_value,
        secondary_metric="TVD",
        secondary_value=tvd_value,
        drift_detected=drift_detected,
        details=details,
    )


def compute_drift_results(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
) -> list[DriftResult]:
    """Calcula resultados de drift para las columnas seleccionadas."""
    results: list[DriftResult] = []
    for column in columns:
        reference_column = reference[column]
        current_column = current[column]
        if is_numeric_pair(reference_column, current_column):
            results.append(summarize_numeric_column(column, reference_column, current_column))
        else:
            results.append(summarize_categorical_column(column, reference_column, current_column))
    return results


def format_percent(value: float) -> str:
    """Formatea un decimal como porcentaje."""
    return f"{value:.2%}"


def render_results_table(results: Sequence[DriftResult]) -> str:
    """Convierte resultados de drift en una tabla HTML."""
    rows = []
    for result in results:
        status = "⚠️ Drift" if result.drift_detected else "✅ Sin alerta"
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.column)}</td>"
            f"<td>{html.escape(result.feature_type)}</td>"
            f"<td>{format_percent(result.reference_missing_rate)}</td>"
            f"<td>{format_percent(result.current_missing_rate)}</td>"
            f"<td>{html.escape(result.primary_metric)}={result.primary_value:.4f}</td>"
            f"<td>{html.escape(result.secondary_metric)}={result.secondary_value:.4f}</td>"
            f"<td>{status}</td>"
            f"<td>{html.escape(result.details)}</td>"
            "</tr>"
        )

    return "\n".join(rows)


def write_html_report(
    output_path: str | Path,
    reference_path: str | Path,
    current_path: str | Path,
    reference_rows: int,
    current_rows: int,
    results: Sequence[DriftResult],
) -> Path:
    """Escribe el reporte HTML de monitoreo de drift."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    drift_count = sum(result.drift_detected for result in results)

    html_report = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Reporte de Drift de Datos</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 1rem; background: #f9fafb; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem; text-align: left; vertical-align: top; }}
    th {{ background: #e5e7eb; }}
    tr:nth-child(even) {{ background: #f9fafb; }}
    .note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>Reporte de Drift de Datos</h1>
  <p>Generado en UTC: <strong>{generated_at}</strong></p>
  <div class="summary">
    <div class="card"><strong>Dataset referencia</strong><br>{html.escape(str(reference_path))}<br>{reference_rows:,} filas</div>
    <div class="card"><strong>Dataset actual</strong><br>{html.escape(str(current_path))}<br>{current_rows:,} filas</div>
    <div class="card"><strong>Columnas monitoreadas</strong><br>{len(results)}</div>
    <div class="card"><strong>Alertas de drift</strong><br>{drift_count}</div>
  </div>
  <div class="note">
    <strong>Criterios de alerta:</strong> PSI ≥ {NUMERIC_PSI_ALERT_THRESHOLD:.2f},
    KS p-value &lt; {KS_PVALUE_ALERT_THRESHOLD:.2f}, TVD ≥ {TVD_ALERT_THRESHOLD:.2f}
    o cambio de faltantes ≥ {MISSING_RATE_DELTA_ALERT_THRESHOLD:.2%}.
  </div>
  <h2>Resultados por columna</h2>
  <table>
    <thead>
      <tr>
        <th>Columna</th>
        <th>Tipo</th>
        <th>Faltantes referencia</th>
        <th>Faltantes actual</th>
        <th>Métrica principal</th>
        <th>Métrica secundaria</th>
        <th>Estado</th>
        <th>Detalle</th>
      </tr>
    </thead>
    <tbody>
      {render_results_table(results)}
    </tbody>
  </table>
</body>
</html>
"""
    output.write_text(html_report, encoding="utf-8")
    return output


def write_summary_markdown(
    output_path: str | Path,
    reference_path: str | Path,
    current_path: str | Path,
    report_path: Path,
    results: Sequence[DriftResult],
) -> Path:
    """Escribe un resumen Markdown ejecutivo del monitoreo."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    drifted_columns = [result.column for result in results if result.drift_detected]
    drift_lines = "\n".join(f"- {column}" for column in drifted_columns)

    output.write_text(
        "\n".join(
            [
                "# Resumen de monitoreo de drift de datos",
                "",
                f"Generado: {generated_at}",
                f"Dataset de referencia: `{reference_path}`",
                f"Dataset actual/producción: `{current_path}`",
                f"Reporte HTML: `{report_path}`",
                f"Columnas monitoreadas: {len(results)}",
                f"Columnas con alerta de drift: {len(drifted_columns)}",
                "",
                "## Columnas con alerta",
                drift_lines or "- No se detectaron alertas con los umbrales configurados.",
                "",
                "## Recomendación",
                "- Revisar las columnas alertadas, validar cambios de negocio y evaluar reentrenamiento si el drift persiste.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output


def run_data_drift_monitoring(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    columns: Sequence[str] | None = None,
    text_columns: Iterable[str] = DEFAULT_TEXT_COLUMNS,
) -> tuple[Path, Path | None]:
    """Ejecuta el flujo completo de monitoreo y devuelve rutas generadas."""
    reference = add_text_monitoring_features(load_dataset(reference_path), text_columns)
    current = add_text_monitoring_features(load_dataset(current_path), text_columns)
    monitored_columns = select_columns(reference, current, columns)
    results = compute_drift_results(reference, current, monitored_columns)

    report_path = Path(output_path) if output_path else timestamped_path(DEFAULT_OUTPUT_DIR, "data_drift", "html")
    saved_report = write_html_report(
        output_path=report_path,
        reference_path=reference_path,
        current_path=current_path,
        reference_rows=len(reference),
        current_rows=len(current),
        results=results,
    )

    saved_summary = None
    if summary_path:
        saved_summary = write_summary_markdown(
            output_path=summary_path,
            reference_path=reference_path,
            current_path=current_path,
            report_path=saved_report,
            results=results,
        )

    return saved_report, saved_summary


def comma_separated_values(value: str | None) -> list[str] | None:
    """Convierte una lista separada por comas en una lista limpia de valores."""
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Define argumentos de línea de comandos para el monitoreo."""
    parser = argparse.ArgumentParser(description="Genera un reporte HTML de drift de datos.")
    parser.add_argument("--reference", required=True, help="CSV histórico usado como referencia.")
    parser.add_argument("--current", required=True, help="CSV nuevo, actual o de producción.")
    parser.add_argument("--output", default=None, help="Ruta opcional del reporte HTML.")
    parser.add_argument("--summary", default=None, help="Ruta opcional del resumen Markdown.")
    parser.add_argument(
        "--columns",
        default=None,
        help="Columnas a monitorear separadas por coma. Si se omite, usa todas las comunes.",
    )
    parser.add_argument(
        "--text-columns",
        default=",".join(DEFAULT_TEXT_COLUMNS),
        help="Columnas de texto separadas por coma para crear variables derivadas.",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del script."""
    args = parse_args()
    report_path, summary_path = run_data_drift_monitoring(
        reference_path=args.reference,
        current_path=args.current,
        output_path=args.output,
        summary_path=args.summary,
        columns=comma_separated_values(args.columns),
        text_columns=comma_separated_values(args.text_columns) or DEFAULT_TEXT_COLUMNS,
    )
    print(f"Reporte de drift de datos generado en: {report_path}")
    if summary_path:
        print(f"Resumen generado en: {summary_path}")


if __name__ == "__main__":
    main()
