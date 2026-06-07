"""Utilidades compartidas para generar reportes de monitoreo con Evidently.

El módulo mantiene compatibilidad con APIs clásicas y recientes de Evidently para
que los reportes puedan generarse aun cuando cambie la versión instalada.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

MONITORING_REPORTS_DIR = Path("reports/monitoring")
DATA_DRIFT_REPORTS_DIR = MONITORING_REPORTS_DIR / "data_drift"
MODEL_PERFORMANCE_REPORTS_DIR = MONITORING_REPORTS_DIR / "model_performance"
SUMMARY_REPORTS_DIR = MONITORING_REPORTS_DIR / "summary"


DEFAULT_TEXT_COLUMNS = (
    "reviews.text",
    "review_text",
    "review_texto",
    "text",
    "content",
    "review",
)


def timestamped_report_path(report_type: str, suffix: str = "html") -> Path:
    """Construye una ruta fechada dentro de reports/monitoring para un reporte."""
    directories = {
        "data_drift": DATA_DRIFT_REPORTS_DIR,
        "model_performance": MODEL_PERFORMANCE_REPORTS_DIR,
        "summary": SUMMARY_REPORTS_DIR,
    }
    if report_type not in directories:
        raise ValueError(f"Tipo de reporte no soportado: {report_type}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return directories[report_type] / f"{report_type}_{timestamp}.{suffix}"


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carga un CSV y normaliza los nombres de columnas."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def add_text_monitoring_features(
    df: pd.DataFrame,
    text_columns: tuple[str, ...] = DEFAULT_TEXT_COLUMNS,
) -> pd.DataFrame:
    """Agrega variables derivadas para monitorear cambios en reseñas de texto."""
    monitored = df.copy()
    for column in text_columns:
        if column not in monitored.columns:
            continue

        text = monitored[column].fillna("").astype(str)
        safe_name = column.replace(".", "_").replace(" ", "_")
        monitored[f"{safe_name}_char_count"] = text.str.len()
        monitored[f"{safe_name}_word_count"] = text.str.split().str.len()
        monitored[f"{safe_name}_empty"] = text.str.strip().eq("").astype(int)

    return monitored


def validate_shared_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    required_columns: tuple[str | None, ...] = (),
) -> None:
    """Valida columnas comunes y columnas requeridas para reportes específicos."""
    common_columns = set(reference.columns).intersection(current.columns)
    if not common_columns:
        raise ValueError("Los datasets no tienen columnas en común para comparar.")

    missing_required = [
        column
        for column in required_columns
        if column and (column not in reference.columns or column not in current.columns)
    ]
    if missing_required:
        raise ValueError(
            "Las siguientes columnas deben existir en referencia y producción: "
            + ", ".join(missing_required)
        )


def save_report(report: Any, output_path: str | Path) -> Path:
    """Guarda un reporte HTML de Evidently y crea la carpeta destino si no existe."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(report, "save_html"):
        report.save_html(str(output))
    else:
        report.save(str(output))

    return output


def write_summary_markdown(
    output_path: str | Path,
    title: str,
    reference_path: str | Path,
    current_path: str | Path,
    generated_reports: list[Path],
    notes: list[str] | None = None,
) -> Path:
    """Genera un resumen Markdown simple de los reportes creados."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    report_lines = "\n".join(f"- `{path}`" for path in generated_reports)
    note_lines = "\n".join(f"- {note}" for note in notes or [])
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    output.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"Generado: {generated_at}",
                f"Dataset de referencia: `{reference_path}`",
                f"Dataset nuevo/producción: `{current_path}`",
                "",
                "## Reportes generados",
                report_lines or "- No se generaron reportes HTML.",
                "",
                "## Notas de monitoreo",
                note_lines or "- Revisar los reportes HTML para interpretar alertas y métricas.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output
