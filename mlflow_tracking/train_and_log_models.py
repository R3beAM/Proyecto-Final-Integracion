"""Entrena modelos de sentimiento y los registra en MLflow como archivos pickle.

El script puede ejecutarse con un CSV real o con un dataset de ejemplo incluido en
memoria. En cada corrida guarda modelos con nombres distintos en formato `.pkl` y
los registra como artefactos de MLflow.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DEFAULT_EXPERIMENT_NAME = "amazon_reviews_sentiment"
DEFAULT_TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"
DEFAULT_PICKLE_DIR = Path(__file__).resolve().parent / "models_pickle"


@dataclass(frozen=True)
class ModelConfig:
    """Configuración de un modelo que se entrenará y registrará."""

    name: str
    pipeline: Pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena modelos de clasificación de sentimiento y guarda cada "
            "corrida como artefacto pickle en MLflow."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Ruta opcional a un CSV con reseñas. Si no se informa, usa datos de ejemplo.",
    )
    parser.add_argument(
        "--text-column",
        default="review_text",
        help="Nombre de la columna de texto en el CSV.",
    )
    parser.add_argument(
        "--target-column",
        default="sentiment",
        help="Nombre de la columna objetivo en el CSV.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Nombre del experimento en MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=f"file://{DEFAULT_TRACKING_DIR}",
        help="URI de tracking de MLflow. Por defecto usa la carpeta local mlruns/.",
    )
    parser.add_argument(
        "--pickle-dir",
        type=Path,
        default=DEFAULT_PICKLE_DIR,
        help="Carpeta local donde se guardan los modelos pickle generados.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Proporción del dataset usada para evaluación.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para partición reproducible del dataset.",
    )
    return parser


def load_training_data(
    data_path: Path | None,
    text_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Carga datos reales o devuelve un dataset mínimo de ejemplo."""
    if data_path is None:
        return pd.DataFrame(
            {
                text_column: [
                    "excelente producto y entrega rapida",
                    "muy buena calidad lo recomiendo",
                    "funciona perfecto y cumple lo prometido",
                    "mala compra llego defectuoso",
                    "pesima calidad no lo volveria a comprar",
                    "el producto se rompio en pocos dias",
                    "gran relacion calidad precio",
                    "no sirve y la atencion fue mala",
                ],
                target_column: [
                    "positivo",
                    "positivo",
                    "positivo",
                    "negativo",
                    "negativo",
                    "negativo",
                    "positivo",
                    "negativo",
                ],
            }
        )

    dataset = pd.read_csv(data_path)
    missing_columns = {text_column, target_column} - set(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"El CSV no contiene las columnas requeridas: {missing}")
    return dataset[[text_column, target_column]].dropna()


def get_model_configs(random_state: int) -> Iterable[ModelConfig]:
    """Define los modelos que se ejecutarán con nombres diferenciados."""
    return (
        ModelConfig(
            name="tfidf_logistic_regression",
            pipeline=Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                    (
                        "classifier",
                        LogisticRegression(max_iter=1000, random_state=random_state),
                    ),
                ]
            ),
        ),
        ModelConfig(
            name="tfidf_multinomial_nb",
            pipeline=Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1)),
                    ("classifier", MultinomialNB()),
                ]
            ),
        ),
    )


def build_run_name(model_name: str) -> str:
    """Construye un nombre único para identificar la corrida y el pickle."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{model_name}_{timestamp}"


def save_pickle_model(model: Pipeline, output_dir: Path, run_name: str) -> Path:
    """Guarda el modelo en formato pickle con un nombre único por corrida."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_name}.pkl"
    with output_path.open("wb") as file_handler:
        pickle.dump(model, file_handler)
    return output_path


def train_and_log_models(args: argparse.Namespace) -> list[Path]:
    """Entrena todos los modelos configurados y registra artefactos en MLflow."""
    dataset = load_training_data(args.data, args.text_column, args.target_column)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[args.text_column],
        dataset[args.target_column],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=dataset[args.target_column],
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    generated_pickles: list[Path] = []
    for config in get_model_configs(args.random_state):
        run_name = build_run_name(config.name)
        with mlflow.start_run(run_name=run_name):
            config.pipeline.fit(x_train, y_train)
            predictions = config.pipeline.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            f1_macro = f1_score(y_test, predictions, average="macro")

            pickle_path = save_pickle_model(
                model=config.pipeline,
                output_dir=args.pickle_dir,
                run_name=run_name,
            )
            generated_pickles.append(pickle_path)

            mlflow.log_param("run_name", run_name)
            mlflow.log_param("model_name", config.name)
            mlflow.log_param("text_column", args.text_column)
            mlflow.log_param("target_column", args.target_column)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_artifact(str(pickle_path), artifact_path="pickle_models")

    return generated_pickles


def main() -> None:
    args = build_arg_parser().parse_args()
    generated_pickles = train_and_log_models(args)
    for pickle_path in generated_pickles:
        print(f"Modelo guardado: {pickle_path}")


if __name__ == "__main__":
    main()
