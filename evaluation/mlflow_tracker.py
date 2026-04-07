"""
MLflow tracking for FinSight agent evaluation.

Logs citation accuracy, hallucination rate, and faithfulness scores
as MLflow metrics under the configured experiment.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, Generator

import mlflow

from config.settings import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from evaluation.metrics import evaluate_response

logger = logging.getLogger(__name__)


def _setup_mlflow() -> None:
    """Configure MLflow tracking URI and ensure the experiment exists."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


@contextmanager
def mlflow_run(run_name: str | None = None) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager that wraps an MLflow run with automatic setup."""
    _setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_evaluation(
    query: str,
    report: str,
    citations: list,
    retrieved_docs_text: list,
    extra_params: Dict | None = None,
    run_name: str | None = None,
) -> Dict:
    """
    Evaluate a research response and log all metrics to MLflow.

    Parameters
    ----------
    query:               Original user query.
    report:              Generated research report text.
    citations:           Citation list from the pipeline.
    retrieved_docs_text: Raw text of retrieved documents (for grounding checks).
    extra_params:        Additional key-value pairs to log as MLflow params.
    run_name:            Optional name for the MLflow run.

    Returns
    -------
    The evaluation results dict (same as evaluate_response output).
    """
    results = evaluate_response(query, report, citations, retrieved_docs_text)

    with mlflow_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("query", query[:250])  # MLflow param limit
        mlflow.log_param("num_retrieved_docs", len(retrieved_docs_text))
        if extra_params:
            for key, value in extra_params.items():
                mlflow.log_param(key, value)

        # Log metrics
        mlflow.log_metric("citation_accuracy", results["citation_accuracy"])
        mlflow.log_metric("hallucination_rate", results["hallucination_rate"])
        mlflow.log_metric("faithfulness", results["faithfulness"])
        mlflow.log_metric("num_citations", results["num_citations"])
        mlflow.log_metric("report_length_chars", results["report_length_chars"])

        logger.info(
            "[MLflow] run_id=%s  citation_accuracy=%.4f  "
            "hallucination_rate=%.4f  faithfulness=%.4f",
            run.info.run_id,
            results["citation_accuracy"],
            results["hallucination_rate"],
            results["faithfulness"],
        )

    return results


def get_experiment_summary() -> Dict:
    """
    Return a summary of all runs in the configured experiment.

    Includes average citation_accuracy, hallucination_rate, and faithfulness.
    """
    _setup_mlflow()
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return {"error": "Experiment not found"}

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    if runs.empty:
        return {"total_runs": 0}

    metrics = ["citation_accuracy", "hallucination_rate", "faithfulness"]
    summary: Dict = {"total_runs": len(runs)}
    for metric in metrics:
        col = f"metrics.{metric}"
        if col in runs.columns:
            summary[f"avg_{metric}"] = round(float(runs[col].mean()), 4)
    return summary
