
"""
Model verification and comparative evaluation.

This module compares predictions from two models evaluated on the
same labelled test set. It does not itself perform HE training.
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def wilson_confidence_interval(correct, total, z=1.95996):
    """Calculate the 95% Wilson confidence interval for accuracy."""
    if total == 0:
        return [0.0, 0.0]

    p = correct / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = (
        z * np.sqrt(
            (p * (1 - p) + z**2 / (4 * total)) / total
        ) / denominator
    )

    return [
        max(0.0, float(centre - margin)),
        min(1.0, float(centre + margin)),
    ]


def evaluate_predictions(y_true, y_pred, class_names=None):
    """
    Calculate classification metrics for one model.

    y_true: Ground-truth class labels.
    y_pred: Predicted class labels.
    class_names: Optional list of class names.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Labels and predictions must be 1D arrays.")

    if len(y_true) == 0:
        raise ValueError("Test dataset cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("Labels and predictions must have equal lengths.")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Labels and predictions must be finite.")

    accuracy = float(accuracy_score(y_true, y_pred))
    correct = int(np.sum(y_true == y_pred))
    total = len(y_true)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)) if class_names else None,
        average=None,
        zero_division=0,
    )

    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels = (
        np.arange(len(class_names))
        if class_names
        else np.unique(np.concatenate([y_true, y_pred]))
    )

    names = (
        list(class_names)
        if class_names
        else [str(label) for label in labels]
    )

    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "accuracy_percentage": accuracy * 100,
        "accuracy_ci_95": wilson_confidence_interval(correct, total),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": {
            names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(names))
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }


def compare_models(
    y_true,
    plaintext_predictions,
    comparison_predictions,
    class_names=None,
    plaintext_inference_time_s=None,
    comparison_inference_time_s=None,
):
    """
    Compare two prediction sets on identical test samples.

    The comparison model could be an FHE-compatible model or an
    encrypted-inference model. The actual method must be documented
    separately using training provenance.
    """
    y_true = np.asarray(y_true)
    plain = np.asarray(plaintext_predictions)
    comparison = np.asarray(comparison_predictions)

    if not (len(y_true) == len(plain) == len(comparison)):
        raise ValueError("All prediction arrays must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Test dataset cannot be empty.")

    plain_metrics = evaluate_predictions(y_true, plain, class_names)
    comparison_metrics = evaluate_predictions(
        y_true, comparison, class_names
    )

    agreement = float(np.mean(plain == comparison))

    result = {
        "experiment": "plaintext_vs_comparison_model",
        "evaluation_samples": len(y_true),
        "plaintext": plain_metrics,
        "comparison_model": comparison_metrics,
        "prediction_agreement": agreement,
        "prediction_agreement_percentage": agreement * 100,
        "accuracy_difference_percentage_points": (
            plain_metrics["accuracy_percentage"]
            - comparison_metrics["accuracy_percentage"]
        ),
    }

    if plaintext_inference_time_s is not None:
        result["plaintext_inference_time_s"] = float(
            plaintext_inference_time_s
        )

    if comparison_inference_time_s is not None:
        result["comparison_inference_time_s"] = float(
            comparison_inference_time_s
        )

    if (
        plaintext_inference_time_s is not None
        and comparison_inference_time_s is not None
        and plaintext_inference_time_s > 0
    ):
        result["inference_time_ratio"] = (
            comparison_inference_time_s / plaintext_inference_time_s
        )

    return result


def save_results(results, output_path):
    """Save experimental results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    return str(output_path)
