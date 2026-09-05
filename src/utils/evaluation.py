import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric

def evaluate_performance(y_true, y_pred, favorable_label=1.0):
    """
    Calculates the classic Machine Learning performance metrics.

    Args:
        y_true (array-like): True labels (ground truth).
        y_pred (array-like): Labels predicted by the model.
        favorable_label (float/int).

    Returns:
        dict: Dictionary containing Accuracy, Precision, Recall, and F1-Score.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=favorable_label, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=favorable_label, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=favorable_label, zero_division=0)

    return {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-Score":  f1
    }


def evaluate_fairness(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    """
    Calculates fairness metrics using AIF360.

    Args:
        dataset_true (BinaryLabelDataset): Dataset with the true labels.
        dataset_pred (BinaryLabelDataset): Dataset with the predicted labels.
        unprivileged_groups (list).
        privileged_groups (list).

    Returns:
        dict: Dictionary containing the disparity metrics.
    """
    metric = ClassificationMetric(
        dataset_true, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    dp_diff = metric.statistical_parity_difference()
    eo_diff = metric.equal_opportunity_difference()
    ppv_diff = (
        metric.positive_predictive_value(privileged=False)
        - metric.positive_predictive_value(privileged=True)
    )
    npv_diff = (
        metric.negative_predictive_value(privileged=False)
        - metric.negative_predictive_value(privileged=True)
    )
    avg_pv_diff = (ppv_diff + npv_diff) / 2.0
    avg_odds_diff = metric.average_odds_difference()

    return {
        "Demographic Parity Diff.":        dp_diff,
        "Equal Opportunity Diff.":         eo_diff,
        "Predictive Parity Diff.":         ppv_diff,
        "Average Predictive Value Diff.":  avg_pv_diff,
        "Average Odds Diff.":              avg_odds_diff
    }


def evaluate_pipeline(dataset_true, dataset_pred, unprivileged_groups, privileged_groups, pipeline_name="Pipeline"):
    """
    Main function that consolidates Performance and Fairness metrics
    and prints a formatted report.

    Args:
        dataset_true (BinaryLabelDataset): Dataset AIF360 de teste com rótulos reais.
        dataset_pred (BinaryLabelDataset): Dataset AIF360 de teste com rótulos preditos.
        unprivileged_groups (list).
        privileged_groups (list).
        pipeline_name (str): For logging only.

    Returns:
        pd.DataFrame: Um DataFrame de uma linha contendo todas as métricas.
    """
    y_true = dataset_true.labels.ravel()
    y_pred = dataset_pred.labels.ravel()
    favorable_label = dataset_true.favorable_label

    perf_metrics = evaluate_performance(y_true, y_pred, favorable_label)
    fair_metrics = evaluate_fairness(
        dataset_true, dataset_pred, unprivileged_groups, privileged_groups
    )

    # Log prints ---------------------------------------------------------
    print(f"\n{'='*52}")
    print(f" RESULTS: {pipeline_name.upper()}")
    print(f"{'='*52}")

    print("\n--- Performance Metrics ---")
    for k, v in perf_metrics.items():
        print(f"{k+':':<20} {v:.4f}")

    print("\n--- Fairness Metrics (Ideally close to 0.0) ---")
    for k, v in fair_metrics.items():
        print(f"{k+':':<34} {v:+.4f}")

    print(f"{'='*52}\n")
    # --------------------------------------------------------------------

    all_metrics = {"Pipeline": pipeline_name}
    all_metrics.update(perf_metrics)
    all_metrics.update(fair_metrics)

    return pd.DataFrame([all_metrics])
