"""排序模型的 AUC、GAUC 与 LogLoss 指标。"""

from typing import Dict, Mapping, Sequence

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def sigmoid(logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    probabilities = np.empty_like(values)
    positive = values >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1.0 + exp_values)
    return probabilities


def gauc_score(
    targets: Sequence[float],
    probabilities: Sequence[float],
    user_ids: Sequence[int],
) -> float:
    targets_array = np.asarray(targets).reshape(-1)
    probabilities_array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    user_ids_array = np.asarray(user_ids).reshape(-1)
    if not (targets_array.size == probabilities_array.size == user_ids_array.size):
        raise ValueError(
            "targets, probabilities and user_ids must have the same length"
        )
    if targets_array.size == 0:
        return float("nan")

    order = np.argsort(user_ids_array, kind="stable")
    sorted_users = user_ids_array[order]
    sorted_targets = targets_array[order]
    sorted_probabilities = probabilities_array[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sorted_users.size]))

    weighted_auc = 0.0
    total_weight = 0
    for start, end in zip(starts, ends):
        group_targets = sorted_targets[start:end]
        if np.unique(group_targets).size < 2:
            continue
        weight = end - start
        weighted_auc += (
            float(roc_auc_score(group_targets, sorted_probabilities[start:end]))
            * weight
        )
        total_weight += weight

    if total_weight == 0:
        return float("nan")
    return weighted_auc / total_weight


def _finite_mean(values: Sequence[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def compute_multitask_metrics(
    targets: Mapping[str, Sequence[float]],
    logits: Mapping[str, Sequence[float]],
    user_ids: Sequence[int],
    task_names: Sequence[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    auc_values = []
    gauc_values = []
    logloss_values = []

    for task in task_names:
        task_targets = np.asarray(targets[task]).reshape(-1)
        task_probabilities = sigmoid(logits[task])
        if task_targets.size != task_probabilities.size:
            raise ValueError(
                f"targets and logits for task '{task}' must have the same length"
            )
        if task_targets.size == 0:
            task_logloss = float("nan")
            task_auc = float("nan")
        else:
            task_logloss = float(
                log_loss(task_targets, task_probabilities, labels=[0, 1])
            )
            task_auc = (
                float(roc_auc_score(task_targets, task_probabilities))
                if np.unique(task_targets).size >= 2
                else float("nan")
            )
        task_gauc = gauc_score(task_targets, task_probabilities, user_ids)

        metrics[f"{task}_auc"] = task_auc
        metrics[f"{task}_gauc"] = task_gauc
        metrics[f"{task}_logloss"] = task_logloss
        auc_values.append(task_auc)
        gauc_values.append(task_gauc)
        logloss_values.append(task_logloss)

    metrics["mean_auc"] = _finite_mean(auc_values)
    metrics["mean_gauc"] = _finite_mean(gauc_values)
    metrics["mean_logloss"] = _finite_mean(logloss_values)
    return metrics
