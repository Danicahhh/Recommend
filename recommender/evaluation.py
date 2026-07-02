"""排序与召回模型的离线评价指标。"""

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


def compute_topk_retrieval_metrics(
    recommendations: Mapping[int, Sequence[int]],
    relevant_items: Mapping[int, Sequence[int]],
    ks: Sequence[int],
) -> Dict[str, float]:
    """计算用户级宏平均 Recall@K、HitRate@K 和 NDCG@K。

    只评估至少有一个相关物品的用户。推荐列表和相关集合中的重复物品均按
    一个物品处理，避免重复召回被重复计分。
    """
    normalized_ks = sorted({int(k) for k in ks})
    if not normalized_ks or any(k <= 0 for k in normalized_ks):
        raise ValueError("ks must contain at least one positive integer")

    user_values = {
        k: {"recall": [], "hit_rate": [], "ndcg": []} for k in normalized_ks
    }
    evaluated_users = 0

    for user_id, raw_relevant in relevant_items.items():
        relevant = set(raw_relevant)
        if not relevant:
            continue

        ranked_items = list(dict.fromkeys(recommendations.get(user_id, ())))
        evaluated_users += 1
        for k in normalized_ks:
            top_items = ranked_items[:k]
            hit_positions = [
                rank
                for rank, item_id in enumerate(top_items, start=1)
                if item_id in relevant
            ]
            hit_count = len(hit_positions)
            dcg = sum(1.0 / np.log2(rank + 1.0) for rank in hit_positions)
            ideal_hits = min(k, len(relevant))
            idcg = sum(
                1.0 / np.log2(rank + 1.0)
                for rank in range(1, ideal_hits + 1)
            )

            user_values[k]["recall"].append(hit_count / len(relevant))
            user_values[k]["hit_rate"].append(float(hit_count > 0))
            user_values[k]["ndcg"].append(dcg / idcg if idcg > 0 else 0.0)

    metrics: Dict[str, float] = {"evaluated_users": float(evaluated_users)}
    for k in normalized_ks:
        for metric_name, values in user_values[k].items():
            metrics[f"{metric_name}@{k}"] = (
                float(np.mean(values)) if values else float("nan")
            )
    return metrics
