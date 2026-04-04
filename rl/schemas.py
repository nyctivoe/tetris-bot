from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


TRAINING_SAMPLE_VERSION = 1
REPLAY_SHARD_VERSION = 1
EVAL_METRICS_VERSION = 1


class TrainingSampleV1(TypedDict):
    version: int
    board_tensor: np.ndarray
    piece_ids: np.ndarray
    context_scalars: np.ndarray
    candidate_features: np.ndarray
    candidate_count: int
    policy_target: np.ndarray
    value_target: float
    attack_target: float
    survival_target: float
    surge_target: float
    metadata: dict[str, Any]


class ReplayShardV1(TypedDict):
    version: int
    metadata: dict[str, Any]
    samples: list[TrainingSampleV1]


class EvalMetricsV1(TypedDict):
    version: int
    win_rate: float
    realized_surge_rate: float
    unintentional_b2b_break_rate: float
    difficult_clear_fraction: float
    cancel_efficiency: float
    average_pending_garbage: float
    mean_decision_latency: float


def validate_training_sample(sample: dict[str, Any]) -> TrainingSampleV1:
    if int(sample.get("version", -1)) != TRAINING_SAMPLE_VERSION:
        raise RuntimeError(f"Unsupported training sample version: {sample.get('version')!r}")
    board = np.asarray(sample["board_tensor"], dtype=np.float32)
    pieces = np.asarray(sample["piece_ids"], dtype=np.int64)
    context = np.asarray(sample["context_scalars"], dtype=np.float32)
    candidates = np.asarray(sample["candidate_features"], dtype=np.float32)
    policy = np.asarray(sample["policy_target"], dtype=np.float32)
    if board.shape != (12, 20, 10):
        raise RuntimeError(f"Invalid board_tensor shape: {board.shape!r}")
    if pieces.shape != (7,):
        raise RuntimeError(f"Invalid piece_ids shape: {pieces.shape!r}")
    if context.shape != (28,):
        raise RuntimeError(f"Invalid context_scalars shape: {context.shape!r}")
    if candidates.ndim != 2 or candidates.shape[1] != 32:
        raise RuntimeError(f"Invalid candidate_features shape: {candidates.shape!r}")
    if policy.shape != (candidates.shape[0],):
        raise RuntimeError("policy_target length must match candidate_features rows.")
    return TrainingSampleV1(
        version=TRAINING_SAMPLE_VERSION,
        board_tensor=board,
        piece_ids=pieces,
        context_scalars=context,
        candidate_features=candidates,
        candidate_count=int(sample["candidate_count"]),
        policy_target=policy,
        value_target=float(sample["value_target"]),
        attack_target=float(sample["attack_target"]),
        survival_target=float(sample["survival_target"]),
        surge_target=float(sample["surge_target"]),
        metadata=dict(sample.get("metadata") or {}),
    )


def validate_replay_shard(shard: dict[str, Any]) -> ReplayShardV1:
    if int(shard.get("version", -1)) != REPLAY_SHARD_VERSION:
        raise RuntimeError(f"Unsupported replay shard version: {shard.get('version')!r}")
    samples = [validate_training_sample(sample) for sample in list(shard.get("samples") or [])]
    return ReplayShardV1(
        version=REPLAY_SHARD_VERSION,
        metadata=dict(shard.get("metadata") or {}),
        samples=samples,
    )


def validate_eval_metrics(metrics: dict[str, Any]) -> EvalMetricsV1:
    if int(metrics.get("version", -1)) != EVAL_METRICS_VERSION:
        raise RuntimeError(f"Unsupported eval metrics version: {metrics.get('version')!r}")
    return EvalMetricsV1(
        version=EVAL_METRICS_VERSION,
        win_rate=float(metrics["win_rate"]),
        realized_surge_rate=float(metrics["realized_surge_rate"]),
        unintentional_b2b_break_rate=float(metrics["unintentional_b2b_break_rate"]),
        difficult_clear_fraction=float(metrics["difficult_clear_fraction"]),
        cancel_efficiency=float(metrics["cancel_efficiency"]),
        average_pending_garbage=float(metrics["average_pending_garbage"]),
        mean_decision_latency=float(metrics["mean_decision_latency"]),
    )
