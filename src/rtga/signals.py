from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmpiricalRiskCalibrator:
    """Mahalanobis-style empirical risk calibration for diagnostic features."""

    center: np.ndarray
    precision: np.ndarray
    threshold: float
    eps: float = 1e-8

    @classmethod
    def fit(cls, features: np.ndarray, quantile: float = 0.90, eps: float = 1e-6) -> "EmpiricalRiskCalibrator":
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[0] < 2:
            raise ValueError("risk calibration requires at least two feature vectors")
        center = values.mean(axis=0)
        cov = np.cov(values.T)
        cov = np.atleast_2d(cov) + np.eye(values.shape[1]) * eps
        precision = np.linalg.pinv(cov)
        scores = cls(center=center, precision=precision, threshold=0.0, eps=eps).score_many(values)
        threshold = float(np.quantile(scores, quantile))
        return cls(center=center, precision=precision, threshold=threshold, eps=eps)

    def score(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=float).reshape(1, -1)
        return float(self.score_many(values)[0])

    def score_many(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        centered = values - self.center.reshape(1, -1)
        raw = np.einsum("ij,jk,ik->i", centered, self.precision, centered)
        return np.sqrt(np.maximum(raw, 0.0))


def risk_to_weight(score: float, threshold: float, slope: float = 1.0, floor: float = 0.05) -> float:
    """Map a risk score to a sparse repair weight."""

    if score < threshold:
        return 0.0
    denom = max(abs(threshold) * max(slope, 1e-8), 1e-8)
    return float(np.clip((score - threshold) / denom, floor, 1.0))

