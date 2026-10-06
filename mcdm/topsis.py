"""TOPSIS decision making for selecting one architecture from a Pareto front."""

from __future__ import annotations

import numpy as np


def topsis(
    matrix: np.ndarray,
    weights: np.ndarray,
    benefit: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return the TOPSIS winner and closeness coefficients.

    Parameters
    ----------
    matrix:
        Shape (n_solutions, n_criteria). All values must be finite.
    weights:
        Positive criterion weights. They are normalized to sum to one.
    benefit:
        Boolean array. True means larger is better; False means smaller is better.

    Returns
    -------
    winner_index, coefficients, weighted_normalized_matrix
    """
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    benefit = np.asarray(benefit, dtype=bool)

    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("TOPSIS requires a non-empty 2-D decision matrix.")
    if matrix.shape[1] != len(weights) or len(weights) != len(benefit):
        raise ValueError("Criteria, weights and benefit flags must have the same length.")
    if not np.isfinite(matrix).all():
        raise ValueError("TOPSIS matrix contains non-finite values.")
    if np.any(weights <= 0):
        raise ValueError("TOPSIS weights must be strictly positive.")

    weights = weights / weights.sum()

    # Vector normalization prevents SynFlow and parameter count from being
    # compared in their raw, incompatible units.
    denominators = np.linalg.norm(matrix, axis=0)
    if np.any(denominators == 0):
        # A constant criterion carries no decision information.
        denominators = np.where(denominators == 0, 1.0, denominators)

    normalized = matrix / denominators
    weighted = normalized * weights

    ideal_positive = np.empty(matrix.shape[1], dtype=float)
    ideal_negative = np.empty(matrix.shape[1], dtype=float)

    for j, is_benefit in enumerate(benefit):
        if is_benefit:
            ideal_positive[j] = weighted[:, j].max()
            ideal_negative[j] = weighted[:, j].min()
        else:
            ideal_positive[j] = weighted[:, j].min()
            ideal_negative[j] = weighted[:, j].max()

    distance_positive = np.linalg.norm(weighted - ideal_positive, axis=1)
    distance_negative = np.linalg.norm(weighted - ideal_negative, axis=1)

    coefficients = distance_negative / (
        distance_positive + distance_negative + np.finfo(float).eps
    )

    winner_index = int(np.argmax(coefficients))
    return winner_index, coefficients, weighted
