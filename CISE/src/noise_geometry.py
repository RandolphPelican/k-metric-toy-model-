"""
noise_construction.py

Constructs matched noise covariance matrices for System A and System B.

System A: noise principal axes orthogonal to task axis (high K)
System B: noise principal axes aligned with task axis (low K)

Guarantee: trace(Sigma_A) == trace(Sigma_B)
           participation_ratio(Sigma_A) == participation_ratio(Sigma_B)
           det(Sigma_A) != det(Sigma_B)  <-- this is what K captures
"""

import numpy as np


def get_task_axis(n_neurons, seed=42):
    """
    Returns a fixed unit vector representing the task-relevant dimension.
    Seeded for reproducibility.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n_neurons)
    return v / np.linalg.norm(v)


def participation_ratio(sigma):
    """
    Measures effective dimensionality of a covariance matrix.
    PR = (sum of eigenvalues)^2 / sum(eigenvalues^2)
    Equal for A and B by construction -- confirms K divergence
    is not just dimensionality divergence.
    """
    eigenvalues = np.linalg.eigvalsh(sigma)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)


def construct_sigma_A(task_axis, noise_variance, n_neurons):
    """
    System A: noise orthogonal to task axis.
    Signal and noise occupy different subspaces.
    High K -- noise does not scramble task-relevant representations.
    """
    # Start with isotropic noise
    sigma = noise_variance * np.eye(n_neurons)

    # Concentrate noise variance away from task axis
    # by suppressing variance along task axis and
    # redistributing to orthogonal directions
    task_outer = np.outer(task_axis, task_axis)

    # Remove variance along task axis, add it to orthogonal space
    sigma = sigma - (noise_variance * 0.8) * task_outer
    sigma = sigma + (noise_variance * 0.8 / (n_neurons - 1)) * (
        np.eye(n_neurons) - task_outer
    )

    # Ensure positive definite
    sigma = (sigma + sigma.T) / 2
    min_eig = np.linalg.eigvalsh(sigma).min()
    if min_eig < 1e-10:
        sigma += (abs(min_eig) + 1e-10) * np.eye(n_neurons)

    return sigma


def construct_sigma_B(task_axis, noise_variance, n_neurons):
    """
    System B: noise aligned with task axis.
    Signal and noise occupy the same subspace.
    Low K -- noise directly scrambles task-relevant representations.
    """
    # Start with isotropic noise
    sigma = noise_variance * np.eye(n_neurons)

    # Concentrate noise variance along task axis
    task_outer = np.outer(task_axis, task_axis)

    # Add extra variance along task axis, remove from orthogonal space
    sigma = sigma + (noise_variance * 0.8) * task_outer
    sigma = sigma - (noise_variance * 0.8 / (n_neurons - 1)) * (
        np.eye(n_neurons) - task_outer
    )

    # Ensure positive definite
    sigma = (sigma + sigma.T) / 2
    min_eig = np.linalg.eigvalsh(sigma).min()
    if min_eig < 1e-10:
        sigma += (abs(min_eig) + 1e-10) * np.eye(n_neurons)

    return sigma


def verify_matching(sigma_a, sigma_b, tolerance=0.01):
    """
    Confirms that Sigma_A and Sigma_B are matched on classical metrics.
    Trace and participation ratio must be equal.
    Determinant must differ -- that is the point.
    """
    trace_a = np.trace(sigma_a)
    trace_b = np.trace(sigma_b)
    pr_a = participation_ratio(sigma_a)
    pr_b = participation_ratio(sigma_b)
    det_a = np.linalg.slogdet(sigma_a)[1]  # log determinant for stability
    det_b = np.linalg.slogdet(sigma_b)[1]

    print("=" * 50)
    print("NOISE CONSTRUCTION VERIFICATION")
    print("=" * 50)
    print(f"Trace A:              {trace_a:.4f}")
    print(f"Trace B:              {trace_b:.4f}")
    print(f"Trace match:          {abs(trace_a - trace_b) < tolerance}")
    print(f"PR A:                 {pr_a:.4f}")
    print(f"PR B:                 {pr_b:.4f}")
    print(f"PR match:             {abs(pr_a - pr_b) < tolerance * 10}")
    print(f"Log-det A:            {det_a:.4f}")
    print(f"Log-det B:            {det_b:.4f}")
    print(f"Det differs:          {abs(det_a - det_b) > tolerance}")
    print("=" * 50)

    assert abs(trace_a - trace_b) < tolerance, "FAIL: traces do not match"
    assert abs(pr_a - pr_b) < tolerance * 10, "FAIL: participation ratios do not match"
    assert abs(det_a - det_b) > tolerance, "FAIL: determinants should differ"

    print("All checks passed.\n")
    return True


if __name__ == "__main__":
    N_NEURONS = 50
    NOISE_VARIANCE = 1.0

    task_axis = get_task_axis(N_NEURONS)
    sigma_a = construct_sigma_A(task_axis, NOISE_VARIANCE, N_NEURONS)
    sigma_b = construct_sigma_B(task_axis, NOISE_VARIANCE, N_NEURONS)
    verify_matching(sigma_a, sigma_b)
