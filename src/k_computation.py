"""
k_computation.py

Computes the coherent information fraction K for each system.

K = I_navigable / H_max

Where:
- I_navigable = MI between task labels and navigable population responses
- H_max = maximum entropy of the task-axis projection of the noise
          distribution. Computed in 1D to match the MI estimation
          which also operates on the task-axis projection.

H_max = 0.5 * log(2 * pi * e * sigma²_task)

Where sigma²_task = task_axis^T @ Sigma @ task_axis

System B has higher noise variance along the task axis by construction,
yielding higher H_max and lower K despite similar navigable MI.
This is the core theoretical result.
"""

import numpy as np


def compute_h_max(sigma, task_axis):
    """
    Maximum entropy of the task-axis projection of the noise distribution.

    Computed in 1D to match MI estimation which uses task-axis projection.
    System A: low noise along task axis -> low H_max -> higher K
    System B: high noise along task axis -> high H_max -> lower K

    Returns entropy in nats.
    """
    task_axis_normalized = task_axis / np.linalg.norm(task_axis)
    var_task = task_axis_normalized @ sigma @ task_axis_normalized
    h_max = 0.5 * np.log(2 * np.pi * np.e * var_task)
    return h_max


def compute_k(mi_navigable, h_max):
    """
    K = I_navigable / H_max

    Clipped to [0, 1] -- K is a fraction of usable capacity.
    Values above 1 would indicate estimation error.
    """
    if h_max <= 0:
        raise ValueError("H_max must be positive")
    return np.clip(mi_navigable / h_max, 0.0, 1.0)


def verify_k(k_a, k_b, h_max_a, h_max_b,
             mi_nav_a, mi_nav_b,
             mi_raw_a, mi_raw_b):
    """
    Confirms K_A > K_B and reports full breakdown.
    """
    print("=" * 50)
    print("K COMPUTATION RESULTS")
    print("=" * 50)
    print(f"System A:")
    print(f"  H_max (1D task):    {h_max_a:.4f} nats")
    print(f"  Raw MI:             {mi_raw_a:.4f} nats")
    print(f"  Navigable MI:       {mi_nav_a:.4f} nats")
    print(f"  K:                  {k_a:.4f}")
    print(f"System B:")
    print(f"  H_max (1D task):    {h_max_b:.4f} nats")
    print(f"  Raw MI:             {mi_raw_b:.4f} nats")
    print(f"  Navigable MI:       {mi_nav_b:.4f} nats")
    print(f"  K:                  {k_b:.4f}")
    print(f"K_A > K_B:            {k_a > k_b}")
    print(f"K difference:         {k_a - k_b:.4f}")
    print("=" * 50)
    print("K computation complete.\n")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.noise_construction import (
        get_task_axis, construct_sigma_A, construct_sigma_B
    )
    from src.response_generation import get_task_means, generate_responses
    from src.permutation_test import compute_navigability
    from src.mi_estimation import compute_raw_and_navigable_mi

    N_NEURONS = 50
    N_TRIALS = 10000
    SIGNAL_STRENGTH = 2.0
    NOISE_VARIANCE = 1.0

    task_axis = get_task_axis(N_NEURONS)
    sigma_a = construct_sigma_A(task_axis, NOISE_VARIANCE, N_NEURONS)
    sigma_b = construct_sigma_B(task_axis, NOISE_VARIANCE, N_NEURONS)
    mu_0, mu_1 = get_task_means(task_axis, SIGNAL_STRENGTH, N_NEURONS)

    responses_a, labels_a = generate_responses(
        sigma_a, mu_0, mu_1, N_TRIALS, seed=0
    )
    responses_b, labels_b = generate_responses(
        sigma_b, mu_0, mu_1, N_TRIALS, seed=1
    )

    print("Computing navigability...")
    nav_a, thresh_a, acc_a, frac_a, idx_a = compute_navigability(
        responses_a, labels_a, n_permutations=1000, seed=42
    )
    nav_b, thresh_b, acc_b, frac_b, idx_b = compute_navigability(
        responses_b, labels_b, n_permutations=1000, seed=42
    )

    print("Estimating MI...")
    mi_raw_a, mi_nav_a = compute_raw_and_navigable_mi(
        responses_a, labels_a, nav_a, idx_a, task_axis
    )
    mi_raw_b, mi_nav_b = compute_raw_and_navigable_mi(
        responses_b, labels_b, nav_b, idx_b, task_axis
    )

    print("Computing K...")
    h_max_a = compute_h_max(sigma_a, task_axis)
    h_max_b = compute_h_max(sigma_b, task_axis)

    k_a = compute_k(mi_nav_a, h_max_a)
    k_b = compute_k(mi_nav_b, h_max_b)

    verify_k(k_a, k_b, h_max_a, h_max_b,
             mi_nav_a, mi_nav_b,
             mi_raw_a, mi_raw_b)
