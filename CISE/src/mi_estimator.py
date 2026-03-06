"""
mi_estimation.py

Estimates mutual information between task labels and population responses
by projecting onto the known task axis.

This is the correct approach for this toy model because:
- The task axis is known by construction
- PCA finds maximum variance which is the NOISE axis in System B
- Projecting onto the task axis gives the signal component directly
- Raw MI should be nearly equal for A and B (same signal, different noise)
- Navigable MI should be higher for A than B
"""

import numpy as np


def project_onto_task_axis(responses, task_axis):
    """
    Projects population responses onto the task-relevant dimension.
    Returns 1D array of projections.
    """
    task_axis_normalized = task_axis / np.linalg.norm(task_axis)
    return responses @ task_axis_normalized


def estimate_mi_binning(r_1d, labels, n_bins=20):
    """
    Estimates MI using binning on a 1D projection.
    I(T; R) = H(T) - H(T|R)
    Returns MI in nats.
    """
    n_trials = len(labels)

    bins = np.linspace(r_1d.min(), r_1d.max(), n_bins + 1)
    bin_assignments = np.digitize(r_1d, bins) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    p_label = np.array([
        np.mean(labels == 0),
        np.mean(labels == 1)
    ])

    # H(T)
    h_t = -np.sum(p_label[p_label > 0] * np.log(p_label[p_label > 0]))

    # H(T|R)
    h_t_given_r = 0.0
    for b in range(n_bins):
        mask = bin_assignments == b
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        p_bin = n_in_bin / n_trials
        labels_in_bin = labels[mask]
        for label in [0, 1]:
            p_label_given_bin = np.mean(labels_in_bin == label)
            if p_label_given_bin > 0:
                h_t_given_r -= p_bin * p_label_given_bin * np.log(
                    p_label_given_bin
                )

    return max(h_t - h_t_given_r, 0.0)


def compute_raw_and_navigable_mi(responses, labels, navigable_mask,
                                  test_idx, task_axis, n_bins=20):
    """
    Computes raw MI and navigable MI for a system.

    Projects onto task axis for consistent comparison across systems.
    Raw MI uses all trials.
    Navigable MI uses navigable test trials only.
    """
    # Project all trials onto task axis
    r_1d_all = project_onto_task_axis(responses, task_axis)

    # Raw MI over all trials
    mi_raw = estimate_mi_binning(r_1d_all, labels, n_bins)

    # Project test trials
    r_1d_test = r_1d_all[test_idx]
    test_labels = labels[test_idx]

    # Navigable MI over navigable test trials only
    navigable_r = r_1d_test[navigable_mask]
    navigable_labels = test_labels[navigable_mask]

    if len(navigable_labels) < 50:
        mi_navigable = 0.0
    else:
        mi_navigable = estimate_mi_binning(
            navigable_r, navigable_labels, n_bins
        )

    return mi_raw, mi_navigable


def verify_mi(mi_raw_a, mi_nav_a, mi_raw_b, mi_nav_b):
    """
    Confirms:
    - Raw MI approximately equal for A and B
    - Navigable MI higher for A than B
    """
    print("=" * 50)
    print("MUTUAL INFORMATION VERIFICATION")
    print("=" * 50)
    print(f"System A:")
    print(f"  Raw MI:             {mi_raw_a:.4f} nats")
    print(f"  Navigable MI:       {mi_nav_a:.4f} nats")
    print(f"System B:")
    print(f"  Raw MI:             {mi_raw_b:.4f} nats")
    print(f"  Navigable MI:       {mi_nav_b:.4f} nats")
    print(f"Raw MI match:         {abs(mi_raw_a - mi_raw_b) < 0.05}")
    print(f"Nav MI A > B:         {mi_nav_a > mi_nav_b}")
    print("=" * 50)
    print("MI estimation complete.\n")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.noise_construction import (
        get_task_axis, construct_sigma_A, construct_sigma_B
    )
    from src.response_generation import get_task_means, generate_responses
    from src.permutation_test import compute_navigability

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

    print("Estimating mutual information...")
    mi_raw_a, mi_nav_a = compute_raw_and_navigable_mi(
        responses_a, labels_a, nav_a, idx_a, task_axis
    )
    mi_raw_b, mi_nav_b = compute_raw_and_navigable_mi(
        responses_b, labels_b, nav_b, idx_b, task_axis
    )

    verify_mi(mi_raw_a, mi_nav_a, mi_raw_b, mi_nav_b)
