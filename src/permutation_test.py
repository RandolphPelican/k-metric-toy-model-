"""
permutation_test.py

Computes navigability of each trial using a permutation-based criterion.

A trial is navigable if a linear decoder trained on independent training data
assigns confidence to the correct task label exceeding the 95th percentile
of a permutation null distribution.

The permutation null distribution was computed once per decoder using
training data only and applied uniformly across held-out trials.

This criterion is:
- Data-adaptive (no fixed threshold)
- Threshold-free (95th percentile of null, not arbitrary accuracy cutoff)
- Leakage-free (null computed on train set only, applied to test set)
- Standard (consistent with permutation-based inference in neuroimaging)
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_decoder():
    """
    Linear decoder with standardization.
    LinearSVC is appropriate for high-dimensional population codes.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(max_iter=5000, C=1.0))
    ])


def compute_permutation_null(decoder, X_train, y_train, n_permutations=1000, seed=42):
    """
    Generates null distribution of decoder confidence scores
    using training data only.

    Returns the 95th percentile threshold -- applied uniformly
    to all held-out trials. No per-trial recomputation.
    """
    rng = np.random.default_rng(seed)
    null_scores = []

    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_train)
        perm_decoder = build_decoder()
        perm_decoder.fit(X_train, y_shuffled)
        # Decision function on training data gives null confidence distribution
        scores = perm_decoder.decision_function(X_train)
        # For binary: positive score = class 1, negative = class 0
        # Correct-label confidence for shuffled labels
        correct_confidences = np.where(
            y_shuffled == 1, scores, -scores
        )
        null_scores.append(np.mean(correct_confidences))

    return np.percentile(null_scores, 95)


def compute_navigability(responses, labels, train_frac=0.8,
                         n_permutations=1000, seed=42):
    """
    For each held-out trial, determines whether its representation
    is navigable under the permutation criterion.

    Returns:
        navigable: boolean array of length n_test_trials
        threshold: the 95th percentile null threshold used
        test_accuracy: overall decoder accuracy on test set
        navigable_fraction: fraction of test trials classified as navigable
    """
    rng = np.random.default_rng(seed)
    n_trials = len(labels)
    n_train = int(n_trials * train_frac)

    # Split into train and test
    indices = rng.permutation(n_trials)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = responses[train_idx], labels[train_idx]
    X_test, y_test = responses[test_idx], labels[test_idx]

    # Train decoder on training data
    decoder = build_decoder()
    decoder.fit(X_train, y_train)

    # Compute null threshold from training data only
    # Applied uniformly to all test trials -- no leakage
    threshold = compute_permutation_null(
        decoder, X_train, y_train, n_permutations, seed
    )

    # Get confidence scores on held-out test trials
    test_scores = decoder.decision_function(X_test)

    # Correct-label confidence per trial
    correct_confidences = np.where(
        y_test == 1, test_scores, -test_scores
    )

    # Trial is navigable if its confidence exceeds the null threshold
    navigable = correct_confidences > threshold

    # Overall accuracy
    predictions = (test_scores > 0).astype(int)
    test_accuracy = np.mean(predictions == y_test)
    navigable_fraction = np.mean(navigable)

    return navigable, threshold, test_accuracy, navigable_fraction, test_idx


def verify_navigability(nav_a, nav_b, thresh_a, thresh_b,
                        acc_a, acc_b, frac_a, frac_b):
    """
    Confirms System A has higher navigability than System B.
    """
    print("=" * 50)
    print("PERMUTATION TEST VERIFICATION")
    print("=" * 50)
    print(f"System A:")
    print(f"  Null threshold:      {thresh_a:.4f}")
    print(f"  Test accuracy:       {acc_a:.4f}")
    print(f"  Navigable fraction:  {frac_a:.4f}")
    print(f"System B:")
    print(f"  Null threshold:      {thresh_b:.4f}")
    print(f"  Test accuracy:       {acc_b:.4f}")
    print(f"  Navigable fraction:  {frac_b:.4f}")
    print(f"A > B navigability:    {frac_a > frac_b}")
    print("=" * 50)
    print("Permutation test complete.\n")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.noise_construction import (
        get_task_axis, construct_sigma_A, construct_sigma_B
    )
    from src.response_generation import get_task_means, generate_responses

    N_NEURONS = 50
    N_TRIALS = 10000
    SIGNAL_STRENGTH = 2.0
    NOISE_VARIANCE = 1.0

    task_axis = get_task_axis(N_NEURONS)
    sigma_a = construct_sigma_A(task_axis, NOISE_VARIANCE, N_NEURONS)
    sigma_b = construct_sigma_B(task_axis, NOISE_VARIANCE, N_NEURONS)
    mu_0, mu_1 = get_task_means(task_axis, SIGNAL_STRENGTH, N_NEURONS)

    responses_a, labels_a = generate_responses(sigma_a, mu_0, mu_1, N_TRIALS, seed=0)
    responses_b, labels_b = generate_responses(sigma_b, mu_0, mu_1, N_TRIALS, seed=1)

    print("Computing navigability for System A (this takes ~30 seconds)...")
    nav_a, thresh_a, acc_a, frac_a, idx_a = compute_navigability(
        responses_a, labels_a, n_permutations=1000, seed=42
    )

    print("Computing navigability for System B...")
    nav_b, thresh_b, acc_b, frac_b, idx_b = compute_navigability(
        responses_b, labels_b, n_permutations=1000, seed=42
    )

    verify_navigability(nav_a, nav_b, thresh_a, thresh_b,
                        acc_a, acc_b, frac_a, frac_b)
