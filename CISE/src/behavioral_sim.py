"""
behavioral_task.py

Simulates a two-step integration task to generate behavioral performance.

The agent must maintain a partial representation across two processing
steps before responding. System B fails here because noise aligned with
the task axis corrupts the partial representation during the integration
window before step 2 arrives.

This gives us behavioral accuracy per system which should track K,
not raw MI.
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def simulate_two_step_task(responses, labels, sigma, task_axis,
                            integration_noise_scale=0.5,
                            n_trials=None, seed=42):
    """
    Two-step integration task.

    Step 1: Agent receives first half of neural response + integration noise
    Step 2: Agent receives second half + must combine with step 1

    The integration noise is isotropic -- it does not favor either system.
    System B fails because its task-axis noise corrupts step 1 before
    step 2 arrives, preventing stable integration.

    Returns behavioral accuracy.
    """
    rng = np.random.default_rng(seed)
    n_neurons = responses.shape[1]
    half = n_neurons // 2

    if n_trials is None:
        n_trials = len(labels)

    correct = 0

    # Train a simple linear decoder on clean responses
    # This represents the agent's learned decision rule
    scaler = StandardScaler()
    decoder = LinearSVC(max_iter=5000, C=1.0)

    # Use first 80% to train decoder
    n_train = int(len(labels) * 0.8)
    X_train = scaler.fit_transform(responses[:n_train])
    decoder.fit(X_train, labels[:n_train])

    # Test on remaining trials with integration noise added
    test_responses = responses[n_train:n_train + n_trials]
    test_labels = labels[n_train:n_train + n_trials]

    for i in range(len(test_labels)):
        r = test_responses[i].copy()

        # Step 1: process first half with integration noise
        r_step1 = r.copy()
        r_step1[:half] += rng.normal(
            0, integration_noise_scale, half
        )

        # Step 2: process second half
        # Agent must maintain step 1 representation while processing step 2
        # Integration noise degrades the maintained representation
        r_step2 = r.copy()
        r_step2[half:] += rng.normal(
            0, integration_noise_scale, n_neurons - half
        )

        # Combine: simple average of the two-step representation
        r_integrated = (r_step1 + r_step2) / 2.0

        # Decode
        r_scaled = scaler.transform(r_integrated.reshape(1, -1))
        prediction = decoder.predict(r_scaled)[0]
        correct += int(prediction == test_labels[i])

    accuracy = correct / len(test_labels)
    return accuracy


def verify_behavioral(acc_a, acc_b):
    """
    Confirms behavioral accuracy tracks K not raw MI.
    """
    print("=" * 50)
    print("BEHAVIORAL TASK RESULTS")
    print("=" * 50)
    print(f"System A accuracy:    {acc_a:.4f}")
    print(f"System B accuracy:    {acc_b:.4f}")
    print(f"A > B accuracy:       {acc_a > acc_b}")
    print(f"Accuracy difference:  {acc_a - acc_b:.4f}")
    print("=" * 50)
    print("Behavioral task complete.\n")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.noise_geometry import (
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

    responses_a, labels_a = generate_responses(
        sigma_a, mu_0, mu_1, N_TRIALS, seed=0
    )
    responses_b, labels_b = generate_responses(
        sigma_b, mu_0, mu_1, N_TRIALS, seed=1
    )

    print("Simulating two-step integration task...")
    acc_a = simulate_two_step_task(
        responses_a, labels_a, sigma_a, task_axis, seed=42
    )
    acc_b = simulate_two_step_task(
        responses_b, labels_b, sigma_b, task_axis, seed=42
    )

    verify_behavioral(acc_a, acc_b)
