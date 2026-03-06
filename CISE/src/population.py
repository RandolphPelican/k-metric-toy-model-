"""
response_generation.py

Generates synthetic neural population responses for System A and System B.

Each trial:
- Task label T sampled from Bernoulli(0.5)
- Response r ~ MultivariateNormal(mu[T], Sigma_system)

Signal means are identical for both systems.
Only the noise covariance differs.
"""

import numpy as np
from src.noise_geometry import get_task_axis, construct_sigma_A, construct_sigma_B


def get_task_means(task_axis, signal_strength, n_neurons):
    """
    mu[0] = zero vector (baseline)
    mu[1] = signal_strength * task_axis (shifted along task dimension)
    """
    mu_0 = np.zeros(n_neurons)
    mu_1 = signal_strength * task_axis
    return mu_0, mu_1


def generate_responses(sigma, mu_0, mu_1, n_trials, seed=None):
    """
    Generates population responses for one system.

    Returns:
        responses: (n_trials, n_neurons) array
        labels:    (n_trials,) array of task labels {0, 1}
    """
    rng = np.random.default_rng(seed)

    labels = rng.integers(0, 2, size=n_trials)
    responses = np.zeros((n_trials, len(mu_0)))

    for i, label in enumerate(labels):
        mean = mu_0 if label == 0 else mu_1
        responses[i] = rng.multivariate_normal(mean, sigma)

    return responses, labels


def verify_responses(responses_a, labels_a, responses_b, labels_b, task_axis):
    """
    Basic sanity checks on generated responses.
    Confirms signal is present and balanced across systems.
    """
    print("=" * 50)
    print("RESPONSE GENERATION VERIFICATION")
    print("=" * 50)

    for name, responses, labels in [
        ("A", responses_a, labels_a),
        ("B", responses_b, labels_b),
    ]:
        n_trials = len(labels)
        balance = labels.mean()
        mean_0 = responses[labels == 0].mean(axis=0)
        mean_1 = responses[labels == 1].mean(axis=0)
        signal = np.dot(mean_1 - mean_0, task_axis)

        print(f"System {name}:")
        print(f"  Trials:           {n_trials}")
        print(f"  Label balance:    {balance:.3f} (target 0.500)")
        print(f"  Signal along axis:{signal:.3f}")
        print(f"  Response shape:   {responses.shape}")

    print("=" * 50)
    print("Response generation complete.\n")


if __name__ == "__main__":
    N_NEURONS = 50
    N_TRIALS = 10000
    SIGNAL_STRENGTH = 2.0
    NOISE_VARIANCE = 1.0

    task_axis = get_task_axis(N_NEURONS)

    sigma_a = construct_sigma_A(task_axis, NOISE_VARIANCE, N_NEURONS)
    sigma_b = construct_sigma_B(task_axis, NOISE_VARIANCE, N_NEURONS)

    mu_0, mu_1 = get_task_means(task_axis, SIGNAL_STRENGTH, N_NEURONS)

    print("Generating responses for System A...")
    responses_a, labels_a = generate_responses(
        sigma_a, mu_0, mu_1, N_TRIALS, seed=0
    )

    print("Generating responses for System B...")
    responses_b, labels_b = generate_responses(
        sigma_b, mu_0, mu_1, N_TRIALS, seed=1
    )

    verify_responses(responses_a, labels_a, responses_b, labels_b, task_axis)
