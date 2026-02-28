"""
plotting.py

Generates the money figure for the K metric toy model.

Four panel figure:
Panel 1: Raw MI comparison (A vs B) -- nearly equal, blind to performance
Panel 2: Navigable MI comparison (A vs B) -- partial separation
Panel 3: K comparison (A vs B) -- full separation
Panel 4: Behavioral accuracy (A vs B) -- tracks K not raw MI

This figure is the core empirical demonstration that K captures
something classical metrics cannot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def plot_results(mi_raw_a, mi_raw_b,
                 mi_nav_a, mi_nav_b,
                 k_a, k_b,
                 acc_a, acc_b,
                 save_path="figures/k_metric_results.png"):
    """
    Generates and saves the four-panel results figure.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)

    colors = {
        "A": "#2196F3",  # blue -- high K
        "B": "#F44336",  # red -- low K
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "K Metric Toy Model: Noise Direction Determines Cognitive Capacity\n"
        "Same signal strength, same dimensionality, same total noise volume",
        fontsize=13, y=1.02
    )

    # Panel 1: Raw MI
    ax = axes[0]
    bars = ax.bar(
        ["System A\n(Noise ⊥ Task)", "System B\n(Noise ∥ Task)"],
        [mi_raw_a, mi_raw_b],
        color=[colors["A"], colors["B"]],
        width=0.5, edgecolor="black", linewidth=0.8
    )
    ax.set_title("Raw Mutual Information", fontweight="bold")
    ax.set_ylabel("MI (nats)")
    ax.set_ylim(0, max(mi_raw_a, mi_raw_b) * 1.3)
    for bar, val in zip(bars, [mi_raw_a, mi_raw_b]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.annotate("Classical metric:\nblind to noise direction",
                xy=(0.5, 0.85), xycoords="axes fraction",
                ha="center", fontsize=9, style="italic",
                color="gray")

    # Panel 2: Navigable MI
    ax = axes[1]
    bars = ax.bar(
        ["System A\n(Noise ⊥ Task)", "System B\n(Noise ∥ Task)"],
        [mi_nav_a, mi_nav_b],
        color=[colors["A"], colors["B"]],
        width=0.5, edgecolor="black", linewidth=0.8
    )
    ax.set_title("Navigable Mutual Information", fontweight="bold")
    ax.set_ylabel("MI (nats)")
    ax.set_ylim(0, max(mi_nav_a, mi_nav_b) * 1.3)
    for bar, val in zip(bars, [mi_nav_a, mi_nav_b]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.annotate("Partial separation\nafter navigability filter",
                xy=(0.5, 0.85), xycoords="axes fraction",
                ha="center", fontsize=9, style="italic",
                color="gray")

    # Panel 3: K
    ax = axes[2]
    bars = ax.bar(
        ["System A\n(Noise ⊥ Task)", "System B\n(Noise ∥ Task)"],
        [k_a, k_b],
        color=[colors["A"], colors["B"]],
        width=0.5, edgecolor="black", linewidth=0.8
    )
    ax.set_title("Coherent Information\nFraction K", fontweight="bold")
    ax.set_ylabel("K (unitless, 0-1)")
    ax.set_ylim(0, 1.2)
    for bar, val in zip(bars, [k_a, k_b]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.annotate("K captures full\ncapacity difference",
                xy=(0.5, 0.85), xycoords="axes fraction",
                ha="center", fontsize=9, style="italic",
                color="gray")

    # Panel 4: Behavioral Accuracy
    ax = axes[3]
    bars = ax.bar(
        ["System A\n(Noise ⊥ Task)", "System B\n(Noise ∥ Task)"],
        [acc_a, acc_b],
        color=[colors["A"], colors["B"]],
        width=0.5, edgecolor="black", linewidth=0.8
    )
    ax.set_title("Behavioral Accuracy\n(Two-Step Integration Task)",
                 fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.2)
    ax.axhline(0.5, color="black", linestyle="--",
               linewidth=0.8, label="Chance")
    for bar, val in zip(bars, [acc_a, acc_b]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.annotate("Tracks K,\nnot raw MI",
                xy=(0.5, 0.85), xycoords="axes fraction",
                ha="center", fontsize=9, style="italic",
                color="gray")

    # Legend
    patch_a = mpatches.Patch(color=colors["A"],
                              label="System A: Noise ⊥ Task Axis (High K)")
    patch_b = mpatches.Patch(color=colors["B"],
                              label="System B: Noise ∥ Task Axis (Low K)")
    fig.legend(handles=[patch_a, patch_b],
               loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08),
               fontsize=11, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.close()


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
    from src.k_computation import compute_h_max, compute_k
    from src.behavioral_task import simulate_two_step_task

    N_NEURONS = 50
    N_TRIALS = 10000
    SIGNAL_STRENGTH = 2.0
    NOISE_VARIANCE = 1.0

    print("Building all components...")

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

    print("Simulating behavioral task...")
    beh_acc_a = simulate_two_step_task(
        responses_a, labels_a, sigma_a, task_axis, seed=42
    )
    beh_acc_b = simulate_two_step_task(
        responses_b, labels_b, sigma_b, task_axis, seed=42
    )

    print("Generating figure...")
    plot_results(
        mi_raw_a, mi_raw_b,
        mi_nav_a, mi_nav_b,
        k_a, k_b,
        beh_acc_a, beh_acc_b
    )
