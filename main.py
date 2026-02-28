"""
main.py

K Metric Toy Model -- Full Pipeline

Demonstrates that the Coherent Information Fraction K predicts
behavioral performance in a two-alternative forced choice task
where classical metrics (raw MI, dimensionality) are blind to
the critical variable: noise direction relative to the task axis.

Experimental Design:
- System A: noise orthogonal to task axis (high K)
- System B: noise aligned with task axis (low K)
- Matched on: signal strength, dimensionality, total noise volume
- Differs on: noise direction only

Result:
- K_A >> K_B
- Behavioral accuracy tracks K, not raw MI
- Classical metrics fail to predict the performance gap

Usage:
    python3 main.py

Output:
    figures/k_metric_results.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.noise_construction import (
    get_task_axis, construct_sigma_A, construct_sigma_B, verify_matching
)
from src.response_generation import get_task_means, generate_responses
from src.permutation_test import compute_navigability, verify_navigability
from src.mi_estimation import compute_raw_and_navigable_mi, verify_mi
from src.k_computation import compute_h_max, compute_k, verify_k
from src.behavioral_task import simulate_two_step_task, verify_behavioral
from src.plotting import plot_results

# ============================================================
# PARAMETERS
# ============================================================
N_NEURONS = 50
N_TRIALS = 10000
SIGNAL_STRENGTH = 2.0
NOISE_VARIANCE = 1.0
N_PERMUTATIONS = 1000
SEED_A = 0
SEED_B = 1
SEED_PERM = 42
SEED_TASK = 42

print("=" * 60)
print("K METRIC TOY MODEL")
print("Coherent Information Fraction -- Full Pipeline")
print("=" * 60)

# ============================================================
# STEP 1: CONSTRUCT NOISE COVARIANCE MATRICES
# ============================================================
print("\n[1/6] Constructing noise covariance matrices...")
task_axis = get_task_axis(N_NEURONS)
sigma_a = construct_sigma_A(task_axis, NOISE_VARIANCE, N_NEURONS)
sigma_b = construct_sigma_B(task_axis, NOISE_VARIANCE, N_NEURONS)
verify_matching(sigma_a, sigma_b)

# ============================================================
# STEP 2: GENERATE POPULATION RESPONSES
# ============================================================
print("[2/6] Generating population responses...")
mu_0, mu_1 = get_task_means(task_axis, SIGNAL_STRENGTH, N_NEURONS)
responses_a, labels_a = generate_responses(
    sigma_a, mu_0, mu_1, N_TRIALS, seed=SEED_A
)
responses_b, labels_b = generate_responses(
    sigma_b, mu_0, mu_1, N_TRIALS, seed=SEED_B
)
print(f"  Generated {N_TRIALS} trials per system over {N_NEURONS} neurons.\n")

# ============================================================
# STEP 3: COMPUTE NAVIGABILITY
# ============================================================
print("[3/6] Computing navigability via permutation test...")
print("  (This takes ~60 seconds -- 1000 permutations per system)")
nav_a, thresh_a, acc_a, frac_a, idx_a = compute_navigability(
    responses_a, labels_a, n_permutations=N_PERMUTATIONS, seed=SEED_PERM
)
nav_b, thresh_b, acc_b, frac_b, idx_b = compute_navigability(
    responses_b, labels_b, n_permutations=N_PERMUTATIONS, seed=SEED_PERM
)
verify_navigability(nav_a, nav_b, thresh_a, thresh_b,
                    acc_a, acc_b, frac_a, frac_b)

# ============================================================
# STEP 4: ESTIMATE MUTUAL INFORMATION
# ============================================================
print("[4/6] Estimating mutual information...")
mi_raw_a, mi_nav_a = compute_raw_and_navigable_mi(
    responses_a, labels_a, nav_a, idx_a, task_axis
)
mi_raw_b, mi_nav_b = compute_raw_and_navigable_mi(
    responses_b, labels_b, nav_b, idx_b, task_axis
)
verify_mi(mi_raw_a, mi_nav_a, mi_raw_b, mi_nav_b)

# ============================================================
# STEP 5: COMPUTE K
# ============================================================
print("[5/6] Computing K...")
h_max_a = compute_h_max(sigma_a, task_axis)
h_max_b = compute_h_max(sigma_b, task_axis)
k_a = compute_k(mi_nav_a, h_max_a)
k_b = compute_k(mi_nav_b, h_max_b)
verify_k(k_a, k_b, h_max_a, h_max_b,
         mi_nav_a, mi_nav_b,
         mi_raw_a, mi_raw_b)

# ============================================================
# STEP 6: BEHAVIORAL TASK + FIGURE
# ============================================================
print("[6/6] Simulating behavioral task and generating figure...")
beh_acc_a = simulate_two_step_task(
    responses_a, labels_a, sigma_a, task_axis, seed=SEED_TASK
)
beh_acc_b = simulate_two_step_task(
    responses_b, labels_b, sigma_b, task_axis, seed=SEED_TASK
)
verify_behavioral(beh_acc_a, beh_acc_b)

plot_results(
    mi_raw_a, mi_raw_b,
    mi_nav_a, mi_nav_b,
    k_a, k_b,
    beh_acc_a, beh_acc_b
)

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Signal strength:      {SIGNAL_STRENGTH} (identical)")
print(f"  Total noise volume:   {NOISE_VARIANCE * N_NEURONS:.1f} (identical)")
print(f"  Dimensionality:       {N_NEURONS} neurons (identical)")
print(f"  Noise direction:      DIFFERS (orthogonal vs aligned)")
print()
print(f"  Raw MI     A={mi_raw_a:.3f}  B={mi_raw_b:.3f}")
print(f"  Nav MI     A={mi_nav_a:.3f}  B={mi_nav_b:.3f}")
print(f"  K          A={k_a:.3f}  B={k_b:.3f}  <-- separates cleanly")
print(f"  Accuracy   A={beh_acc_a:.3f}  B={beh_acc_b:.3f}  <-- tracks K")
print()
print(f"  Figure saved to figures/k_metric_results.png")
print("=" * 60)
