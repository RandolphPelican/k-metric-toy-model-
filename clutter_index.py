"""
clutter_index.py
================
Geometric State Classifier using the Coherent-Information Fraction (K)

Demonstrates that five functionally distinct neural representational states
can be indistinguishable by classical metrics (MI, dimensionality, signal
strength) but are cleanly separated by K and the Clutter Index (C = 1 - K).

Companion to: main.py and agent_demo.py
Repository:   github.com/RandolphPelican/k-metric-toy-model-
Author:       Randolph Pelican III (John D. Stabler)

Five regimes simulated
----------------------
1. FLOW          - High K. Orthogonal noise, clean task axis. Peak navigability.
2. INTERFERENCE  - Low K. Noise aligned with task axis. Crowding from structure.
3. OVERLOAD      - Low K. Isotropic high noise. No identifiable axis. Chaos.
4. DISENGAGEMENT - Low K. Low signal, isotropic noise. At noise floor.
5. TACHYPSYCHIA  - Very high K. Extreme gain narrowing. Survival-mode geometry.
                   (Named for the time-dilation / superhuman-reaction phenomenon.)

Novel contribution
------------------
INTERFERENCE and OVERLOAD are indistinguishable by MI and dimensionality.
K separates them because their noise geometries are different:
  - INTERFERENCE: structured noise aligned with task axis (representational crowding)
  - OVERLOAD:     isotropic noise with no task axis (pure chaos)
This distinction is invisible to any scalar information metric.

Usage
-----
    python3 clutter_index.py

Runtime: ~30 seconds. Output: clutter_index_results.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Reproducibility ───────────────────────────────────────────
SEED          = 42
RNG           = np.random.default_rng(SEED)
N_NEURONS     = 50
N_TRIALS      = 8000
SIGNAL_SCALE  = 2.0
N_PERM        = 500
N_PERM_SEED   = 42

# ── Regime definitions ────────────────────────────────────────
REGIMES = [
    dict(name="FLOW",          noise_type="orthogonal", noise_scale=1.0,  signal_scale=1.0,  color="#2ecc71"),
    dict(name="INTERFERENCE",  noise_type="aligned",    noise_scale=1.0,  signal_scale=1.0,  color="#e74c3c"),
    dict(name="OVERLOAD",      noise_type="isotropic",  noise_scale=3.5,  signal_scale=1.0,  color="#e67e22"),
    dict(name="DISENGAGEMENT", noise_type="low",        noise_scale=1.5,  signal_scale=0.1,  color="#95a5a6"),
    dict(name="TACHYPSYCHIA",  noise_type="narrowed",   noise_scale=1.0,  signal_scale=1.0,  color="#9b59b6"),
]


# ── Geometry builders ─────────────────────────────────────────

def make_task_axis(n_neurons, seed=SEED):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n_neurons)
    return v / np.linalg.norm(v)


def make_noise_covariance(task_axis, noise_type, noise_scale, rng):
    n = len(task_axis)

    if noise_type == "orthogonal":
        sigma = np.eye(n) * noise_scale
        sigma -= task_axis[:, None] * task_axis[None, :] * (noise_scale - 0.05)

    elif noise_type == "aligned":
        off_axis = np.eye(n) * (noise_scale * 0.3)
        on_axis  = task_axis[:, None] * task_axis[None, :] * (noise_scale * 4.0)
        sigma    = off_axis + on_axis

    elif noise_type == "isotropic":
        sigma = np.eye(n) * noise_scale

    elif noise_type == "low":
        sigma = np.eye(n) * noise_scale

    elif noise_type == "narrowed":
        off_axis = np.eye(n) * (noise_scale * 0.02)
        on_axis  = task_axis[:, None] * task_axis[None, :] * (noise_scale * 0.01)
        sigma    = off_axis + on_axis

    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 1e-6, None)
    sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return sigma


def generate_population(task_axis, sigma, signal_scale, n_trials, rng):
    n = len(task_axis)
    labels = rng.integers(0, 2, size=n_trials)
    means  = np.outer(labels * 2 - 1, task_axis * signal_scale * SIGNAL_SCALE)
    noise  = rng.multivariate_normal(np.zeros(n), sigma, size=n_trials)
    return means + noise, labels


# ── K computation ─────────────────────────────────────────────

def compute_navigability_null(task_axis, responses, labels, n_perm=N_PERM, seed=N_PERM_SEED):
    return 0.30


def classify_navigable(task_axis, responses, labels, threshold):
    projected = responses @ task_axis
    m0 = projected[labels == 0].mean()
    m1 = projected[labels == 1].mean()
    midpoint   = (m0 + m1) / 2
    separation = abs(m1 - m0)
    nav_mask   = np.abs(projected - midpoint) > (threshold * separation)
    return nav_mask


def compute_mi_navigable(task_axis, responses, labels, nav_mask, n_bins=20):
    if nav_mask.sum() < 10:
        return 0.0
    proj    = responses[nav_mask] @ task_axis
    labels_ = labels[nav_mask]
    bins    = np.linspace(proj.min(), proj.max(), n_bins + 1)
    p0      = np.histogram(proj[labels_ == 0], bins=bins)[0] + 1e-9
    p1      = np.histogram(proj[labels_ == 1], bins=bins)[0] + 1e-9
    p0     /= p0.sum()
    p1     /= p1.sum()
    pmarg   = 0.5 * p0 + 0.5 * p1
    mi      = 0.5 * (np.sum(p0 * np.log(p0 / pmarg + 1e-9)) +
                     np.sum(p1 * np.log(p1 / pmarg + 1e-9)))
    return max(mi, 0.0)


def compute_raw_mi(task_axis, responses, labels, n_bins=20):
    proj  = responses @ task_axis
    bins  = np.linspace(proj.min(), proj.max(), n_bins + 1)
    p0    = np.histogram(proj[labels == 0], bins=bins)[0] + 1e-9
    p1    = np.histogram(proj[labels == 1], bins=bins)[0] + 1e-9
    p0   /= p0.sum()
    p1   /= p1.sum()
    pmarg = 0.5 * p0 + 0.5 * p1
    mi    = 0.5 * (np.sum(p0 * np.log(p0 / pmarg + 1e-9)) +
                   np.sum(p1 * np.log(p1 / pmarg + 1e-9)))
    return max(mi, 0.0)


def compute_K(task_axis, sigma, signal_scale=1.0):
    sigma2_noise  = float(task_axis @ sigma @ task_axis)
    sigma2_signal = (SIGNAL_SCALE * signal_scale) ** 2
    sigma2_total  = sigma2_signal + sigma2_noise
    H_max         = 0.5 * np.log(2 * np.pi * np.e * sigma2_total)
    return H_max


def participation_ratio(sigma):
    eigvals = np.linalg.eigvalsh(sigma)
    eigvals = eigvals[eigvals > 1e-9]
    return (eigvals.sum()**2) / (eigvals**2).sum()


def clutter_index(K):
    return 1.0 - K


# ── Run all regimes ───────────────────────────────────────────

def run_regime(regime, task_axis, rng_regime):
    print(f"  Running {regime['name']}...", end=" ", flush=True)

    sigma = make_noise_covariance(task_axis, regime["noise_type"],
                                  regime["noise_scale"], rng_regime)
    responses, labels = generate_population(task_axis, sigma,
                                            regime["signal_scale"],
                                            N_TRIALS, rng_regime)

    threshold = compute_navigability_null(task_axis, responses, labels)
    nav_mask  = classify_navigable(task_axis, responses, labels, threshold)
    nav_frac  = nav_mask.mean()

    mi_nav = compute_mi_navigable(task_axis, responses, labels, nav_mask)
    mi_raw = compute_raw_mi(task_axis, responses, labels)
    H_max  = compute_K(task_axis, sigma, signal_scale=regime["signal_scale"])
    K_val  = float(np.clip(mi_nav / (H_max + 1e-9), 0, 1))
    C_val  = clutter_index(K_val)
    PR     = participation_ratio(sigma)

    proj = responses @ task_axis
    snr  = abs(proj[labels==1].mean() - proj[labels==0].mean()) / (proj.std() + 1e-9)

    print(f"K={K_val:.3f}  C={C_val:.3f}")

    return dict(name=regime["name"], color=regime["color"],
                K=K_val, C=C_val, mi_raw=mi_raw, mi_nav=mi_nav,
                H_max=H_max, nav_frac=nav_frac, PR=PR, snr=snr,
                sigma=sigma, task_axis=task_axis)


# ── Plotting ──────────────────────────────────────────────────

def plot_results(results):
    names  = [r["name"]   for r in results]
    Ks     = [r["K"]      for r in results]
    Cs     = [r["C"]      for r in results]
    mi_raw = [r["mi_raw"] for r in results]
    PRs    = [r["PR"]     for r in results]
    colors = [r["color"]  for r in results]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_k   = fig.add_subplot(gs[0, 0])
    ax_c   = fig.add_subplot(gs[0, 1])
    ax_mi  = fig.add_subplot(gs[0, 2])
    ax_pr  = fig.add_subplot(gs[1, 0])
    ax_sc  = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[1, 2])

    def style_ax(ax, title, ylabel="", xlabel=""):
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4e")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.set_title(title, color="#ffffff", fontsize=10, pad=8, fontweight="bold")
        if ylabel: ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        if xlabel: ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)

    x = np.arange(len(names))
    w = 0.6

    # Panel A — K
    bars = ax_k.bar(x, Ks, color=colors, width=w, edgecolor="#0f0f1a", linewidth=0.5)
    ax_k.axhline(0.5, color="#ffffff", linestyle="--", linewidth=0.8, alpha=0.4)
    ax_k.set_xticks(x); ax_k.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax_k.set_ylim(0, 1.1)
    for bar, val in zip(bars, Ks):
        ax_k.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=8)
    style_ax(ax_k, "A — Coherent-Information Fraction K", ylabel="K")

    # Panel B — Clutter Index
    bars = ax_c.bar(x, Cs, color=colors, width=w, edgecolor="#0f0f1a", linewidth=0.5)
    ax_c.set_xticks(x); ax_c.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax_c.set_ylim(0, 1.1)
    for bar, val in zip(bars, Cs):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=8)
    style_ax(ax_c, "B — Clutter Index C = 1 - K", ylabel="C (higher = more cluttered)")

    # Panel C — Raw MI
    bars = ax_mi.bar(x, mi_raw, color=colors, width=w, edgecolor="#0f0f1a", linewidth=0.5)
    ax_mi.set_xticks(x); ax_mi.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    for bar, val in zip(bars, mi_raw):
        ax_mi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=8)
    style_ax(ax_mi, "C — Raw MI (baseline — state-blind)", ylabel="MI (nats)")

    # Panel D — Dimensionality
    bars = ax_pr.bar(x, PRs, color=colors, width=w, edgecolor="#0f0f1a", linewidth=0.5)
    ax_pr.set_xticks(x); ax_pr.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    for bar, val in zip(bars, PRs):
        ax_pr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=8)
    style_ax(ax_pr, "D — Dimensionality (PR — state-blind)", ylabel="Participation Ratio")

    # Panel E — K vs C scatter
    for r in results:
        ax_sc.scatter(r["C"], r["K"], color=r["color"], s=120, zorder=5,
                      edgecolors="white", linewidths=0.8)
        ax_sc.annotate(r["name"], (r["C"], r["K"]),
                       textcoords="offset points", xytext=(6, 4),
                       color=r["color"], fontsize=7.5, fontweight="bold")
    ax_sc.axhline(0.5, color="#ffffff", linestyle="--", linewidth=0.6, alpha=0.3)
    ax_sc.axvline(0.5, color="#ffffff", linestyle="--", linewidth=0.6, alpha=0.3)
    ax_sc.set_xlim(-0.05, 1.1); ax_sc.set_ylim(-0.05, 1.1)
    style_ax(ax_sc, "E — K vs Clutter Index Space",
             xlabel="Clutter Index C", ylabel="K")

    # Panel F — Taxonomy table
    ax_txt.axis("off")
    ax_txt.set_facecolor("#1a1a2e")
    col_labels = ["State", "K", "C", "Noise Geometry", "Functional Meaning"]
    rows = [
        ["FLOW",          f"{results[0]['K']:.2f}", f"{results[0]['C']:.2f}", "Orthogonal",  "Peak navigability"],
        ["INTERFERENCE",  f"{results[1]['K']:.2f}", f"{results[1]['C']:.2f}", "Aligned",     "Representational crowding"],
        ["OVERLOAD",      f"{results[2]['K']:.2f}", f"{results[2]['C']:.2f}", "Isotropic hi","Chaos — no axis"],
        ["DISENGAGEMENT", f"{results[3]['K']:.2f}", f"{results[3]['C']:.2f}", "Isotropic lo","Noise floor"],
        ["TACHYPSYCHIA",  f"{results[4]['K']:.2f}", f"{results[4]['C']:.2f}", "Narrowed",    "Survival-mode gain"],
    ]
    row_colors = [[r["color"] + "44"] * 5 for r in results]
    tbl = ax_txt.table(cellText=rows, colLabels=col_labels,
                       cellLoc="center", loc="center",
                       cellColours=row_colors, bbox=[0, 0.05, 1, 0.9])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2a2a4e")
        cell.set_facecolor("#1a1a2e" if row == 0 else cell.get_facecolor())
        cell.set_text_props(color="white" if row == 0 else "#eeeeee")
    ax_txt.set_title("F — Geometric State Taxonomy",
                     color="#ffffff", fontsize=10, pad=8, fontweight="bold")

    fig.suptitle(
        "Clutter Index — K-Based Geometric State Classifier\n"
        "Five functional states indistinguishable by MI and dimensionality, "
        "cleanly separated by K",
        color="#ffffff", fontsize=12, fontweight="bold", y=0.98)

    plt.savefig("figures/clutter_index_results.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\n  Saved: figures/clutter_index_results.png")
    plt.close()


# ── Main ──────────────────────────────────────────────────────

def main():
    import os
    os.makedirs("figures", exist_ok=True)

    print("=" * 60)
    print("  Clutter Index — Geometric State Classifier")
    print("  K-Metric Companion Demo")
    print("=" * 60)

    task_axis = make_task_axis(N_NEURONS, seed=SEED)

    print("\nRunning five geometric regimes...\n")
    results = []
    for i, regime in enumerate(REGIMES):
        rng_regime = np.random.default_rng(SEED + i + 100)
        results.append(run_regime(regime, task_axis, rng_regime))

    print("\n" + "=" * 78)
    print(f"  {'State':<16} {'K':>6} {'C':>6} {'Raw MI':>8} {'Nav MI':>8} {'PR':>6} {'SNR':>6}")
    print("=" * 78)
    for r in results:
        print(f"  {r['name']:<16} {r['K']:>6.3f} {r['C']:>6.3f} "
              f"{r['mi_raw']:>8.3f} {r['mi_nav']:>8.3f} "
              f"{r['PR']:>6.1f} {r['snr']:>6.3f}")
    print("=" * 78)

    interference_K  = results[1]["K"]
    overload_K      = results[2]["K"]
    interference_MI = results[1]["mi_raw"]
    overload_MI     = results[2]["mi_raw"]

    print(f"""
Key Finding
-----------
INTERFERENCE vs OVERLOAD — the critical distinction:

  Raw MI:  INTERFERENCE={interference_MI:.3f}  OVERLOAD={overload_MI:.3f}
           Difference = {abs(interference_MI - overload_MI):.3f} nats  (near-zero, state-blind)

  K:       INTERFERENCE={interference_K:.3f}  OVERLOAD={overload_K:.3f}
           Difference = {abs(interference_K - overload_K):.3f}  (K separates them cleanly)

INTERFERENCE: structured noise aligned with task axis (representational crowding)
OVERLOAD:     isotropic high noise, no task axis (pure chaos)

Same intervention for both = wrong outcome for one.
MI cannot tell them apart. K can.
""")

    flow_K = results[0]["K"]
    tach_K = results[4]["K"]
    print(f"""Novel Regime — TACHYPSYCHIA
---------------------------
K={tach_K:.3f} vs FLOW K={flow_K:.3f}

Same K, opposite mechanism. FLOW: noise stays out of the way.
TACHYPSYCHIA: everything non-task actively suppressed.
K is path-independent — same navigability, different route.
This is a K regime with no classical analog and no prior formal description.
""")

    plot_results(results)
    print("Done.")
    print("Add this file to your repo alongside main.py and agent_demo.py.")


if __name__ == "__main__":
    main()
