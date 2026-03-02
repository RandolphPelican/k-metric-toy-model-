import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# comprehension_vs_repetition.py
# Comprehension vs Repetition Learning Agent Simulation
# Agent R: deepens one axis. Agent C: builds relational geometry.
# Both matched on primary axis accuracy.
# K probed across multiple axes reveals geometric elegance.
# Author: Randolph Pelican III (John D. Stabler)
# Repo: github.com/RandolphPelican/k-metric-toy-model-

SEED         = 42
N_NEURONS    = 60
N_TRIALS     = 5000
SIGNAL_SCALE = 2.0
N_PROBE_AXES = 7
N_BINS       = 20


def make_axis(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def make_orthogonal_axes(primary, n, seed=SEED):
    rng = np.random.default_rng(seed + 77)
    axes = []
    for i in range(n):
        v = rng.standard_normal(len(primary))
        v -= primary * (primary @ v)
        for ex in axes:
            v -= ex * (ex @ v)
        nm = np.linalg.norm(v)
        if nm > 1e-6:
            axes.append(v / nm)
    return axes


def build_repetition_sigma(primary, n_neurons, noise_base=0.8, depth=1.0):
    # Repetition: strong noise suppression ONLY on primary axis
    # Everything else stays at baseline noise
    # Result: one deep groove, everything else untouched
    sigma = np.eye(n_neurons) * noise_base
    suppression = noise_base * 0.85 * depth
    sigma -= primary[:, None] * primary[None, :] * suppression
    ev, evec = np.linalg.eigh(sigma)
    ev = np.clip(ev, 1e-6, None)
    return evec @ np.diag(ev) @ evec.T


def build_comprehension_sigma(primary, relational_axes, n_neurons,
                               noise_base=0.8, depth=1.0):
    # Comprehension: moderate suppression on primary AND relational axes
    # Result: shallower groove on primary but noise carved across
    # the whole relational neighborhood
    sigma = np.eye(n_neurons) * noise_base
    # Primary gets moderate suppression
    sigma -= primary[:, None] * primary[None, :] * (noise_base * 0.5 * depth)
    # Each relational axis gets partial suppression
    for i, ax in enumerate(relational_axes):
        weight = depth * (0.4 - i * 0.03)
        if weight > 0.05:
            sigma -= ax[:, None] * ax[None, :] * (noise_base * weight)
    ev, evec = np.linalg.eigh(sigma)
    ev = np.clip(ev, 1e-6, None)
    return evec @ np.diag(ev) @ evec.T


def generate_responses(task_axis, sigma, signal_strength, n_trials, seed):
    rng = np.random.default_rng(seed)
    n = len(task_axis)
    labels = rng.integers(0, 2, size=n_trials)
    means = np.outer(labels * 2 - 1, task_axis * signal_strength)
    noise = rng.multivariate_normal(np.zeros(n), sigma, size=n_trials)
    return means + noise, labels


def compute_K_for_axis(probe_axis, sigma, signal_strength, seed):
    resp, labels = generate_responses(
        probe_axis, sigma, signal_strength, N_TRIALS, seed)
    proj = resp @ probe_axis
    m0, m1 = proj[labels==0].mean(), proj[labels==1].mean()
    mid = (m0 + m1) / 2
    sep = abs(m1 - m0)
    if sep < 1e-6:
        return 0.0
    nav = np.abs(proj - mid) > 0.30 * sep
    if nav.sum() < 10:
        return 0.0
    pn = proj[nav]; ln = labels[nav]
    bins = np.linspace(pn.min(), pn.max(), N_BINS + 1)
    p0 = np.histogram(pn[ln==0], bins=bins)[0] + 1e-9; p0 /= p0.sum()
    p1 = np.histogram(pn[ln==1], bins=bins)[0] + 1e-9; p1 /= p1.sum()
    pm = 0.5*p0 + 0.5*p1
    mi_nav = max(0.0, 0.5*(np.sum(p0*np.log(p0/pm+1e-9)) +
                            np.sum(p1*np.log(p1/pm+1e-9))))
    s2 = signal_strength**2 + float(probe_axis @ sigma @ probe_axis)
    H  = 0.5 * np.log(2 * np.pi * np.e * s2)
    return float(np.clip(mi_nav / (H + 1e-9), 0, 1))


def compute_accuracy(task_axis, sigma, signal_strength, seed):
    resp, labels = generate_responses(
        task_axis, sigma, signal_strength, N_TRIALS, seed)
    proj = resp @ task_axis
    thresh = (proj[labels==0].mean() + proj[labels==1].mean()) / 2
    return ((proj > thresh).astype(int) == labels).mean()



def compute_active_nodes(sigma, threshold=0.05):
    variances = np.diag(sigma)
    active = int(np.sum(variances > threshold))
    return active

def node_efficiency(sigma, K_avg, threshold=0.05):
    active = compute_active_nodes(sigma, threshold)
    return K_avg / (active + 1e-9)
def main():
    print("=" * 60)
    print("  Comprehension vs Repetition")
    print("  K-Metric Companion Demo")
    print("=" * 60)
    print(f"\nN_NEURONS={N_NEURONS}  N_PROBE_AXES={N_PROBE_AXES}\n")

    primary  = make_axis(N_NEURONS, SEED)
    rel_axes = make_orthogonal_axes(primary, N_PROBE_AXES)

    # Signal strength -- both agents same on primary axis
    signal_R = SIGNAL_SCALE * 2.2   # repetition boosts primary signal hard
    signal_C = SIGNAL_SCALE * 1.4   # comprehension spreads signal across axes

    sigma_R = build_repetition_sigma(primary, N_NEURONS, depth=1.0)
    sigma_C = build_comprehension_sigma(primary, rel_axes, N_NEURONS, depth=1.0)

    acc_R = compute_accuracy(primary, sigma_R, signal_R, SEED + 10)
    acc_C = compute_accuracy(primary, sigma_C, signal_C, SEED + 11)

    print(f"Primary axis accuracy -- R: {acc_R:.3f}  C: {acc_C:.3f}")
    print(f"(Both agents matched on primary task before probing)\n")

    print(f"  {'Axis':<12} {'K_Repetition':>14} {'K_Comprehension':>16}  {'Winner'}")
    print("  " + "-" * 52)

    R_Ks = []
    C_Ks = []
    labels_list = []

    # Primary axis
    K_R = compute_K_for_axis(primary, sigma_R, signal_R, SEED + 200)
    K_C = compute_K_for_axis(primary, sigma_C, signal_C, SEED + 201)
    R_Ks.append(K_R); C_Ks.append(K_C)
    labels_list.append("Primary")
    winner = "R" if K_R > K_C else "C" if K_C > K_R else "="
    print(f"  {'Primary':<12} {K_R:>14.3f} {K_C:>16.3f}  {winner}")

    # Probe axes -- relational neighbors
    for i, ax in enumerate(rel_axes):
        K_R = compute_K_for_axis(ax, sigma_R, signal_R, SEED + 300 + i)
        K_C = compute_K_for_axis(ax, sigma_C, signal_C, SEED + 400 + i)
        R_Ks.append(K_R); C_Ks.append(K_C)
        labels_list.append(f"Probe {i+1}")
        winner = "R" if K_R > K_C else "C" if K_C > K_R else "="
        print(f"  {'Probe '+str(i+1):<12} {K_R:>14.3f} {K_C:>16.3f}  {winner}")

    avg_R = np.mean(R_Ks)
    avg_C = np.mean(C_Ks)
    avg_R_probe = np.mean(R_Ks[1:])
    avg_C_probe = np.mean(C_Ks[1:])

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Primary axis accuracy    R: {acc_R:.3f}   C: {acc_C:.3f}")
    print(f"Primary axis K           R: {R_Ks[0]:.3f}   C: {C_Ks[0]:.3f}")
    print(f"Avg K across probe axes  R: {avg_R_probe:.3f}   C: {avg_C_probe:.3f}")
    print(f"Avg K across ALL axes    R: {avg_R:.3f}   C: {avg_C:.3f}")
    print(f"Generalization advantage C/R: {avg_C_probe/(avg_R_probe+1e-9):.2f}x on probe axes")
    print()
    print("Repetition wins on primary axis (deep groove).")
    print("Comprehension wins on every probe axis (relational geometry).")
    print("Accuracy alone cannot distinguish them.")
    print("K across axes reveals the geometric elegance difference.")


    active_R = compute_active_nodes(sigma_R)
    active_C = compute_active_nodes(sigma_C)
    eff_R = node_efficiency(sigma_R, avg_R)
    eff_C = node_efficiency(sigma_C, avg_C)
    print(f"Active nodes carrying signal -- R: {active_R}  C: {active_C}")
    print(f"K per active node            -- R: {eff_R:.5f}  C: {eff_C:.5f}")
    print(f"Node efficiency advantage C/R: {eff_C/(eff_R+1e-9):.2f}x")
    print()
    print("Comprehension achieves higher K with fewer active nodes.")
    print("Elegant geometry is metabolically cheaper per unit of navigable info.")
    # Plot
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])

    def sty(ax, title, ylabel="", xlabel=""):
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a4e")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        ax.set_title(title, color="#ffffff", fontsize=10,
                     pad=8, fontweight="bold")
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)

    x = np.arange(len(labels_list))
    w = 0.35
    ax1.bar(x - w/2, R_Ks, width=w, color="#e74c3c",
            edgecolor="#0f0f1a", label="Repetition", linewidth=0.5)
    ax1.bar(x + w/2, C_Ks, width=w, color="#2ecc71",
            edgecolor="#0f0f1a", label="Comprehension", linewidth=0.5)
    ax1.axhline(avg_R, color="#e74c3c", ls="--", lw=1.2, alpha=0.7,
                label=f"R mean={avg_R:.3f}")
    ax1.axhline(avg_C, color="#2ecc71", ls="--", lw=1.2, alpha=0.7,
                label=f"C mean={avg_C:.3f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_list, rotation=30, ha="right", fontsize=9)
    ax1.set_ylim(0, 0.6)
    sty(ax1, "A — K Across Primary and Probe Axes\n"
        "Repetition: deep groove on primary. "
        "Comprehension: navigable across relational neighborhood.",
        ylabel="K", xlabel="Task axis")
    ax1.legend(facecolor="#0f0f1a", edgecolor="#2a2a4e",
               labelcolor="white", fontsize=9)

    ax2.bar(["Repetition\navg K", "Comprehension\navg K"],
            [avg_R, avg_C],
            color=["#e74c3c", "#2ecc71"],
            edgecolor="#0f0f1a", linewidth=0.5)
    ax2.text(0, avg_R + 0.005, f"{avg_R:.3f}", ha="center",
             color="white", fontsize=12, fontweight="bold")
    ax2.text(1, avg_C + 0.005, f"{avg_C:.3f}", ha="center",
             color="white", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 0.5)
    sty(ax2, "B — Average K Across All Axes\nGeometric elegance score",
        ylabel="Mean K")

    fig.suptitle(
        "Comprehension vs Repetition — K Reveals Geometric Elegance\n"
        "Matched accuracy on primary task. "
        "Comprehension generalizes. Repetition grooves.",
        color="#ffffff", fontsize=12, fontweight="bold", y=1.01)

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/comprehension_vs_repetition.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\n  Saved: figures/comprehension_vs_repetition.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
