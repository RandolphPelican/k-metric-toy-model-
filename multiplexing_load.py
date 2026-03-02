import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# multiplexing_load.py
# Multiplexing Load Monitor using K
# Shows K degrading as concurrent tasks increase -- not from noise
# but from representational crowding on the primary task axis.
# MI stays relatively flat. K detects the geometric threshold.
# Author: Randolph Pelican III (John D. Stabler)
# Repo: github.com/RandolphPelican/k-metric-toy-model-

SEED         = 42
N_NEURONS    = 50
N_TRIALS     = 6000
SIGNAL_SCALE = 2.0
NOISE_SCALE  = 0.3
MAX_TASKS    = 10
N_BINS       = 20


def make_task_axis(n, seed=SEED):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def make_concurrent_axes(primary, n_concurrent, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    axes = []
    for i in range(n_concurrent):
        v = rng.standard_normal(len(primary))
        # Partial orthogonalization -- gets worse as space fills
        for existing in axes:
            v -= existing * (existing @ v) * max(0, 1.0 - i * 0.18)
        v /= np.linalg.norm(v) + 1e-9
        axes.append(v)
    return axes


def make_sigma(primary, concurrent, noise_scale):
    n = len(primary)
    sigma = np.eye(n) * noise_scale
    for i, ax in enumerate(concurrent):
        # Each concurrent task bleeds onto primary axis
        sigma += ax[:, None] * ax[None, :] * SIGNAL_SCALE * (1.5 + i * 0.4)
    ev, evec = np.linalg.eigh(sigma)
    ev = np.clip(ev, 1e-6, None)
    return evec @ np.diag(ev) @ evec.T


def generate_responses(primary, concurrent, sigma, n_trials, seed):
    rng = np.random.default_rng(seed)
    n = len(primary)
    labels = rng.integers(0, 2, size=n_trials)
    means = np.outer(labels * 2 - 1, primary * SIGNAL_SCALE)
    for ax in concurrent:
        cl = rng.integers(0, 2, size=n_trials)
        means += np.outer(cl * 2 - 1, ax * SIGNAL_SCALE * 0.9)
    noise = rng.multivariate_normal(np.zeros(n), sigma, size=n_trials)
    return means + noise, labels


def compute_K(primary, sigma, responses, labels):
    proj = responses @ primary
    m0, m1 = proj[labels==0].mean(), proj[labels==1].mean()
    mid = (m0 + m1) / 2
    sep = abs(m1 - m0)
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
    s2 = SIGNAL_SCALE**2 + float(primary @ sigma @ primary)
    H = 0.5 * np.log(2 * np.pi * np.e * s2)
    return float(np.clip(mi_nav / (H + 1e-9), 0, 1))


def compute_mi(primary, responses, labels):
    proj = responses @ primary
    bins = np.linspace(proj.min(), proj.max(), N_BINS + 1)
    p0 = np.histogram(proj[labels==0], bins=bins)[0] + 1e-9; p0 /= p0.sum()
    p1 = np.histogram(proj[labels==1], bins=bins)[0] + 1e-9; p1 /= p1.sum()
    pm = 0.5*p0 + 0.5*p1
    return max(0.0, 0.5*(np.sum(p0*np.log(p0/pm+1e-9)) +
                          np.sum(p1*np.log(p1/pm+1e-9))))


def compute_interference(primary, concurrent):
    if not concurrent:
        return 0.0
    return float(np.mean([float(primary @ ax)**2 for ax in concurrent]))


def compute_pr(sigma):
    ev = np.linalg.eigvalsh(sigma)
    ev = ev[ev > 1e-9]
    return (ev.sum()**2) / (ev**2).sum()


def main():
    print("=" * 60)
    print("  Multiplexing Load Monitor")
    print("  K-Metric Companion Demo")
    print("=" * 60)
    print(f"\nN_NEURONS={N_NEURONS}  MAX_TASKS={MAX_TASKS}\n")

    primary = make_task_axis(N_NEURONS)

    loads = []
    K_vals = []
    mi_vals = []
    pr_vals = []
    intf_vals = []
    threshold = None

    for n in range(MAX_TASKS):
        concurrent = make_concurrent_axes(primary, n)
        sigma = make_sigma(primary, concurrent, NOISE_SCALE)
        resp, labels = generate_responses(
            primary, concurrent, sigma, N_TRIALS, SEED + n * 77)

        K    = compute_K(primary, sigma, resp, labels)
        mi   = compute_mi(primary, resp, labels)
        pr   = compute_pr(sigma)
        intf = compute_interference(primary, concurrent)

        loads.append(n)
        K_vals.append(K)
        mi_vals.append(mi)
        pr_vals.append(pr)
        intf_vals.append(intf)

        if threshold is None and n > 0 and K < K_vals[0] * 0.6:
            threshold = n

        print(f"  Load={n:2d} | K={K:.3f} | MI={mi:.3f} | "
              f"PR={pr:.1f} | interference={intf:.3f}")

    K0  = K_vals[0];  Kf  = K_vals[-1]
    mi0 = mi_vals[0]; mif = mi_vals[-1]
    Kd  = K0 - Kf;   mid = mi0 - mif

    print("\nSummary")
    print("-------")
    print(f"K at load=0:       {K0:.3f}")
    print(f"K at max load:     {Kf:.3f}")
    print(f"K total drop:      {Kd:.3f}")
    print(f"MI at load=0:      {mi0:.3f}")
    print(f"MI at max load:    {mif:.3f}")
    print(f"MI total drop:     {mid:.3f}")
    print(f"K/MI drop ratio:   {Kd / (mid + 1e-9):.2f}")
    print(f"Threshold:         load={threshold}")
    print()
    print("K drops as concurrent task axes crowd the primary axis.")
    print("MI stays higher because it measures total info not geometry.")
    print("K identifies the threshold. MI cannot.")

    # Plot
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    def sty(ax, title, ylabel="", xlabel="Concurrent tasks"):
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a4e")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        ax.set_title(title, color="#ffffff", fontsize=10,
                     pad=8, fontweight="bold")
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)

    ax1.plot(loads, K_vals,  color="#2ecc71", lw=2.5, marker="o",
             ms=7, label="K (primary task)")
    ax1.plot(loads, mi_vals, color="#3498db", lw=2.0, marker="s",
             ms=5, ls="--", label="Raw MI (state-blind)")
    if threshold:
        ax1.axvline(threshold, color="#e74c3c", lw=1.5, ls="--",
                    label=f"K threshold (load={threshold})")
    ax1.set_xticks(loads)
    ax1.set_ylim(0, 0.8)
    sty(ax1, "A — K Degradation Under Multiplexing Load\n"
        "K detects representational crowding. MI stays higher.",
        ylabel="Metric value")
    ax1.legend(facecolor="#0f0f1a", edgecolor="#2a2a4e",
               labelcolor="white", fontsize=9)

    ax2.plot(loads, intf_vals, color="#e67e22", lw=2.5, marker="o", ms=6)
    if threshold:
        ax2.axvline(threshold, color="#e74c3c", lw=1.5, ls="--")
    sty(ax2, "B — Cross-Task Interference vs Load",
        ylabel="Mean squared axis overlap")

    ax2.set_xticks(loads)

    clrs = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(loads)))
    for i, (k, intf, ld) in enumerate(zip(K_vals, intf_vals, loads)):
        ax3.scatter(intf, k, color=clrs[i], s=80, zorder=5,
                    edgecolors="white", lw=0.6)
        ax3.annotate(f"L{ld}", (intf, k),
                     textcoords="offset points", xytext=(5, 3),
                     color=clrs[i], fontsize=7)
    sty(ax3, "C — K vs Interference\n(green=low load  red=high load)",
        ylabel="K", xlabel="Cross-task interference")

    fig.suptitle(
        "Multiplexing Load Monitor — K Detects Representational Crowding\n"
        "MI is relatively flat. K identifies the geometric threshold.",
        color="#ffffff", fontsize=12, fontweight="bold", y=0.99)

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/multiplexing_load_results.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\n  Saved: figures/multiplexing_load_results.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
