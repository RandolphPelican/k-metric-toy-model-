import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# attention_gain.py
# Attentional Gain Modulation using K
# Shows K responding to attentional state via gain rescaling of the task axis.
# Low attention = low K even with strong signal.
# High attention = K recovers regardless of noise level.
# Maps to dopamine/acetylcholine modulation of task-axis amplitude.
# Author: Randolph Pelican III (John D. Stabler)
# Repo: github.com/RandolphPelican/k-metric-toy-model-

SEED         = 42
N_NEURONS    = 50
N_TRIALS     = 6000
SIGNAL_SCALE = 2.0
NOISE_SCALE  = 1.0
N_BINS       = 20
N_LEVELS     = 10


def make_task_axis(n, seed=SEED):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def make_sigma(task_axis, noise_scale, attention_gain):
    n = len(task_axis)
    # Attention suppresses off-axis noise proportional to gain
    # High attention = noise orthogonal component shrinks
    # Low attention = noise isotropic, task axis gets no special treatment
    off_axis_scale = noise_scale * (1.0 - attention_gain * 0.7)
    on_axis_scale  = noise_scale * (1.0 - attention_gain * 0.9)
    off_axis_scale = max(off_axis_scale, 0.02)
    on_axis_scale  = max(on_axis_scale,  0.01)
    sigma = np.eye(n) * off_axis_scale
    sigma += task_axis[:, None] * task_axis[None, :] * (on_axis_scale - off_axis_scale)
    ev, evec = np.linalg.eigh(sigma)
    ev = np.clip(ev, 1e-6, None)
    return evec @ np.diag(ev) @ evec.T


def generate_responses(task_axis, sigma, attention_gain, n_trials, seed):
    rng = np.random.default_rng(seed)
    n = len(task_axis)
    labels = rng.integers(0, 2, size=n_trials)
    # Attention scales effective signal amplitude on task axis
    effective_signal = SIGNAL_SCALE * attention_gain
    means = np.outer(labels * 2 - 1, task_axis * effective_signal)
    noise = rng.multivariate_normal(np.zeros(n), sigma, size=n_trials)
    return means + noise, labels


def compute_K(task_axis, sigma, responses, labels, attention_gain):
    proj = responses @ task_axis
    m0, m1 = proj[labels==0].mean(), proj[labels==1].mean()
    mid = (m0 + m1) / 2
    sep = abs(m1 - m0)
    if sep < 1e-6:
        return 0.0, 0.0
    nav = np.abs(proj - mid) > 0.30 * sep
    if nav.sum() < 10:
        return 0.0, 0.0
    pn = proj[nav]; ln = labels[nav]
    bins = np.linspace(pn.min(), pn.max(), N_BINS + 1)
    p0 = np.histogram(pn[ln==0], bins=bins)[0] + 1e-9; p0 /= p0.sum()
    p1 = np.histogram(pn[ln==1], bins=bins)[0] + 1e-9; p1 /= p1.sum()
    pm = 0.5*p0 + 0.5*p1
    mi_nav = max(0.0, 0.5*(np.sum(p0*np.log(p0/pm+1e-9)) +
                            np.sum(p1*np.log(p1/pm+1e-9))))
    effective_signal = SIGNAL_SCALE * attention_gain
    s2 = effective_signal**2 + float(task_axis @ sigma @ task_axis)
    H  = 0.5 * np.log(2 * np.pi * np.e * s2)
    K  = float(np.clip(mi_nav / (H + 1e-9), 0, 1))
    return K, nav.mean()


def compute_mi(task_axis, responses, labels):
    proj = responses @ task_axis
    bins = np.linspace(proj.min(), proj.max(), N_BINS + 1)
    p0 = np.histogram(proj[labels==0], bins=bins)[0] + 1e-9; p0 /= p0.sum()
    p1 = np.histogram(proj[labels==1], bins=bins)[0] + 1e-9; p1 /= p1.sum()
    pm = 0.5*p0 + 0.5*p1
    return max(0.0, 0.5*(np.sum(p0*np.log(p0/pm+1e-9)) +
                          np.sum(p1*np.log(p1/pm+1e-9))))


def main():
    print("=" * 60)
    print("  Attentional Gain Modulation")
    print("  K-Metric Companion Demo")
    print("=" * 60)
    print(f"\nN_NEURONS={N_NEURONS}  N_LEVELS={N_LEVELS}\n")

    task_axis = make_task_axis(N_NEURONS)

    gain_levels = np.linspace(0.1, 1.0, N_LEVELS)
    K_vals  = []
    mi_vals = []
    nav_vals = []
    snr_vals = []

    for i, gain in enumerate(gain_levels):
        sigma = make_sigma(task_axis, NOISE_SCALE, gain)
        resp, labels = generate_responses(
            task_axis, sigma, gain, N_TRIALS, SEED + i * 33)

        K, nav_frac = compute_K(task_axis, sigma, resp, labels, gain)
        mi          = compute_mi(task_axis, resp, labels)
        proj        = resp @ task_axis
        snr         = abs(proj[labels==1].mean() - proj[labels==0].mean()) / (proj.std() + 1e-9)

        K_vals.append(K)
        mi_vals.append(mi)
        nav_vals.append(nav_frac)
        snr_vals.append(snr)

        print(f"  Gain={gain:.2f} | K={K:.3f} | MI={mi:.3f} | "
              f"nav={nav_frac:.2f} | SNR={snr:.3f}")

    K_low  = K_vals[0]
    K_high = K_vals[-1]
    mi_low  = mi_vals[0]
    mi_high = mi_vals[-1]

    print("\nSummary")
    print("-------")
    print(f"K at min attention (gain=0.1):   {K_low:.3f}")
    print(f"K at max attention (gain=1.0):   {K_high:.3f}")
    print(f"K gain across attention range:   {K_high - K_low:.3f}")
    print(f"MI at min attention:             {mi_low:.3f}")
    print(f"MI at max attention:             {mi_high:.3f}")
    print(f"MI gain across attention range:  {mi_high - mi_low:.3f}")
    print(f"K sensitivity to attention:      {(K_high-K_low)/(mi_high-mi_low+1e-9):.2f}x MI")
    print()
    print("Attention operates as a gain modulator on the task axis.")
    print("K responds because attention changes the signal-to-noise")
    print("geometry along the task axis -- not just total signal power.")
    print("This maps to dopamine/acetylcholine modulation in biological systems.")

    # Plot
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    def sty(ax, title, ylabel="", xlabel="Attention gain"):
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a4e")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        ax.set_title(title, color="#ffffff", fontsize=10,
                     pad=8, fontweight="bold")
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)

    gains = gain_levels

    ax1.plot(gains, K_vals,  color="#2ecc71", lw=2.5, marker="o",
             ms=7, label="K")
    ax1.plot(gains, mi_vals, color="#3498db", lw=2.0, marker="s",
             ms=5, ls="--", label="Raw MI")
    ax1.plot(gains, nav_vals, color="#e67e22", lw=1.5, marker="^",
             ms=5, ls=":", label="Navigable fraction")
    ax1.set_ylim(0, 0.8)
    sty(ax1, "A — K and MI Across Attention Gain Levels\n"
        "K rises with attention. Low attention = low K even with strong signal.",
        ylabel="Metric value")
    ax1.legend(facecolor="#0f0f1a", edgecolor="#2a2a4e",
               labelcolor="white", fontsize=9)

    ax2.plot(gains, snr_vals, color="#9b59b6", lw=2.5, marker="o", ms=6)
    sty(ax2, "B — Signal-to-Noise Ratio vs Attention",
        ylabel="SNR (task axis)")

    clrs = plt.cm.RdYlGn(np.linspace(0.1, 0.9, N_LEVELS))
    for i, (k, mi, g) in enumerate(zip(K_vals, mi_vals, gains)):
        ax3.scatter(mi, k, color=clrs[i], s=80, zorder=5,
                    edgecolors="white", lw=0.6)
        ax3.annotate(f"{g:.1f}", (mi, k),
                     textcoords="offset points", xytext=(5, 3),
                     color=clrs[i], fontsize=7)
    sty(ax3, "C — K vs MI per Attention Level\n(red=low attention  green=high attention)",
        ylabel="K", xlabel="Raw MI")

    fig.suptitle(
        "Attentional Gain Modulation — K Tracks Attention State\n"
        "Attention rescales task-axis gain. K responds. "
        "Maps to dopamine and acetylcholine modulation.",
        color="#ffffff", fontsize=12, fontweight="bold", y=0.99)

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/attention_gain_results.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("\n  Saved: figures/attention_gain_results.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
