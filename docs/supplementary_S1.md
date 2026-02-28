# Supplementary Material S1
## K-Grounded Cognitive Agent: Applied Demonstration

### Overview

The core toy model (main.py) establishes K as a superior metric by comparing
two matched neural populations post-hoc. This supplementary demonstration
extends the claim to an applied agent architecture: K operating as a
real-time decision gate during multi-step reasoning. The central question
is whether a K-grounded agent makes qualitatively different and measurably
better decisions than an MI-only baseline — not merely scores differently,
but behaves differently under conditions where noise geometry is
task-destructive.

---

### Agent Architectures

**K-Agent.** At each reasoning step, the agent encodes a representational
vector in DIM=6 dimensional task space. It maintains an empirical noise
covariance matrix Σ computed over accumulated representations, and holds
a fixed task-axis vector determined by the task structure. K is computed
at each step as:

    K = MI_nav / H_max
    H_max = 0.5 * log(2 * pi * e * sigma^2_task)
    sigma^2_task = task_axis^T @ Sigma @ task_axis

Navigability is assessed against a permutation null distribution
(N=500 permutations, 95th percentile threshold), consistent with the
criterion established in the core toy model. Action selection is gated
on K: PROCEED (K > 0.4), REORIENT (0.2 < K <= 0.4, with corrective
re-alignment), or HALT (K <= 0.2). The agent cannot proceed through a
step it has flagged as non-navigable without first attempting reorientation.

**MI-Agent (baseline).** A scalar mutual information proxy is computed at
each step from signal magnitude and additive noise. Noise direction is
not represented; the agent has no access to the noise covariance matrix
and no concept of task-axis alignment. Action selection gates only on
whether raw MI exceeds a fixed threshold. This agent is structurally
identical to any system that monitors information quantity without
information geometry.

---

### Tasks

Three reasoning tasks were evaluated, each with a distinct task-axis
structure and signal decay profile:

- **Nested Recursion** (4 steps, decay 0.15/step): Signal attenuates
  with embedding depth, modeling the performance degradation observed
  in center-embedded sentence processing.

- **Multi-Hop Inference** (4 steps, decay 0.10/step): Causal chain
  where each step depends on prior steps, modeling multi-hop reasoning
  benchmarks.

- **Contradiction Resolution** (4 steps, decay 0.08/step): Belief
  update under conflicting evidence, where the task-axis structure is
  more distributed across dimensions.

---

### Noise Conditions

**Orthogonal noise (System A):** Noise vectors are constrained to be
orthogonal to the task axis via Gram-Schmidt projection. sigma^2_task
remains low; H_max is small; K stays high. This is the favorable
condition.

**Aligned noise (System B):** Noise vectors are drawn along the task
axis with random sign, directly inflating sigma^2_task and H_max. K
degrades because the denominator grows while MI_nav does not compensate.
The MI-agent cannot detect this distinction; its scalar proxy is
insensitive to noise direction.

---

### Results

Results are reported across N=50 independent runs per condition.

**K discrimination vs MI discrimination (orthogonal minus aligned score):**

| Task                   | K delta | MI delta | K advantage |
|------------------------|---------|----------|-------------|
| Nested Recursion       | +0.147  | +0.023   | +0.124      |
| Multi-Hop Inference    | +0.456  | +0.015   | +0.441      |
| Contradiction Resol.   | +0.860  | +0.005   | +0.855      |

K responds strongly and consistently to the change in noise geometry
across all three tasks. Raw MI is effectively flat (delta = 0.005-0.023),
confirming that scalar mutual information carries no information about
noise direction regardless of task structure.

**Step success rate under aligned noise:**

The K-agent completes approximately 50-54% of reasoning steps under
aligned noise (compared to 75-100% under orthogonal noise). The MI-agent
completes 100% of steps under both noise conditions. The MI-agent's
perfect step rate under aligned noise is not a success — it is the
failure mode. It proceeds confidently through a degraded representational
geometry that the K-agent detects and refuses to propagate.

This replicates the behavioral gap observed in the core toy model
(System A: 97.1% accuracy, System B: 78.0%) but at the level of
agent decision-making rather than population-code readout. The
19-point gap in the toy model corresponds here to a ~45-50 point
gap in step completion rate between conditions that are identical
by every MI-visible measure.

---

### Interpretation

The MI-agent's 100% step success rate under aligned noise should
be read as a false positive rate of 100%. It has no mechanism to
distinguish a navigable representation from a decoherent one.
The K-agent's reduced step rate under aligned noise reflects genuine
detection of representational decoherence — the same phenomenon
the toy model demonstrates at the population-code level, now
operating as a real-time architectural constraint.

This demonstrates that K is not merely a useful post-hoc evaluation
metric. It is a computationally tractable quantity that can be
maintained incrementally, computed at each reasoning step, and
used to gate agent behavior in ways that are measurably superior
to MI-only baselines. The scaffold for a K-grounded cognitive
architecture is the direct applied extension of the claims made
in the core paper.

---

### Reproducing This Demonstration

    python3 agent_demo.py

Output: figures/agent_demo_results.png

Runtime: approximately 10-15 seconds on a standard laptop.
