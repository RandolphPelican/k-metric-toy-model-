# K Metric Toy Model

Two systems with identical signal strength, dimensionality, and total noise volume can exhibit sharply different cognitive performance depending solely on the alignment of noise with task-relevant axes — a distinction captured by the coherent-information fraction K but missed by raw or navigable mutual information.

---

## What This Is

A fully reproducible computational proof of concept for the Coherent Information Fraction (K), a substrate-agnostic metric of cognitive capacity grounded in population-code mutual information.

This repository accompanies the paper:

"The Coherent-Information Fraction: A Substrate-Agnostic Metric of Cognitive Capacity Grounded in Population-Code Mutual Information" J. Stabler, Independent Researcher bioRxiv preprint link -- coming soon

---

## The Core Demonstration

Two synthetic neural populations are constructed, matched on every classical metric. Signal strength 2.0 for both. Total noise volume (trace) 50.0 for both. Dimensionality (participation ratio) 49.36 for both. The only difference: System A has noise orthogonal to the task axis, System B has noise aligned with the task axis.

Results:

Raw MI: A = 0.658 nats B = 0.215 nats Navigable MI: A = 0.683 nats B = 0.633 nats K: A = 1.000 B = 0.370 Behavioral accuracy: A = 97.1% B = 78.0%

K predicts the 19-point behavioral gap. Raw MI and dimensionality do not.

---

## The K Metric

K = I(task; population)_navigable / H_max(population | noise covariance)

Numerator: mutual information between task labels and population responses, restricted to navigable trials only.

Denominator: maximum entropy of the population response distribution under the empirical noise covariance, computed in the task-relevant subspace as:

H_max = 0.5 * log(2 * pi * e * sigma^2_task)

where sigma^2_task = task_axis^T @ Sigma @ task_axis

Contextual interference, order effects, and instability of partial representations motivate the intuition of coherence loss under noise. We refer to this intuitively as decoherence by analogy; however, all operational definitions are grounded purely in information theory.

---

## Navigability Criterion

A trial's representation is classified as navigable if a linear decoder trained on an independent subset of trials assigns confidence to the correct task label exceeding the 95th percentile of a permutation null distribution. The permutation null distribution was computed once per decoder using training data only and applied uniformly across held-out trials. This criterion is data-adaptive, threshold-free, and consistent with standard permutation-based inference in population coding neuroscience.

---

## Repository Structure

k-metric-toy-model/
├── main.py                    # Core toy model -- one command
├── agent_demo.py              # K as real-time decision gate
├── clutter_index.py           # Geometric state classifier
├── drift_stability.py         # K stability across neural drift
├── multiplexing_load.py       # K under concurrent task load
├── src/
│   ├── noise_construction.py
│   ├── response_generation.py
│   ├── permutation_test.py
│   ├── mi_estimation.py
│   ├── k_computation.py
│   ├── behavioral_task.py
│   └── plotting.py
├── figures/
│   ├── k_metric_results.png
│   ├── agent_demo_results.png
│   ├── clutter_index_results.png
│   ├── drift_stability_results.png
│   └── multiplexing_load_results.png
├── docs/
│   └── methods_note.md
└── requirements.txt

---

## Quickstart

git clone https://github.com/RandolphPelican/k-metric-toy-model-.git
cd k-metric-toy-model-
pip install -r requirements.txt
python3 main.py

Runtime: approximately 90 seconds on a standard laptop. Output: verified console output + figures/k_metric_results.png

---

## Companion Demonstrations

Four additional scripts extend K into new domains. Each is self-contained and runnable independently.

**agent_demo.py** -- K as a real-time decision gate across three multi-step reasoning tasks. K-agent shows 6-172x greater noise-geometry sensitivity than MI-only baseline. The MI-agent's 100% step rate under aligned noise is a 100% false positive rate -- it cannot distinguish navigable from decoherent representations.

**clutter_index.py** -- Five geometric states (FLOW, INTERFERENCE, OVERLOAD, DISENGAGEMENT, TACHYPSYCHIA) that are indistinguishable by MI and dimensionality are cleanly separated by K. Critical finding: INTERFERENCE and OVERLOAD require opposite interventions but look identical to MI. Same treatment, wrong outcome for one.

**drift_stability.py** -- K is 40.9x more stable than single-neuron MI across 15%/epoch representational drift. K measures manifold geometry not neuron identity, so it survives the rotation of which neurons carry the pattern. Empirical basis for K's substrate-agnostic claim.

**multiplexing_load.py** -- Both K and MI degrade under concurrent task loading. K provides an earlier threshold signal but MI shows greater total sensitivity at high load levels. This is an honest boundary condition: when concurrent tasks add real signal rather than noise, MI correctly detects total information loss. K's advantage in this regime is geometric specificity -- identifying where degradation is occurring rather than just that it is. This result is retained as-is rather than tuned to favor K.

---

## Key Design Decisions

Why project onto the task axis for MI estimation? PCA finds the maximum variance direction, which in System B is the noise axis rather than the signal axis. Projecting onto the known task axis gives a consistent and theoretically motivated comparison across systems.

Why use a 1D denominator? H_max computed over all 50 dimensions dwarfs the navigable MI, producing K values near zero for both systems and destroying the contrast. The denominator should answer: what is the maximum information available in the task-relevant subspace given the noise structure? The 1D task-axis projection answers exactly that question.

Why does raw MI differ between systems? System B's noise is aligned with the task axis, which directly attenuates task-relevant signal variance in the projection. This makes the comparison conservative -- K must explain performance variance beyond even this raw signal difference. That K does so confirms it captures navigability structure independent of raw signal strength.

Why permutation-based navigability? A fixed accuracy threshold introduces a free parameter that hostile reviewers can challenge. The 95th percentile of a permutation null distribution is data-adaptive, requires no threshold selection, and is consistent with standard practice in neuroimaging and population coding neuroscience.

---

## Theoretical Background

K is motivated by three converging observations.

First, classical MI is noise-blind. Two systems with identical raw MI can have radically different cognitive performance if their noise structures differ in direction relative to task-relevant axes.

Second, navigability is not captured by dimensionality. Participation ratio and other dimensionality measures are matched between System A and System B by construction, yet performance differs by 19 points.

Third, the denominator must be task-local. Maximum entropy computed over the full representational space conflates task-relevant and task-irrelevant dimensions. K uses the noise variance along the task axis specifically, giving a meaningful capacity fraction.

---

## Relation to Existing Literature

Panzeri et al. (2007) -- bias-corrected MI estimation in neural population codes.

Busemeyer and Bruza (2012) -- Hilbert space geometry for cognitive modeling and quantum cognition.

Rao and Ballard (1999) -- hierarchical predictive coding in visual cortex.

Friston (2010) -- free energy principle and precision-weighted prediction errors.

Kriegeskorte and Kievit (2013) -- representational geometry in neural systems.

---

## Reproducibility

All random seeds are fixed. Results are fully deterministic.

Seed 42 -- task axis construction
Seed 0  -- System A response generation
Seed 1  -- System B response generation
Seed 42 -- permutation null distribution
Seed 42 -- behavioral task simulation

---

## License

MIT

---

## Contact

J. Stabler
Independent Researcher, Boca Raton FL
GitHub: https://github.com/RandolphPelican

**attention_gain.py** -- K shows an inverted-U response to attentional gain,
peaking at mid-range attention and declining at very high gain even as raw MI
saturates monotonically. This matches the well-known norepinephrine inverted-U
curve empirically -- too little attention and K collapses, sweet spot around
gain=0.5, too much and K declines as H_max grows faster than navigable MI.
Maps directly to dopamine and acetylcholine modulation of task-axis amplitude
in biological systems. MI misses the peak entirely.

**comprehension_vs_repetition.py** -- Comprehension-based learning produces
higher K across all probe axes vs repetition-based learning at matched accuracy.
Both agents use identical node count. Comprehension achieves 1.18x greater K
per active node -- elegance is geometric not structural. Repetition carves one
deep groove. Comprehension builds a navigable relational neighborhood. Accuracy
alone cannot distinguish them. K can. Novel prediction: expert geometries should
show higher K than novice geometries matched on behavioral accuracy.
