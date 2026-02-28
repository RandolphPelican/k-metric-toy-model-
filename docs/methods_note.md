# Methods Notes

## Raw MI Divergence Between Systems

Note that raw MI on the task-axis projection also differs between systems,
consistent with the noise construction: System B's aligned noise directly
attenuates task-relevant signal variance. This makes the comparison more
conservative — K must explain performance variance beyond even this raw
signal difference. That K does so confirms it captures navigability
structure independent of raw signal strength.

## Navigability Criterion

A trial's representation was classified as navigable if a linear decoder
trained on an independent subset of trials assigned confidence to the
correct task label exceeding the 95th percentile of a permutation null
distribution. The permutation null distribution was computed once per
decoder using training data only and applied uniformly across held-out
trials. This criterion is data-adaptive, threshold-free, and consistent
with standard permutation-based inference in population coding neuroscience.

## Denominator Justification

Contextual interference, order effects, and instability of partial
representations motivate the intuition of coherence loss under noise.
We refer to this intuitively as decoherence by analogy; however, all
operational definitions are grounded purely in information theory.

K = I(task; population)_navigable / H_max(population | noise covariance)

Where H_max is the maximum entropy of the population response distribution
under the empirical noise covariance, computed as:

H_max = 0.5 * log(det(2 * pi * e * Sigma_noise))
