# Methodological Appendix (generic)

This appendix describes the evaluation methodology in venue-agnostic terms. It
deliberately contains **no repository paths, no command-line details, and no tool
names**, so that it can be reproduced by any comparable implementation and included
as an online appendix. Concrete file/CLI details live only in the repository README.

## Problem framing

Persistent memory for a frozen language model is framed as a rate–distortion /
information-bottleneck / minimum-description-length problem: given a stream of
observations and a bounded storage/token budget, decide *what* to keep, *at what
precision*, and *in what organized form*, so as to maximize downstream task success
per stored bit. No claim of Shannon- or Kolmogorov-optimality is made; the
optimal code length is uncomputable and is used only as a conceptual North Star.

## Independent variable and controls

The base language model is **frozen** and identical across all conditions. Within
a run, every condition shares the same set of examples, the same prompt template,
the same decoding parameters (greedy/deterministic), the same tokenizer, the same
embedding function, the same random seed, and the same evaluator. Consequently the
**only** independent variable is the memory policy: how observations are chunked,
scored for importance, assigned to tiers, consolidated, quantized, retrieved, and
assembled into context. These shared invariants are hashed into a fingerprint and
recorded in the run manifest so that a reviewer can verify fairness.

## Lifecycle stage

Two distinct artifacts are reported at different lifecycle stages and never
conflated: (i) an **inference-time external/persistent memory** system over frozen
weights (the evaluated proof of concept), and (ii) a **proposed architecture-level**
memory layer that reads/fuses/writes around a reasoning block and may be frozen or
jointly trained (a prototype, verified for shape and gradient behavior on
self-contained blocks, and explicitly **not** benchmarked). Generic hidden-state
injection into arbitrary pretrained models is left as future work.

## Conditions

The fair comparison spans: (a) a no-memory control that isolates parametric
knowledge; (b) a full-context control that places the entire history in the prompt
uncompressed; (c) an uncompressed retrieval baseline that is the fidelity anchor;
(d) a recency/FIFO forgetting policy that evicts by age regardless of utility;
(e) a static-prototype policy without adaptive consolidation; (f) a uniform
(non-utility) quantization policy; and (g) the proposed hierarchical adaptive
memory. Single-signal ablations remove one importance signal (recency, novelty,
reuse), disable consolidation, randomize tier assignment, or restrict retrieval to
lexical matching. Every condition is an implemented behavioral analogue under one
harness; none is presented as a reproduction of a specific external system, and no
externally reported metric is mixed into a measured local table.

## Outcome measures

For each condition the following are measured on real run outputs: task quality
(exact match, token-F1, and a containment-or-exact task score); retrieval quality
(recall@k and mean reciprocal rank against gold memory identity, where available);
prompt/total token counts using the model's exact tokenizer; **physically
serialized** on-disk bytes of the memory store (not an estimate), with the logical
uncompressed size and their ratio; retrieval, prefill, decode, and total latency;
and peak resource use. Per-condition deltas against the no-memory and
uncompressed-retrieval anchors are reported.

## Statistics

Paired comparisons across the shared example set use bootstrap confidence intervals
on the mean difference, a permutation test for significance, and — for the central
claim that compressed memory is not worse than uncompressed retrieval beyond a small
margin — a non-inferiority test at a pre-registered margin. A paired binary-outcome
test complements the continuous analysis.

## Reproducibility and honesty safeguards

Runs on a deterministic mock backend are watermarked as smoke tests and are never
presented as scientific results; small real-model runs on limited samples are
labeled explicitly as proofs of concept, with model identity, revision, and hardware
recorded, and are never presented as publication evidence. When required data or
dependencies are unavailable, the pipeline fails loudly with guidance rather than
substituting fabricated values.
