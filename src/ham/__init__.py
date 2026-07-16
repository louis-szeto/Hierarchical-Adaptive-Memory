"""HAM: Utility-Rate Adaptive (Hierarchical Adaptive) Memory for frozen LLMs.

A reproducible experiment harness that treats persistent memory for a frozen
language model as a rate-distortion / information-bottleneck / minimum-description-length
problem, and reports *physically serialized* bytes alongside tokens, latency, and quality.

The package is intentionally importable with only numpy/scipy/pyyaml/pandas installed;
heavy dependencies (transformers, torch, faiss, sentence-transformers, zstandard,
datasets, matplotlib, codecarbon, psutil) are optional and degrade gracefully.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
