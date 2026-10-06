"""LTEEG: point-level (per-sample) event detection on long-term EEG.

Subpackages
    data        recordings, annotations, preprocessing, cache, windows, datasets
    models      model contract and registry (SeizureTransformer baseline, TCN)
    engine      training loop, optimizers, EMA, checkpoints
    inference   continuous long-range inference and post-processing
    evaluation  SzCORE sample/event scoring, threshold-free metrics, reports, sweeps
"""

__version__ = "0.1.0"
