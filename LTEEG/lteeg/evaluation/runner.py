"""Long-range evaluation: full-recording inference -> post-processing -> scoring.

Every recording is scored on its own (SzCORE convention), then counts are pooled:

* ``pooled``  - counts summed over all recordings of all patients (micro average; this
  is what the original training script optimized);
* ``macro``   - metrics computed per patient (counts summed over the patient's
  recordings), then averaged across patients (SzCORE ``avg_per_subject``). The
  per-patient table is always written, because a pooled number can hide the one or
  two patients on which a detector fails completely.

Threshold-free metrics (AUROC, AUPRC) use the 1 Hz-binned probability trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import Config, PostprocessConfig, ScoringConfig
from ..data.annotations import events_to_intervals
from ..data.preprocess import Pipeline
from ..data.store import Recording
from ..inference.longrange import predict_recording
from ..inference.postprocess import postprocess
from ..task import Task
from ..utils import format_duration
from ..utils.logging import get_logger
from .ranking import average_precision, bin_mean, roc_auc
from .szcore import Counts, event_scoring, events_from_mask, mask_from_events, sample_scoring

log = get_logger("evaluation")


@dataclass
class RecordResult:
    patient: str
    record: str
    fs: float
    n_samples: int
    sample: Counts
    event: Counts
    ref_events: List[tuple]
    hyp_events: List[tuple]
    event_details: List[dict]
    fp_events: List[dict]
    auc_scores: np.ndarray = field(repr=False)
    auc_labels: np.ndarray = field(repr=False)
    nll_sum: float = 0.0

    @property
    def hours(self) -> float:
        return self.n_samples / self.fs / 3600


def reference_mask(rec: Recording, fs: float) -> np.ndarray:
    n = rec.n_samples_at(fs)
    mask = np.zeros(n, dtype=bool)
    for a, b, _ in events_to_intervals(rec.events, fs, n):
        mask[a:b] = True
    return mask


def score_record(patient: str, record: str, prob: np.ndarray, ref_mask: np.ndarray, fs: float,
                 post: PostprocessConfig, scoring: ScoringConfig, auc_fs: float) -> RecordResult:
    """Score one probability trace against its reference mask (pure numpy; reused by sweeps)."""
    n = len(prob)
    if len(ref_mask) != n:
        raise ValueError(f"{patient}/{record}: prediction length {n} != reference length {len(ref_mask)}")
    hyp_mask = postprocess(prob, fs, post)
    ref_ev = events_from_mask(ref_mask, fs)
    hyp_ev = events_from_mask(hyp_mask, fs)
    s = sample_scoring(ref_ev, hyp_ev, n, fs, scoring.sample_fs)
    e, details, fps = event_scoring(ref_ev, hyp_ev, n, fs, scoring)
    p = np.clip(prob.astype(np.float64), 1e-7, 1 - 1e-7)
    nll = -float(np.sum(np.where(ref_mask, np.log(p), np.log1p(-p))))
    auc_s = bin_mean(prob, fs, auc_fs)
    auc_y = mask_from_events(ref_ev, auc_fs, len(auc_s))
    return RecordResult(patient, record, fs, n, s, e, ref_ev, hyp_ev, details, fps, auc_s, auc_y, nll)


@dataclass
class EvalResult:
    records: List[RecordResult]
    label: str = ""

    # ------------------------------------------------------------------ aggregation
    def patients(self) -> List[str]:
        seen: List[str] = []
        for r in self.records:
            if r.patient not in seen:
                seen.append(r.patient)
        return seen

    def patient_rows(self) -> List[dict]:
        rows = []
        for p in self.patients():
            recs = [r for r in self.records if r.patient == p]
            rows.append(_summary_row(p, recs))
        return rows

    def pooled_row(self) -> dict:
        return _summary_row("POOLED", self.records)

    def record_rows(self) -> List[dict]:
        return [dict(_summary_row(r.patient, [r]), record=r.record) for r in self.records]

    def metrics(self) -> Dict[str, float]:
        pooled = _summary_row("pooled", self.records)
        rows = self.patient_rows()
        out: Dict[str, float] = {"n_patients": len(rows), "n_records": len(self.records),
                                 "hours": pooled["hours"], "ref_events": pooled["event_ref"]}
        keys = [k for k in pooled if k.startswith(("sample_", "event_", "auroc", "auprc", "nll"))
                and k not in ("event_ref", "event_tp", "event_fp", "sample_ref", "sample_tp", "sample_fp")]
        for k in keys:
            out[f"{k}_pooled"] = pooled[k]
            vals = np.array([r[k] for r in rows], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            out[f"{k}_macro"] = float(np.mean(finite)) if len(finite) else float("nan")
            out[f"{k}_macro_std"] = float(np.std(finite)) if len(finite) else float("nan")
            out[f"{k}_macro_min"] = float(np.min(finite)) if len(finite) else float("nan")
            out[f"{k}_macro_median"] = float(np.median(finite)) if len(finite) else float("nan")
        lat = [d["latency_sec"] for r in self.records for d in r.event_details if d["detected"]]
        out["event_latency_median_sec"] = float(np.median(lat)) if lat else float("nan")
        return out


def _summary_row(name: str, recs: Sequence[RecordResult]) -> dict:
    s = sum((r.sample for r in recs), Counts())
    e = sum((r.event for r in recs), Counts())
    scores = np.concatenate([r.auc_scores for r in recs]) if recs else np.zeros(0)
    labels = np.concatenate([r.auc_labels for r in recs]) if recs else np.zeros(0, bool)
    n_total = sum(r.n_samples for r in recs)
    row = {"patient": name, "records": len(recs), "hours": sum(r.hours for r in recs),
           "event_ref": e.ref_true, "event_tp": e.tp, "event_fp": e.fp,
           "sample_ref": s.ref_true, "sample_tp": s.tp, "sample_fp": s.fp}
    for prefix, c in (("sample", s), ("event", e)):
        for k, v in c.scores().items():
            row[f"{prefix}_{k}"] = v
    row["auroc"] = roc_auc(labels, scores)
    row["auprc"] = average_precision(labels, scores)
    row["nll"] = sum(r.nll_sum for r in recs) / max(1, n_total)
    return row


# ---------------------------------------------------------------------- driver
@torch.no_grad()
def evaluate_recordings(model: torch.nn.Module, recordings: Sequence[Recording], source, cfg: Config,
                        task: Task, device: torch.device, probs_dir: Optional[Path] = None,
                        label: str = "", log_each: bool = False) -> EvalResult:
    model.eval()
    window_pipeline = Pipeline(cfg.preprocess.window, cfg.fs) if cfg.preprocess.window else None
    results: List[RecordResult] = []
    t0 = time.time()
    hours = 0.0
    for i, rec in enumerate(recordings, 1):
        signal = source.open(rec)
        probs = predict_recording(model, signal, task, cfg.window_samples, cfg.inference_hop_samples,
                                  cfg.inference.batch_size, device, cfg.inference.combine, cfg.inference.tail,
                                  cfg.inference.amp, window_pipeline)
        event_prob = task.event_prob(probs)
        res = score_record(rec.patient, rec.stem, event_prob, reference_mask(rec, cfg.fs), cfg.fs,
                           cfg.postprocess, cfg.scoring, cfg.evaluation.auc_fs)
        results.append(res)
        hours += res.hours
        if probs_dir is not None and cfg.evaluation.save_probs:
            d = Path(probs_dir) / rec.patient
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{rec.stem}.npy", probs.astype(cfg.evaluation.probs_dtype))
        if log_each:
            log.info(f"[{label} {i}/{len(recordings)}] {rec.key}: ref={res.event.ref_true} tp={res.event.tp} "
                     f"fp={res.event.fp}")
    elapsed = time.time() - t0
    log.info(f"[{label}] scored {len(recordings)} recordings ({hours:.1f} h of EEG) in {format_duration(elapsed)} "
             f"({hours * 3600 / max(elapsed, 1e-9):.0f}x real time)")
    return EvalResult(results, label)
