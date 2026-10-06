"""Operating-point sweeps on stored probability traces (no re-inference).

``lteeg evaluate`` stores every recording's probability trace plus a manifest with
the reference events. Re-scoring them under different thresholds / post-processing
settings is then cheap. Choosing the operating point on a split and reporting on
the *same* split is optimistic; the CSV marks the best row explicitly so it can be
transferred to a held-out split instead.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..utils import read_json, write_json
from ..utils.logging import get_logger
from .report import console_table, write_csv
from .runner import EvalResult, score_record
from .szcore import mask_from_events

log = get_logger("evaluation.sweep")

SWEEP_COLUMNS = ["threshold", "min_duration_sec", "event_f1_pooled", "event_sensitivity_pooled",
                 "event_precision_pooled", "event_fp_per_day_pooled", "event_f1_macro",
                 "event_sensitivity_macro", "event_fp_per_day_macro", "sample_f1_pooled", "sample_f1_macro"]


def write_manifest(path: Path, entries: List[dict]) -> None:
    write_json(path, {"version": 1, "records": entries})


def load_traces(eval_dir: Path) -> List[dict]:
    manifest = read_json(Path(eval_dir) / "manifest.json")
    out = []
    for e in manifest["records"]:
        prob = np.load(Path(eval_dir) / "probs" / e["patient"] / f"{e['record']}.npy").astype(np.float32)
        event_prob = prob[0] if e.get("mode", "binary") == "binary" else 1.0 - prob[0]
        ref = mask_from_events([tuple(x) for x in e["ref_events"]], e["fs"], e["n_samples"])
        out.append({**e, "prob": event_prob, "ref_mask": ref})
    return out


def sweep(eval_dir: Path, cfg: Config, thresholds: Sequence[float],
          min_durations: Optional[Sequence[float]] = None, metric: str = "event_f1_pooled") -> List[Dict]:
    traces = load_traces(eval_dir)
    min_durations = list(min_durations) if min_durations else [cfg.postprocess.min_duration_sec]
    rows: List[Dict] = []
    for md in min_durations:
        for th in thresholds:
            post = dataclasses.replace(cfg.postprocess, threshold=float(th), min_duration_sec=float(md))
            res = EvalResult([score_record(t["patient"], t["record"], t["prob"], t["ref_mask"], t["fs"], post,
                                           cfg.scoring, cfg.evaluation.auc_fs) for t in traces])
            m = res.metrics()
            rows.append({"threshold": float(th), "min_duration_sec": float(md),
                         **{k: m.get(k) for k in SWEEP_COLUMNS[2:]}})
    best = max(rows, key=lambda r: -np.inf if r[metric] is None or not np.isfinite(r[metric]) else r[metric])
    for r in rows:
        r["best"] = r is best
    write_csv(Path(eval_dir) / "sweep.csv", rows, SWEEP_COLUMNS + ["best"])
    log.info("Threshold sweep (choosing on this split is optimistic for this split):\n"
             + console_table(rows, SWEEP_COLUMNS))
    log.info(f"Best by {metric}: threshold={best['threshold']} min_duration={best['min_duration_sec']} "
             f"-> {metric}={best[metric]:.4f}")
    return rows
