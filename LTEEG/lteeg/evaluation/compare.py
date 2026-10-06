"""Side-by-side comparison of evaluation results (models, seeds, settings)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..utils import read_json
from .report import console_table, write_csv

KEY_METRICS = ["event_f1_pooled", "event_f1_macro", "event_f1_macro_std", "event_f1_macro_min",
               "event_sensitivity_pooled", "event_precision_pooled", "event_fp_per_day_pooled",
               "sample_f1_pooled", "auprc_pooled", "auroc_pooled"]


def find_summaries(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for p in map(Path, paths):
        if p.is_file() and p.name == "summary.json":
            out.append(p)
        elif (p / "summary.json").is_file():
            out.append(p / "summary.json")
        else:  # a run directory or a directory of runs
            out.extend(sorted(q for q in p.rglob("summary.json") if q.parent.parent.name == "eval"))
    return out


def compare(paths: Sequence[Path], out: Optional[Path] = None) -> List[Dict]:
    rows, patient_rows = [], []
    for s in find_summaries(paths):
        data = read_json(s)
        name = f"{s.parent.parent.parent.name}/{s.parent.name}" if s.parent.parent.name == "eval" else str(s.parent)
        m = data["metrics"]
        rows.append({"run": name, **{k: m.get(k) for k in KEY_METRICS}})
        patient_rows.append({"run": name, **{p["patient"]: p.get("event_f1") for p in data["patients"]}})
    if not rows:
        raise FileNotFoundError(f"no summary.json found under {list(map(str, paths))}")
    print(console_table(rows, ["run"] + KEY_METRICS))
    patients = sorted({k for r in patient_rows for k in r if k != "run"})
    print("\nPer-patient event F1:")
    print(console_table(patient_rows, ["run"] + patients))
    if out:
        write_csv(Path(out), rows, ["run"] + KEY_METRICS)
        write_csv(Path(out).with_name(Path(out).stem + "_patients.csv"), patient_rows, ["run"] + patients)
    return rows
