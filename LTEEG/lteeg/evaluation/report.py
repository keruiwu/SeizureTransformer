"""Evaluation reports: CSV tables, JSON summary and a Markdown overview."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils import write_json
from .runner import EvalResult

TABLE_COLUMNS = [
    "patient", "records", "hours", "event_ref", "event_tp", "event_fp",
    "event_sensitivity", "event_precision", "event_f1", "event_fp_per_day",
    "sample_sensitivity", "sample_precision", "sample_f1", "auroc", "auprc",
]


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    columns = columns or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def console_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    cells = [[str(c) for c in columns]] + [[_fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    lines = ["  ".join(v.rjust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)


SHORT_COLUMNS = ["patient", "hours", "event_ref", "event_tp", "event_fp", "event_sensitivity",
                 "event_precision", "event_f1", "event_fp_per_day", "sample_f1", "auprc"]


def write_report(result: EvalResult, out_dir: Path, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = result.metrics()
    patients = result.patient_rows()
    records = result.record_rows()
    pooled = result.pooled_row()
    macro = dict(_macro_row(metrics), patient="MACRO(mean)")

    write_csv(out_dir / "patients.csv", patients + [pooled, macro], TABLE_COLUMNS)
    write_csv(out_dir / "records.csv", records, ["record"] + TABLE_COLUMNS)
    event_rows, fa_rows = [], []
    for r in result.records:
        for d in r.event_details:
            event_rows.append({"patient": r.patient, "record": r.record, **d})
        for fa in r.fp_events:
            fa_rows.append({"patient": r.patient, "record": r.record, **fa})
    write_csv(out_dir / "events.csv", event_rows,
              ["patient", "record", "ref_start", "ref_end", "detected", "latency_sec", "coverage"])
    write_csv(out_dir / "false_alarms.csv", fa_rows, ["patient", "record", "start", "end", "duration"])
    write_json(out_dir / "summary.json", {"label": result.label, "metrics": metrics, "patients": patients,
                                          "context": context or {}})
    md = [f"# Evaluation: {result.label}", ""]
    if context:
        md += [f"- {k}: {v}" for k, v in context.items()] + [""]
    md += ["## Per patient (counts pooled over each patient's recordings)", "",
           markdown_table(patients + [pooled, macro], TABLE_COLUMNS), "",
           "POOLED: counts summed over all recordings. MACRO: unweighted mean over patients "
           "(SzCORE avg_per_subject). Event scoring: SzCORE (tolerance -30 s/+60 s, merge < 90 s, "
           "split > 300 s); sample scoring on a 1 Hz grid; fp_per_day = false-alarm events per 24 h."]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return metrics


def _macro_row(m: Dict[str, float]) -> Dict[str, Any]:
    return {k[: -len("_macro")]: v for k, v in m.items() if k.endswith("_macro")}


def short_summary(result: EvalResult) -> str:
    rows = result.patient_rows() + [result.pooled_row(), dict(_macro_row(result.metrics()), patient="MACRO")]
    return console_table(rows, SHORT_COLUMNS)
