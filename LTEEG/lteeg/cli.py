"""Command-line interface.

    python -m lteeg <command> [--config configs/chbmit.yaml] [--set key=value ...]

Commands
    inspect         validate data + annotations, print per-patient / per-split statistics
    prepare         build the preprocessed signal cache (optionally in parallel)
    train           train with long-range validation, then evaluate the best checkpoint
    evaluate        long-range evaluation of a checkpoint on a split or patients
    sweep           re-score stored probabilities over thresholds / min durations
    compare         side-by-side table of several runs (pooled, macro, worst patient, per patient)
    predict         annotate arbitrary HDF5 recordings (events TSV per file)
    check-model     verify a model plugs into the framework (shapes, speed, overfit test)
    make-synthetic  write a small CHB-MIT-like synthetic dataset for smoke tests

Without ``--config`` the CHB-MIT config shipped in ``configs/chbmit.yaml`` is used.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "chbmit.yaml"


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None, help=f"YAML config (default: {DEFAULT_CONFIG})")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config entries, e.g. --set train.lr=3e-4 model.name=tcn")


def _load(args: argparse.Namespace):
    from .config import load_config

    path = args.config or DEFAULT_CONFIG
    return load_config(path, args.set)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lteeg", description="Point-level seizure detection on long-term EEG")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="validate the dataset and print statistics")
    _add_config_args(p)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--out", type=Path, default=None, help="optional JSON report path")

    p = sub.add_parser("prepare", help="build the preprocessed cache")
    _add_config_args(p)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    p.add_argument("--force", action="store_true", help="rebuild even if up to date")

    p = sub.add_parser("train", help="train a model")
    _add_config_args(p)
    p.add_argument("--resume", type=Path, default=None, help="run directory to resume (uses its config.yaml)")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes for cache building")
    p.add_argument("--dry-run", action="store_true", help="prepare everything, then exit before training")

    p = sub.add_parser("evaluate", help="evaluate a checkpoint on complete recordings")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override the checkpoint's config, e.g. --set data.root=D:/EEG/CHB-MIT postprocess.threshold=0.6")
    p.add_argument("--splits", nargs="+", default=["dev"])
    p.add_argument("--patients", nargs="*", default=[], help="evaluate these patients instead of splits")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument("--no-cache", action="store_true", help="preprocess on the fly instead of using the cache")
    p.add_argument("--no-ema", action="store_true", help="use raw weights even if EMA weights exist")

    p = sub.add_parser("sweep", help="re-score stored probabilities")
    p.add_argument("eval_dir", type=Path, help="directory written by 'evaluate' (contains manifest.json)")
    p.add_argument("--config", type=Path, default=None,
                   help="config for scoring/post-processing (default: the run's config.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--thresholds", nargs="*", type=float, default=None)
    p.add_argument("--min-durations", nargs="*", type=float, default=None)
    p.add_argument("--metric", default="event_f1_pooled")

    p = sub.add_parser("compare", help="compare evaluation results of several runs")
    p.add_argument("paths", nargs="+", type=Path, help="run directories, eval directories or summary.json files")
    p.add_argument("--out", type=Path, default=None, help="optional CSV path")

    p = sub.add_parser("predict", help="annotate recordings with a trained model")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("inputs", nargs="+", type=Path, help="HDF5 files or directories")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--save-probs", action="store_true")

    p = sub.add_parser("check-model", help="sanity-check a model against the framework contract")
    _add_config_args(p)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--overfit-steps", type=int, default=0)

    p = sub.add_parser("make-synthetic", help="write a synthetic CHB-MIT-like dataset")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-patients", type=int, default=4)
    p.add_argument("--dev-patients", type=int, default=2)
    p.add_argument("--records", type=int, default=3)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from .config import ConfigError
    from .data.annotations import AnnotationError
    from .data.store import DataValidationError
    from .registry import RegistryError
    from .utils.logging import setup_logging

    setup_logging()
    try:
        _run(args)
    except (ConfigError, RegistryError, DataValidationError, AnnotationError, FileNotFoundError,
            FileExistsError) as e:
        if os.environ.get("LTEEG_DEBUG"):
            raise
        print(f"\nerror ({type(e).__name__}): {e}\n(set LTEEG_DEBUG=1 for a traceback)", file=sys.stderr)
        return 2
    return 0


def _run(args: argparse.Namespace) -> None:
    from . import workflows

    if args.command == "inspect":
        workflows.inspect_data(_load(args), args.splits, args.out)
    elif args.command == "prepare":
        workflows.prepare_cache(_load(args), args.splits, args.jobs, args.force)
    elif args.command == "train":
        if args.resume:
            from .config import load_config

            cfg = load_config(Path(args.resume) / "config.yaml", args.set)
        else:
            cfg = _load(args)
        workflows.train(cfg, resume=args.resume, jobs=args.jobs, dry_run=args.dry_run)
    elif args.command == "evaluate":
        workflows.evaluate(args.checkpoint, args.set, args.splits, args.patients, args.out, args.jobs,
                           run_sweep=not args.no_sweep, use_cache=not args.no_cache, use_ema=not args.no_ema)
    elif args.command == "sweep":
        from .config import load_config
        from .evaluation.sweep import sweep

        cfg_path = args.config
        if cfg_path is None:
            for parent in [args.eval_dir] + list(Path(args.eval_dir).resolve().parents)[:3]:
                if (Path(parent) / "config.yaml").is_file():
                    cfg_path = Path(parent) / "config.yaml"
                    break
        cfg = load_config(cfg_path or DEFAULT_CONFIG, args.set)
        sweep(args.eval_dir, cfg, args.thresholds or cfg.evaluation.sweep_thresholds, args.min_durations, args.metric)
    elif args.command == "compare":
        from .evaluation.compare import compare

        compare(args.paths, args.out)
    elif args.command == "predict":
        workflows.predict(args.checkpoint, args.inputs, args.out, args.set, args.save_probs)
    elif args.command == "check-model":
        workflows.check_model(_load(args), args.batch_size, args.overfit_steps)
    elif args.command == "make-synthetic":
        _make_synthetic(args)


def _make_synthetic(args: argparse.Namespace) -> None:
    from .synthetic import make_synthetic_dataset
    from .utils import write_json

    n_tr, n_dev = args.train_patients, args.dev_patients
    patients = [f"chb{i:02d}" for i in range(1, n_tr + n_dev + 1)]
    manifest = make_synthetic_dataset(args.out, patients, records_per_patient=args.records,
                                      record_minutes=args.minutes, seed=args.seed)
    split = {"description": "synthetic smoke-test split", "train": patients[:n_tr], "dev": patients[n_tr:],
             "test": [], "expected_seizures": {
                 "train": sum(len(r["events"]) for p in patients[:n_tr] for r in manifest[p]),
                 "dev": sum(len(r["events"]) for p in patients[n_tr:] for r in manifest[p])}}
    write_json(Path(args.out) / "split.json", split)
    print(f"Synthetic dataset written to {args.out}; split file {Path(args.out) / 'split.json'}")
    print(f"Try: python -m lteeg train --set data.root={args.out} data.split_file={Path(args.out) / 'split.json'} "
          f"model.name=tcn model.params={{}} train.epochs=5 train.batch_size=16")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
