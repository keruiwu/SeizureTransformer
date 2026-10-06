"""Typed experiment configuration.

The dataclass defaults in this module are the single source of truth for every
setting. ``configs/chbmit.yaml`` mirrors them one-to-one (``tests/test_config.py``
fails if the two drift apart). Experiment files can inherit with ``_base_: chbmit.yaml``
and list only what differs.

Loading is strict: unknown keys raise with a "did you mean" hint, values are
coerced to the declared types (e.g. ``"1e-4"`` -> float, which PyYAML would keep
as a string), and cross-field constraints are checked in :meth:`Config.validate`.

Dataset-specific facts (channel montage, sampling rate, file layout, patient split)
live in the ``data`` section and the split file; nothing outside ``lteeg.data``
needs to know about CHB-MIT.
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml

class ConfigError(ValueError):
    """Invalid configuration (unknown key, wrong type, inconsistent values)."""


# --------------------------------------------------------------------------------------
# CHB-MIT defaults (18-channel longitudinal bipolar "double banana", 256 Hz)
# --------------------------------------------------------------------------------------
CHBMIT_CHANNELS: List[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

# Left/right homologous derivations; used by the hemispheric-swap augmentation.
CHBMIT_SYMMETRIC_PAIRS: List[List[str]] = [
    ["FP1-F7", "FP2-F8"], ["F7-T7", "F8-T8"], ["T7-P7", "T8-P8"], ["P7-O1", "P8-O2"],
    ["FP1-F3", "FP2-F4"], ["F3-C3", "F4-C4"], ["C3-P3", "C4-P4"], ["P3-O1", "P4-O2"],
]


def _default_recording_pipeline() -> List[Dict[str, Any]]:
    # Same operators and order as the original SeizureTransformer pipeline:
    # per-recording z-score, 3rd-order Butterworth band-pass 0.5-120 Hz, IIR notches at 1 and 60 Hz.
    return [
        {"op": "zscore"},
        {"op": "bandpass", "low": 0.5, "high": 120.0, "order": 3},
        {"op": "notch", "freq": 1.0, "q": 30.0},
        {"op": "notch", "freq": 60.0, "q": 30.0},
    ]


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str = "seizure_transformer"
    output_dir: str = "runs/chbmit"
    seed: int = 0
    deterministic: bool = False
    timestamp: bool = True
    device: str = "auto"


@dataclass
class DataConfig:
    root: str = "F:/EEG/CHB-MIT"
    split_file: str = "chbmit_split.json"
    fold: Optional[int] = None
    file_pattern: str = "*.h5"
    signal_key: str = "signals"
    fs_attr: str = "fs"
    channel_attr: Optional[str] = None
    require_channel_names: bool = False
    channels: List[str] = field(default_factory=lambda: list(CHBMIT_CHANNELS))
    fs: float = 256.0
    resample_to: Optional[float] = None
    annotation_file: str = "{patient}_annotations.txt"
    annotation_section: str = "seizures"
    annotation_match: str = "exact"
    label_column: Optional[int] = None
    label_map: Dict[str, int] = field(default_factory=dict)
    default_label: int = 1
    end_tolerance_sec: float = 1.0
    overlapping_events: str = "error"
    exclude_records: List[str] = field(default_factory=list)
    cache_dir: str = "cache/chbmit"
    cache_dtype: str = "float32"
    symmetric_pairs: List[List[str]] = field(default_factory=lambda: copy.deepcopy(CHBMIT_SYMMETRIC_PAIRS))


@dataclass
class TaskConfig:
    mode: str = "binary"
    class_names: List[str] = field(default_factory=lambda: ["background", "seizure"])
    ignore_boundary_sec: float = 0.0


@dataclass
class PreprocessConfig:
    recording: List[Dict[str, Any]] = field(default_factory=_default_recording_pipeline)
    window: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WindowConfig:
    length_sec: float = 60.0
    train_stride_sec: float = 15.0
    include_tail: bool = False


@dataclass
class SamplingConfig:
    name: str = "balanced"
    boundary_ratio: float = 1.0
    full_ratio: float = 0.7
    background_ratio: float = 3.0
    seed: Optional[int] = None
    redraw_every_epoch: bool = False
    group_by: str = "none"
    jitter_sec: float = 0.0


@dataclass
class ModelConfig:
    name: str = "seizure_transformer"
    params: Dict[str, Any] = field(
        default_factory=lambda: {"dim_feedforward": 2048, "num_layers": 8, "num_heads": 4, "drop_rate": 0.1}
    )
    init_checkpoint: Optional[str] = None


@dataclass
class LossConfig:
    terms: List[Dict[str, Any]] = field(default_factory=lambda: [{"name": "bce", "weight": 1.0}])


@dataclass
class OptimizerConfig:
    name: str = "radam"
    lr: float = 1e-4
    weight_decay: float = 2e-5
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    momentum: float = 0.9
    decoupled_weight_decay: bool = False
    exclude_norm_bias_from_decay: bool = False


@dataclass
class SchedulerConfig:
    name: str = "none"
    warmup_epochs: float = 0.0
    min_lr_ratio: float = 0.0
    step_epochs: int = 10
    gamma: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 86
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_batches: int = 2
    accum_steps: int = 1
    grad_clip: Optional[float] = None
    amp: str = "none"
    ema_decay: Optional[float] = None
    compile: bool = False
    early_stopping: bool = False
    patience: int = 12
    min_delta: float = 0.0
    val_every: int = 1
    selection_metric: str = "event_f1_pooled"
    selection_mode: str = "max"
    log_every: int = 20
    max_nonfinite_steps: int = 20
    augment: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class InferenceConfig:
    batch_size: int = 8
    hop_sec: Optional[float] = None
    combine: str = "mean"
    tail: str = "align"
    amp: str = "none"


@dataclass
class PostprocessConfig:
    threshold: float = 0.8
    morph_ops: List[str] = field(default_factory=lambda: ["opening", "closing"])
    morph_kernel: int = 5
    min_duration_sec: float = 2.0
    merge_gap_sec: float = 0.0


@dataclass
class ScoringConfig:
    sample_fs: float = 1.0
    event_fs: float = 10.0
    tolerance_start: float = 30.0
    tolerance_end: float = 60.0
    min_overlap: float = 0.0
    max_event_duration: float = 300.0
    min_duration_between_events: float = 90.0


@dataclass
class EvaluationConfig:
    splits: List[str] = field(default_factory=lambda: ["dev"])
    save_probs: bool = True
    probs_dtype: str = "float16"
    auc_fs: float = 1.0
    sweep_thresholds: List[float] = field(
        default_factory=lambda: [round(0.05 * i, 2) for i in range(1, 20)]
    )


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ---------------------------------------------------------------- derived values
    @property
    def fs(self) -> float:
        """Sampling rate of the signals the model sees (after optional resampling)."""
        return float(self.data.resample_to or self.data.fs)

    @property
    def n_channels(self) -> int:
        return len(self.data.channels)

    @property
    def window_samples(self) -> int:
        return _seconds_to_samples(self.windows.length_sec, self.fs, "windows.length_sec")

    @property
    def train_stride_samples(self) -> int:
        return _seconds_to_samples(self.windows.train_stride_sec, self.fs, "windows.train_stride_sec")

    @property
    def inference_hop_samples(self) -> int:
        if self.inference.hop_sec is None:
            return self.window_samples
        return _seconds_to_samples(self.inference.hop_sec, self.fs, "inference.hop_sec")

    @property
    def num_classes(self) -> int:
        return len(self.task.class_names)

    @property
    def sampling_seed(self) -> int:
        return self.experiment.seed if self.sampling.seed is None else self.sampling.seed

    def resolve_path(self, value: str, base: Optional[str] = None) -> Path:
        """Resolve a possibly-relative path. ``base`` defaults to the working directory."""
        p = Path(value).expanduser()
        if p.is_absolute():
            return p
        return (Path(base) if base else Path.cwd()) / p

    @property
    def split_path(self) -> Path:
        # relative paths coming from a YAML file were already made absolute by load_yaml_tree
        return self.resolve_path(self.data.split_file)

    # ---------------------------------------------------------------- (de)serialization
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    # ---------------------------------------------------------------- validation
    def validate(self) -> "Config":
        errors: List[str] = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        d, t = self.data, self.task
        check(len(d.channels) > 0, "data.channels must not be empty")
        check(len(set(c.upper() for c in d.channels)) == len(d.channels), "data.channels contains duplicates")
        check(d.fs > 0, "data.fs must be positive")
        check(d.annotation_match in ("exact", "stem"), "data.annotation_match must be 'exact' or 'stem'")
        check(d.overlapping_events in ("error", "merge"), "data.overlapping_events must be 'error' or 'merge'")
        check(d.cache_dtype in ("float32", "float16"), "data.cache_dtype must be 'float32' or 'float16'")
        for pair in d.symmetric_pairs:
            check(len(pair) == 2, f"data.symmetric_pairs entries must be pairs, got {pair}")
            for ch in pair:
                check(ch.upper() in {c.upper() for c in d.channels},
                      f"data.symmetric_pairs references unknown channel {ch!r}")

        check(t.mode in ("binary", "multiclass"), "task.mode must be 'binary' or 'multiclass'")
        check(len(t.class_names) >= 2, "task.class_names needs a background class plus at least one event class")
        if t.mode == "binary":
            check(len(t.class_names) == 2, "binary task requires exactly two class_names")
        check(t.ignore_boundary_sec >= 0, "task.ignore_boundary_sec must be >= 0")

        try:
            ws = self.window_samples
            st = self.train_stride_samples
            hop = self.inference_hop_samples
            check(st <= ws, "windows.train_stride_sec must not exceed windows.length_sec")
            check(hop <= ws, "inference.hop_sec must not exceed windows.length_sec")
        except ValueError as e:
            errors.append(str(e))

        s = self.sampling
        check(s.name in ("balanced", "all"), "sampling.name must be 'balanced' or 'all'")
        check(s.group_by in ("none", "patient"), "sampling.group_by must be 'none' or 'patient'")
        check(0 <= s.boundary_ratio <= 1, "sampling.boundary_ratio is a fraction in [0, 1]")
        check(s.full_ratio >= 0 and s.background_ratio >= 0, "sampling ratios must be >= 0")
        check(s.jitter_sec >= 0, "sampling.jitter_sec must be >= 0")

        tr = self.train
        check(tr.epochs >= 1 and tr.batch_size >= 1 and tr.accum_steps >= 1, "train epochs/batch/accum must be >= 1")
        check(tr.amp in ("none", "fp16", "bf16"), "train.amp must be 'none', 'fp16' or 'bf16'")
        check(tr.selection_mode in ("max", "min"), "train.selection_mode must be 'max' or 'min'")
        check(tr.val_every >= 1, "train.val_every must be >= 1")
        check(self.inference.amp in ("none", "fp16", "bf16"), "inference.amp must be 'none', 'fp16' or 'bf16'")
        check(self.inference.combine in ("mean", "hann"), "inference.combine must be 'mean' or 'hann'")
        check(self.inference.tail in ("align", "pad"), "inference.tail must be 'align' or 'pad'")
        check(self.optimizer.name in ("adam", "adamw", "radam", "sgd"), "optimizer.name must be adam|adamw|radam|sgd")
        check(self.scheduler.name in ("none", "cosine", "step"), "scheduler.name must be none|cosine|step")

        p = self.postprocess
        check(0.0 <= p.threshold <= 1.0, "postprocess.threshold must be in [0, 1]")
        check(all(op in ("opening", "closing") for op in p.morph_ops), "postprocess.morph_ops: opening|closing")
        check(p.morph_kernel >= 1, "postprocess.morph_kernel must be >= 1")
        check(self.evaluation.probs_dtype in ("float16", "float32"), "evaluation.probs_dtype: float16|float32")

        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return self


def _seconds_to_samples(seconds: float, fs: float, name: str) -> int:
    n = seconds * fs
    if seconds <= 0 or abs(n - round(n)) > 1e-6:
        raise ConfigError(f"{name}={seconds} s is not a positive whole number of samples at fs={fs} Hz")
    return int(round(n))


# --------------------------------------------------------------------------------------
# dict -> dataclass with strict checking and type coercion
# --------------------------------------------------------------------------------------
def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin = get_origin(hint)
    if hint is Any:
        return value
    if origin is Union:
        args = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            if type(None) in get_args(hint):
                return None
            raise ConfigError(f"{where}: null is not allowed")
        return _coerce(value, args[0], where) if len(args) == 1 else value
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        return _from_dict(hint, value, where)
    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        (item_hint,) = get_args(hint) or (Any,)
        return [_coerce(v, item_hint, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        k_hint, v_hint = get_args(hint) or (Any, Any)
        return {_coerce(k, k_hint, where): _coerce(v, v_hint, f"{where}.{k}") for k, v in value.items()}
    if value is None:
        raise ConfigError(f"{where}: null is not allowed")
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected true/false, got {value!r}")
    if hint is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise ConfigError(f"{where}: expected an integer, got {value!r}")
    if hint is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):  # PyYAML parses "1e-4" as a string (YAML 1.1)
            try:
                return float(value)
            except ValueError:
                pass
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    if hint is str:
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
        raise ConfigError(f"{where}: expected a string, got {value!r}")
    return value


def _from_dict(cls: Any, data: Dict[str, Any], where: str = "config") -> Any:
    hints = get_type_hints(cls)
    known = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = [k for k in data if k not in known]
    if unknown:
        msgs = []
        for k in unknown:
            close = difflib.get_close_matches(str(k), list(known), n=1)
            msgs.append(f"{where}.{k}" + (f" (did you mean '{close[0]}'?)" if close else ""))
        raise ConfigError("Unknown configuration key(s): " + ", ".join(msgs))
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def config_from_dict(data: Dict[str, Any]) -> Config:
    return _from_dict(Config, data or {})


def _set_dotted(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if nxt is None:
            nxt = cur[k] = {}
        if not isinstance(nxt, dict):
            raise ConfigError(f"Cannot set '{dotted}': '{k}' is not a section")
        cur = nxt
    cur[keys[-1]] = value


def parse_override(text: str) -> tuple:
    """Parse ``key.sub=value``; the value is read as YAML (so ``null``, ``true``, ``[1,2]`` work)."""
    if "=" not in text:
        raise ConfigError(f"Override must look like key=value, got {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw) if raw.strip() != "" else ""


def load_yaml_tree(path: Union[str, Path], _seen: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Read a YAML config, following ``_base_`` (a path or list of paths relative to the file)
    so experiment files only state what differs from their base. A relative
    ``data.split_file`` is resolved against the file that defines it."""
    path = Path(path).resolve()
    seen = list(_seen or [])
    if path in seen:
        raise ConfigError(f"circular _base_ chain: {' -> '.join(str(p) for p in seen + [path])}")
    with open(path, "r", encoding="utf-8-sig") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    bases = loaded.pop("_base_", None)
    merged: Dict[str, Any] = {}
    for b in ([bases] if isinstance(bases, str) else list(bases or [])):
        merged = _deep_merge(merged, load_yaml_tree(path.parent / b, seen + [path]))
    split_file = (loaded.get("data") or {}).get("split_file")
    if isinstance(split_file, str) and not Path(split_file).expanduser().is_absolute():
        loaded["data"]["split_file"] = str(path.parent / split_file)
    return _deep_merge(merged, loaded)


def load_config(path: Optional[Union[str, Path]] = None, overrides: Optional[List[str]] = None,
                base: Optional[Dict[str, Any]] = None) -> Config:
    """Load a YAML config (optionally on top of ``base``, e.g. the config stored in a
    checkpoint) and apply ``key=value`` overrides. Missing keys keep dataclass defaults."""
    merged: Dict[str, Any] = copy.deepcopy(base) if base else {}
    if path is not None:
        merged = _deep_merge(merged, load_yaml_tree(path))
    for item in overrides or []:
        key, value = parse_override(item)
        _set_dotted(merged, key, value)
    return config_from_dict(merged).validate()


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("params", "label_map"):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
