from pathlib import Path

import pytest
import yaml

from lteeg.config import Config, ConfigError, config_from_dict, load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_yaml_matches_dataclass_defaults():
    """configs/chbmit.yaml must state exactly the dataclass defaults (single source of truth)."""
    with open(CONFIG_DIR / "chbmit.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    from_yaml = config_from_dict(raw).to_dict()
    defaults = Config().to_dict()
    assert from_yaml == defaults
    # and the yaml lists every key (no silent reliance on defaults)
    def keys(d, prefix=""):
        out = set()
        for k, v in d.items():
            out.add(prefix + k)
            if isinstance(v, dict) and k not in ("params", "label_map"):
                out |= keys(v, prefix + k + ".")
        return out
    assert keys(raw) == keys(defaults)


def test_default_config_loads_and_derives():
    cfg = load_config(CONFIG_DIR / "chbmit.yaml")
    assert cfg.window_samples == 60 * 256
    assert cfg.train_stride_samples == 15 * 256
    assert cfg.inference_hop_samples == cfg.window_samples
    assert cfg.n_channels == 18
    assert cfg.split_path == CONFIG_DIR / "chbmit_split.json"


def test_unknown_key_suggests():
    with pytest.raises(ConfigError, match="did you mean 'batch_size'"):
        load_config(None, ["train.batchsize=4"])


def test_override_types():
    cfg = load_config(None, ["optimizer.lr=3e-4", "train.grad_clip=1", "inference.hop_sec=null",
                             "train.augment=[{name: sign_flip}]"])
    assert cfg.optimizer.lr == pytest.approx(3e-4) and isinstance(cfg.optimizer.lr, float)
    assert cfg.train.grad_clip == 1.0
    assert cfg.train.augment == [{"name": "sign_flip"}]
    with pytest.raises(ConfigError):
        load_config(None, ["train.epochs=abc"])
    with pytest.raises(ConfigError):
        load_config(None, ["train.early_stopping=1"])


def test_cross_field_validation():
    with pytest.raises(ConfigError, match="train_stride_sec"):
        load_config(None, ["windows.train_stride_sec=90"])
    with pytest.raises(ConfigError, match="whole number of samples"):
        load_config(None, ["windows.length_sec=0.001"])
    with pytest.raises(ConfigError, match="symmetric_pairs"):
        load_config(None, ["data.symmetric_pairs=[[FP1-F7, XX]]"])


def test_split_file(tmp_path):
    from lteeg.data.split import load_split

    split = load_split(CONFIG_DIR / "chbmit_split.json")
    assert len(split.train) == 18 and len(split.dev) == 6
    assert "chb01" in split.train and "chb21" in split.train
    assert split.expected_seizures == {"train": 159, "dev": 39}
    bad = tmp_path / "bad.json"
    bad.write_text('{"train": ["chb01"], "dev": ["chb21"], "groups": [["chb01", "chb21"]]}')
    with pytest.raises(ValueError, match="same subject"):
        load_split(bad)
    bad.write_text('{"train": ["chb01"], "dev": ["chb01"]}')
    with pytest.raises(ValueError, match="both"):
        load_split(bad)
    folds = tmp_path / "folds.json"
    folds.write_text('{"folds": [{"train": ["a"], "dev": ["b"]}, {"train": ["b"], "dev": ["a"]}], "test": ["c"]}')
    s = load_split(folds, fold=1)
    assert s.train == ["b"] and s.dev == ["a"] and s.test == ["c"]


@pytest.mark.parametrize("path", sorted((CONFIG_DIR / "experiments").glob("*.yaml")))
def test_experiment_configs_load(path):
    cfg = load_config(path)
    assert cfg.split_path == CONFIG_DIR / "chbmit_split.json"  # resolved against the base file


def test_base_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("train: {epochs: 7, batch_size: 3}\nmodel: {params: {num_layers: 2}}\n")
    (tmp_path / "exp.yaml").write_text("_base_: base.yaml\ntrain: {epochs: 9}\n")
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg.train.epochs == 9 and cfg.train.batch_size == 3 and cfg.model.params == {"num_layers": 2}
    (tmp_path / "loop.yaml").write_text("_base_: loop.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "loop.yaml")
