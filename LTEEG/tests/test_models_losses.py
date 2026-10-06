import pytest
import torch
import torch.nn.functional as F

from lteeg.config import load_config
from lteeg.losses import build_loss
from lteeg.models import build_model, count_parameters
from lteeg.models.seizure_transformer import SeizureTransformer, load_original_state_dict
from lteeg.task import Task, align_logits


@pytest.mark.parametrize("length", [15360, 15000, 1001])
def test_seizure_transformer_any_length(length):
    m = SeizureTransformer(in_channels=18, in_samples=length, num_layers=1, dim_feedforward=64).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 18, length))
    assert out.shape == (2, 1, length)


def test_seizure_transformer_default_size_and_multiclass():
    cfg = load_config(None, ["task.mode=multiclass", "task.class_names=[bg, focal, generalized]"])
    m = build_model(cfg, Task(cfg).num_outputs)
    assert 37.8e6 < count_parameters(m) < 37.9e6  # original minus the dead duplicate layer
    assert m.conv_d.out_channels == 3


def test_original_state_dict_conversion():
    m = SeizureTransformer(num_layers=1, dim_feedforward=64)
    sd = {f"module.{k}": v for k, v in m.state_dict().items()}
    sd["transformer_encoder_layer.linear1.weight"] = torch.zeros(1)
    sd["position_encoding.pe"] = torch.zeros(1)
    m.load_state_dict(load_original_state_dict(sd), strict=True)


def test_tcn_output_stride_alignment():
    cfg = load_config(None, ["model.name=tcn", "model.params={}"])
    m = build_model(cfg, 1)
    x = torch.randn(2, 18, cfg.window_samples)
    out = m(x)
    assert out.shape[-1] == cfg.window_samples // 8
    assert align_logits(out, cfg.window_samples).shape == (2, 1, cfg.window_samples)


def test_bce_matches_torch_and_ignores():
    cfg = load_config(None)
    loss = build_loss(cfg)
    logits = torch.randn(3, 1, 50)
    y = torch.randint(0, 2, (3, 50))
    total, parts = loss(logits, y)
    assert torch.allclose(total, F.binary_cross_entropy_with_logits(logits[:, 0], y.float()))
    y2 = y.clone()
    y2[:, :10] = -1
    total2, _ = loss(logits, y2)
    assert torch.allclose(total2, F.binary_cross_entropy_with_logits(logits[:, 0, 10:], y[:, 10:].float()))


@pytest.mark.parametrize("terms", [
    "[{name: focal, gamma: 2.0}]", "[{name: dice}]", "[{name: bce, pos_weight: 3.0}, {name: tmse, weight: 0.15}]",
    "[{name: bce, label_smoothing: 0.1}, {name: dice, weight: 0.5}]"])
def test_binary_losses_finite_and_differentiable(terms):
    cfg = load_config(None, [f"loss.terms={terms}"])
    loss = build_loss(cfg)
    logits = torch.randn(2, 1, 64, requires_grad=True)
    y = torch.randint(-1, 2, (2, 64))
    total, parts = loss(logits, y)
    total.backward()
    assert torch.isfinite(total) and logits.grad is not None and parts


def test_multiclass_losses():
    cfg = load_config(None, ["task.mode=multiclass", "task.class_names=[bg, a, b]",
                             "loss.terms=[{name: ce}, {name: focal}, {name: tmse, weight: 0.1}, {name: dice}]"])
    loss = build_loss(cfg)
    logits = torch.randn(2, 3, 32, requires_grad=True)
    y = torch.randint(-1, 3, (2, 32))
    total, _ = loss(logits, y)
    total.backward()
    assert torch.isfinite(total)


def test_tmse_zero_for_constant_prediction():
    cfg = load_config(None, ["loss.terms=[{name: tmse}]"])
    total, _ = build_loss(cfg)(torch.full((1, 1, 40), 2.0), torch.zeros(1, 40, dtype=torch.long))
    assert total.item() == 0.0


def test_loss_mode_mismatch_rejected():
    cfg = load_config(None, ["loss.terms=[{name: ce}]"])
    with pytest.raises(ValueError, match="multiclass"):
        build_loss(cfg)
