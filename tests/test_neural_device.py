"""Device switch for the torch families: default CPU; MORTCAL_DEVICE=cuda
opt-in must produce the same point forecast as CPU up to float32 noise (it
is the same seeded init and the same full-batch Adam trajectory), and
requesting cuda on a machine without it must raise, never fall back."""
import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mortcal.models import neural as neural_mod  # noqa: E402

N_AGES, T = 20, 40


def _panel(rng):
    ages = np.arange(N_AGES)
    alpha = -7.0 + 4.0 * ages / N_AGES
    k = np.cumsum(-1.0 + rng.normal(0, 0.5, T))
    mx = np.exp(alpha[:, None] + np.outer(np.full(N_AGES, 1 / N_AGES), k))
    E = np.full((N_AGES, T), 1e5)
    return rng.poisson(E * mx).astype(float), E


def _fit_point(device, monkeypatch):
    monkeypatch.setenv("MORTCAL_DEVICE", device)
    rng = np.random.default_rng(5)
    D, E = _panel(rng)
    m = neural_mod.NeuralLC(lr_grid=(1e-2,), epochs_grid=(50,)).fit(D, E)
    assert str(m.device).startswith(device)
    return m._point_logmx(3)


def test_default_device_is_cpu(monkeypatch):
    monkeypatch.delenv("MORTCAL_DEVICE", raising=False)
    assert neural_mod._device().type == "cpu"


def test_cuda_request_without_cuda_raises(monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("cuda present; the raise path is for cuda-less machines")
    monkeypatch.setenv("MORTCAL_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="MORTCAL_DEVICE=cuda"):
        neural_mod._device()


def _initial_loss(device, monkeypatch):
    """Loss at the seeded init BEFORE any optimizer step: identical data and
    identical parameters on both devices must give the same number up to
    float32 rounding. This isolates a device-path BUG (wrong tensor, wrong
    mask, wrong weights) from optimizer-trajectory divergence."""
    monkeypatch.setenv("MORTCAL_DEVICE", device)
    rng = np.random.default_rng(5)
    D, E = _panel(rng)
    m = neural_mod.NeuralLC(lr_grid=(1e-2,), epochs_grid=(50,))
    m.n_ages, m.T = D.shape
    m._subset_cache = {}
    m.device = neural_mod._device()
    m._prepare(D, E)
    for k, v in list(vars(m).items()):        # mirrors _TorchFamily.fit()
        if isinstance(v, torch.Tensor):
            setattr(m, k, v.to(m.device))
    torch.manual_seed(m.seed)
    net = m._build().to(m.device)
    net.eval()                      # dropout OFF: masks are device-RNG-specific
    with torch.no_grad():
        return float(m._loss(net, set()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_gpu_reproduces_cpu_forecast(monkeypatch):
    # (1) structural, dropout off: identical weights (verified 0.0 diff) and
    # identical inputs must give the same loss up to float32 reduction order.
    # A wrong tensor / mask / weight placement shows up here as an O(1) gap.
    l_cpu = _initial_loss("cpu", monkeypatch)
    l_gpu = _initial_loss("cuda", monkeypatch)
    assert abs(l_cpu - l_gpu) < 1e-5 * max(1.0, abs(l_cpu)), (l_cpu, l_gpu)
    # (2) trajectory, dropout ON during training: torch.manual_seed seeds both
    # generators but the CPU and CUDA dropout kernels draw DIFFERENT masks, so
    # the two runs are two valid realizations of the same seeded procedure,
    # not bit-identical. Measured 2026-08-27 (RTX 3050 Ti, TF32 off): first
    # forward pass in train mode differs 4.9% max relative (5% dropout on 64
    # units), final 3-step forecast 1.0% relative, CPU-vs-CPU exactly 0.0.
    # Consequence for the study: a sweep must not mix devices within a regime.
    cpu = _fit_point("cpu", monkeypatch)
    gpu = _fit_point("cuda", monkeypatch)
    assert cpu.shape == gpu.shape == (3, N_AGES)
    assert np.all(np.isfinite(gpu))
    rel = np.abs(cpu - gpu).max() / np.abs(cpu).mean()
    assert rel < 0.03, f"GPU/CPU relative forecast gap {rel:.4f}"
