"""Comprobaciones mínimas de implementación (no son fases experimentales)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ecog_ablations.cleaning import apply_clip, exclude_channels, fit_clip
from ecog_ablations.config import REPO_ROOT, load_config
from ecog_ablations.data import glove_hold_check, glove_to_feature_rate, load_subject, split_blocks, subject_path
from ecog_ablations.evaluation import overlap_add, pearson_per_channel, trajectory_metrics
from ecog_ablations.models import build_model, count_parameters
from ecog_ablations.preprocessing import (
    apply_input_transform,
    build_filter_bank,
    compute_fold_features,
    feature_equivalence,
    fft_conv_same,
    morlet_power_full,
    prepare_fold_context,
)

CPU = torch.device("cpu")
DATA_DIR = Path(os.environ.get("ECOG_DATA_DIR", REPO_ROOT / "data" / "raw"))


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def bank(cfg):
    return build_filter_bank(cfg)


def synthetic_recording(cfg, n_channels=4, seed=0):
    """ECoG sintética con offsets/escala por canal, línea de 60 Hz y gamma modulada por un dedo; guante ZOH."""
    rng = np.random.default_rng(seed)
    n, fs, hold = cfg["data"]["n_samples"], cfg["data"]["fs_ecog"], cfg["data"]["glove_hold_rows"]
    t25 = np.arange(n // hold) * hold / fs
    dg25 = np.stack([np.sin(2 * np.pi * (0.2 + 0.05 * k) * t25 + rng.uniform(0, 2 * np.pi)) for k in range(5)], 1)
    dg = np.repeat(dg25, hold, axis=0)
    t = np.arange(n) / fs
    envelope = 1 + 0.5 * dg[:, 0]
    ecog = rng.standard_normal((n, n_channels)) * rng.uniform(5, 50, n_channels) + rng.uniform(-500, 500, n_channels)
    ecog += 20 * np.sin(2 * np.pi * 60 * t)[:, None]
    ecog += (envelope * np.sin(2 * np.pi * 110 * t))[:, None] * rng.uniform(5, 15, n_channels)
    return ecog, dg


@pytest.fixture(scope="module")
def synthetic(cfg):
    ecog, dg = synthetic_recording(cfg)
    kept = np.arange(ecog.shape[1])
    return ecog, dg, kept, prepare_fold_context(ecog, dg, kept, "S1", "B3", cfg)


# --------------------------------------------------------------------------- datos

@pytest.mark.skipif(not all(subject_path(DATA_DIR, s, load_config()).exists() for s in ("S1", "S2", "S3")),
                    reason="no están los .mat en data/raw (o ECOG_DATA_DIR)")
def test_data_subjects(cfg):
    for subject, expected in (("S1", 61), ("S2", 46), ("S3", 63)):
        raw = load_subject(DATA_DIR, subject, cfg)
        ecog, kept = exclude_channels(raw["ecog"], cfg["data"]["subjects"][subject]["excluded"])
        assert ecog.shape == (400000, expected)
        assert not set(cfg["data"]["subjects"][subject]["excluded"]) & set(kept)
        blocks = split_blocks(ecog, 5)
        assert len(blocks) == 5 and all(b.shape[0] == 80000 for b in blocks)
        assert raw["dg"].shape[1] == 5
        assert glove_hold_check(raw["dg"], 40)["aligned_fraction"] == 1.0
        assert glove_to_feature_rate(split_blocks(raw["dg"], 5)[0], cfg).shape == (5, 8000)


# --------------------------------------------------------------------------- limpieza

def test_clip_and_excluded_channel(cfg):
    rng = np.random.default_rng(1)
    block = rng.standard_normal((5000, 3))
    spiky = block.copy()
    spiky[100, 1] = 1e6
    params = fit_clip([block], 25.0, 1.4826, 1e-6)
    clean, n_low, n_high = apply_clip(spiky, params)
    assert clean[100, 1] == pytest.approx(params["upper"][1])
    assert n_high[1] == 1 and n_low.sum() == 0 and n_high[[0, 2]].sum() == 0

    # Un canal excluido (con valores absurdos) no participa en el CMR ni en la entrada del modelo.
    with_bad = np.column_stack([block[:, :2], 1e9 * rng.standard_normal(5000), block[:, 2]])
    kept_signal, kept = exclude_channels(with_bad, [2])
    assert kept.tolist() == [0, 1, 3]
    np.testing.assert_array_equal(apply_input_transform("P0", kept_signal, {}), apply_input_transform("P0", block, {}))


# --------------------------------------------------------------------------- sin contaminación y ventanas

def test_validation_changes_do_not_leak(cfg, bank, synthetic):
    ecog, dg, kept, ctx = synthetic
    feats = compute_fold_features(ctx, "P3", bank, cfg, CPU)

    altered = ecog.copy()
    altered[160000:240000] = altered[160000:240000] * 100 + 1e4  # bloque validado de B3
    altered[200000, :] = 1e8
    ctx2 = prepare_fold_context(altered, dg, kept, "S1", "B3", cfg)
    feats2 = compute_fold_features(ctx2, "P3", bank, cfg, CPU)

    for key in ctx.clip:
        np.testing.assert_array_equal(ctx.clip[key], ctx2.clip[key])
    for key in feats.params:
        np.testing.assert_array_equal(feats.params[key], feats2.params[key])
    np.testing.assert_array_equal(feats.train_index, feats2.train_index)
    for b in ctx.train_blocks:
        np.testing.assert_array_equal(feats.X[b], feats2.X[b])
    assert not np.array_equal(feats.X[ctx.val_block], feats2.X[ctx.val_block])

    # Ventanas: nunca en el bloque validado, dentro de la región válida y con purga junto a él.
    w = cfg["preprocessing"]["window"]
    assert ctx.val_block == 2 and 2 not in set(feats.train_index[:, 0])
    assert ctx.train_ranges[1] == (125, 7875 - 256) and ctx.train_ranges[3] == (125 + 256, 7875)
    assert ctx.train_ranges[0] == (125, 7875) and ctx.train_ranges[4] == (125, 7875)
    for b, (lo, hi) in ctx.train_ranges.items():
        s = feats.train_index[feats.train_index[:, 0] == b, 1]
        assert s.min() == lo and s.max() + w == hi
    assert feats.delay == 2
    assert feats.val_starts[0] == 125 and feats.val_starts[-1] == 7875 - w
    assert np.all(np.diff(feats.val_starts)[:-1] == 64)


def test_p2_p4_equivalent_to_p1(cfg, bank, synthetic):
    *_, ctx = synthetic
    feats = {p: compute_fold_features(ctx, p, bank, cfg, CPU) for p in ("P0", "P1", "P2", "P4")}
    tol = cfg["equivalence_checks"]["rel_tol"]
    for p in ("P2", "P4"):
        assert feature_equivalence(feats[p], feats["P1"])["rel_max_diff"] <= tol
    assert feature_equivalence(feats["P0"], feats["P1"])["rel_max_diff"] > 100 * tol  # el CMR sí cambia la entrada


# --------------------------------------------------------------------------- DSP frente a MNE

def test_dsp_matches_mne(cfg, bank):
    import mne
    from scipy.signal import fftconvolve

    assert (bank.bandpass.size, bank.notch.size, bank.antialias.size) == (331, 1651, 243)
    assert -(-bank.edge_samples // bank.decim) == 125

    rng = np.random.default_rng(2)
    x = rng.standard_normal((3000, 2))
    ours = morlet_power_full(x, bank, CPU)
    ref = mne.time_frequency.tfr_array_morlet(x.T[None], sfreq=1000, freqs=bank.freqs, n_cycles=7.0,
                                              zero_mean=True, output="power")[0]
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-9 * ref.max())

    xt = torch.as_tensor(x.T)
    np.testing.assert_allclose(fft_conv_same(xt, torch.as_tensor(bank.notch)).numpy(),
                               fftconvolve(x.T, bank.notch[None], mode="same", axes=-1), atol=1e-10)


# --------------------------------------------------------------------------- arquitecturas

EXPECTED_PARAMS_S3 = {"A0": 8.52e6, "A1": 8.52e6, "A2": 8.52e6, "A3": 8.53e6, "A4": 9.36e6, "A5": 9.37e6}


@pytest.mark.parametrize("arch", ["A0", "A1", "A2", "A3", "A4", "A5"])
def test_architectures(cfg, arch):
    spec = cfg["architectures"][arch]
    for n_channels in (61, 46, 63):  # S1, S2, S3
        torch.manual_seed(0)
        model = build_model(arch, cfg, n_channels * 40)
        model.trace_enabled = True
        out = model(torch.randn(2, n_channels, 40, 256))
        assert out.shape == (2, 5, 256)
        out.square().mean().backward()
        assert torch.isfinite(out).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

        ops = [(s, op) for s, op, *_ in model.last_trace if s.startswith("dec")]
        per_level = ["upsample", "conv", "concat"] if spec["decoder_order"] == "upsample_conv_concat" \
            else ["upsample", "concat", "conv"]
        assert [op for _, op in ops] == per_level * 5
        lengths = [n for s, op, c, n in model.last_trace if op == "upsample"]
        assert lengths == [16, 32, 64, 128, 256]

        dims = model.expected["decoder"]
        if spec["decoder_order"] == "upsample_conv_concat":
            assert [d["out"] for d in dims] == [1024, 1024, 512, 256, 128]
            assert model.expected["head_in"] == 128
        else:
            assert [d["conv_in"] for d in dims] == [1024, 1024, 768, 384, 192]
            assert [d["conv_out"] for d in dims] == [512, 512, 256, 128, 64]
            assert model.expected["head_in"] == 64
        if n_channels == 63:
            assert count_parameters(model) == pytest.approx(EXPECTED_PARAMS_S3[arch], rel=5e-3)


# --------------------------------------------------------------------------- métricas

def test_metrics():
    rng = np.random.default_rng(3)
    y = rng.standard_normal((5, 500))
    r, reasons = pearson_per_channel(y, y)
    np.testing.assert_allclose(r, 1.0)
    assert all(reason is None for reason in reasons)

    constant = y.copy()
    constant[2] = 0.7
    metrics = trajectory_metrics(y, constant, ["thumb", "index", "middle", "ring", "little"])
    assert not metrics["valid"] and metrics["pearson_macro"] is None
    assert metrics["invalid_reasons"] == {"middle": "predicción constante"}

    starts = np.array([0, 100, 244])
    windows = np.stack([y[:, s:s + 256] for s in starts])
    np.testing.assert_allclose(overlap_add(windows, starts, 0, 500), y)
