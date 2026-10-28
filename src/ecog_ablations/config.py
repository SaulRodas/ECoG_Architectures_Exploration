"""Carga y validación de configs/experiment.yaml (sin dependencias de torch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"

PREPROCESSINGS = ("P0", "P1", "P2", "P3", "P4", "P5")
# Valores implementados para A0–A5 (A5a y A5b solo los combinan). Los órdenes y modos que solo usaban R-FF/R-DTC
# quedaron fuera de alcance y se rechazan explícitamente.
NORMALIZATIONS = ("batch", "layer")
UPSAMPLES = ("interp", "convtranspose")
DECODER_ORDERS = ("upsample_conv_concat", "upsample_concat_conv")
CONVTRANSPOSE_MODES = ("depthwise",)
SKIP_POSITIONS = ("pre_pool",)
ARCH_KEYS = (
    "stem_channels", "encoder_channels", "kernel_sizes", "dilations", "decoder_channels",
    "decoder_kernel_sizes", "normalization", "dropout", "upsample", "decoder_order",
    "convtranspose_mode", "skip_position",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    cfg["_derived"] = derive(cfg)
    cfg["_path"] = str(path)
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Rutas relativas se interpretan desde la raíz del repositorio."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def derive(cfg: dict[str, Any]) -> dict[str, int]:
    data, prep = cfg["data"], cfg["preprocessing"]
    fs, fs_feat = data["fs_ecog"], prep["fs_features"]
    if fs % fs_feat:
        raise ValueError("fs_ecog debe ser múltiplo de fs_features")
    decim = fs // fs_feat
    n_blocks = len(cfg["folds"]["names"])
    block_samples = int(cfg["folds"]["block_seconds"] * fs)
    if n_blocks * block_samples != data["n_samples"]:
        raise ValueError("los bloques no cubren exactamente n_samples")
    block_columns = block_samples // decim
    delay = prep["delay_ms"] * fs_feat / 1000
    if delay != int(delay):
        raise ValueError(f"delay_ms={prep['delay_ms']} no es un número entero de columnas a {fs_feat} Hz")
    edge = prep["edge_columns"]
    valid_lo, valid_hi = edge, block_columns - edge
    window = prep["window"]
    if valid_hi - valid_lo - prep["purge_columns"] < window:
        raise ValueError("la región válida no admite ventanas tras la purga")
    if valid_hi + int(delay) > block_columns:
        raise ValueError("el retardo saca el objetivo fuera del bloque")
    return {
        "decim": decim,
        "n_blocks": n_blocks,
        "block_samples": block_samples,
        "block_columns": block_columns,
        "delay_columns": int(delay),
        "valid_lo": valid_lo,
        "valid_hi": valid_hi,
    }


def validate_architecture(name: str, spec: dict[str, Any], window: int | None = None) -> None:
    missing = [k for k in ARCH_KEYS if k not in spec]
    if missing:
        raise ValueError(f"{name}: faltan parámetros {missing}")
    n_levels = len(spec["encoder_channels"])
    for key in ("kernel_sizes", "dilations", "decoder_channels", "decoder_kernel_sizes"):
        if len(spec[key]) != n_levels:
            raise ValueError(f"{name}: {key} debe tener {n_levels} elementos")
    for k in list(spec["kernel_sizes"]) + list(spec["decoder_kernel_sizes"]):
        if k % 2 == 0:
            raise ValueError(f"{name}: los kernels deben ser impares (padding d·(k−1)//2)")
    choices = {
        "normalization": NORMALIZATIONS,
        "upsample": UPSAMPLES,
        "decoder_order": DECODER_ORDERS,
        "skip_position": SKIP_POSITIONS,
    }
    for key, allowed in choices.items():
        if spec[key] not in allowed:
            raise ValueError(f"{name}: {key}={spec[key]!r} no implementado; opciones: {allowed}")
    if spec["upsample"] == "convtranspose":
        if spec["convtranspose_mode"] not in CONVTRANSPOSE_MODES:
            raise ValueError(f"{name}: convtranspose_mode={spec['convtranspose_mode']!r} no implementado")
    elif spec["convtranspose_mode"] is not None:
        raise ValueError(f"{name}: convtranspose_mode solo aplica con upsample=convtranspose")
    if not 0.0 <= float(spec["dropout"]) < 1.0:
        raise ValueError(f"{name}: dropout fuera de rango")
    if window is not None and window % (2 ** n_levels):
        raise ValueError(f"{name}: la ventana {window} no es divisible entre 2^{n_levels}")


def validate_config(cfg: dict[str, Any]) -> None:
    subjects = cfg["data"]["subjects"]
    for name, spec in subjects.items():
        excluded = spec["excluded"]
        if len(set(excluded)) != len(excluded) or not all(0 <= c < spec["n_channels_raw"] for c in excluded):
            raise ValueError(f"{name}: canales excluidos inválidos")
        if spec["n_channels_raw"] - len(excluded) != spec["n_channels_used"]:
            raise ValueError(f"{name}: n_channels_used no coincide con las exclusiones")
    window = cfg["preprocessing"]["window"]
    for name, spec in cfg["architectures"].items():
        validate_architecture(name, spec, window)
    for name, var in cfg["variants"].items():
        if var["preprocessing"] not in PREPROCESSINGS:
            raise ValueError(f"{name}: preprocesamiento desconocido {var['preprocessing']}")
        if var["architecture"] not in cfg["architectures"]:
            raise ValueError(f"{name}: arquitectura desconocida {var['architecture']}")
        if var["control"] is not None and var["control"] not in cfg["variants"]:
            raise ValueError(f"{name}: control desconocido {var['control']}")
    for pair in cfg["equivalence_checks"]["pairs"]:
        if pair["preprocessing"] not in PREPROCESSINGS or pair["reference"] not in PREPROCESSINGS:
            raise ValueError(f"verificación de equivalencia inválida: {pair}")
    names = [c["name"] for c in cfg.get("contrasts", [])]
    if len(set(names)) != len(names):
        raise ValueError("contrastes con nombre repetido")
    for contrast in cfg.get("contrasts", []):
        unknown = [v for v in contrast["terms"] if v not in cfg["variants"]]
        if unknown:
            raise ValueError(f"contraste {contrast['name']}: variantes desconocidas {unknown}")
        if sum(contrast["terms"].values()) != 0:
            raise ValueError(f"contraste {contrast['name']}: los pesos deben sumar 0")


def seed_for(cfg: dict[str, Any], subject: str, fold: str) -> int:
    """Semilla compartida entre configuraciones: 42 + 5·índice_sujeto + índice_fold."""
    subject_index = list(cfg["data"]["subjects"]).index(subject)
    fold_index = cfg["folds"]["names"].index(fold)
    return int(cfg["training"]["seed_base"] + 5 * subject_index + fold_index)
