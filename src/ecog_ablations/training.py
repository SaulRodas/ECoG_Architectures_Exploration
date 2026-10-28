"""Entrenamiento común: Adam con L2 acoplada, pérdida 0.5·MSE + 0.5·(1 − coseno), semillas y determinismo."""

from __future__ import annotations

import inspect
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .evaluation import overlap_add, pearson_per_channel
from .preprocessing import FoldFeatures


def configure_torch(runtime: dict[str, Any]) -> None:
    """TF32 según la config; determinismo «cuando sea viable» (avisos en lugar de errores)."""
    allow = bool(runtime["allow_tf32"])
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
    if runtime["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model: torch.nn.Module, tcfg: dict[str, Any]) -> torch.optim.Optimizer:
    kwargs = dict(lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], betas=tuple(tcfg["betas"]), eps=tcfg["eps"])
    if "decoupled_weight_decay" in inspect.signature(torch.optim.Adam.__init__).parameters:
        kwargs["decoupled_weight_decay"] = False  # L2 acoplada, no AdamW
    return torch.optim.Adam(model.parameters(), **kwargs)


def mse_cosine_loss(pred: torch.Tensor, target: torch.Tensor, lcfg: dict[str, Any]):
    """Coseno a lo largo de las posiciones temporales por dedo y ventana, promediado sobre dedos y batch."""
    mse = F.mse_loss(pred, target)
    cos = F.cosine_similarity(pred, target, dim=-1, eps=lcfg["cosine_eps"]).mean()
    loss = lcfg["mse_weight"] * mse + lcfg["cosine_weight"] * (1.0 - cos)
    return loss, mse, cos


@dataclass
class TrainResult:
    history: list[dict[str, Any]]
    y_pred_scaled: np.ndarray        # (dedos, columnas validadas) reconstruido, escala MinMax
    epochs_completed: int
    diverged: bool
    divergence_reason: str | None
    train_seconds: float


class _Windows:
    """Ventanas extraídas por indexado de vistas (sin copiar el tensor completo por ventana)."""

    def __init__(self, feats: FoldFeatures, device: torch.device):
        self.X = torch.from_numpy(feats.X).to(device).unfold(2, feats.window, 1)  # (bloques, F, T−W+1, W)
        self.Y = torch.from_numpy(feats.Y).to(device).unfold(2, feats.window, 1)
        self.delay = feats.delay

    def get(self, blocks: torch.Tensor, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[blocks, :, starts], self.Y[blocks, :, starts + self.delay]


@torch.no_grad()
def _validate(model, windows: _Windows, feats: FoldFeatures, device, lcfg, batch_size: int):
    model.eval()
    starts = torch.as_tensor(feats.val_starts, device=device)
    blocks = torch.full_like(starts, feats.val_block)
    preds, sums, n = [], torch.zeros(3, device=device, dtype=torch.float64), 0
    for i in range(0, len(starts), batch_size):
        xb, yb = windows.get(blocks[i:i + batch_size], starts[i:i + batch_size])
        pred = model(xb)
        loss, mse, cos = mse_cosine_loss(pred, yb, lcfg)
        sums += torch.stack([loss, mse, cos]).double() * len(xb)
        n += len(xb)
        preds.append(pred.float().cpu().numpy())
    model.train()
    y_rec = overlap_add(np.concatenate(preds), feats.val_starts, feats.valid_lo, feats.valid_hi)
    return (sums / n).tolist(), y_rec


def train_and_validate(
    model: torch.nn.Module,
    feats: FoldFeatures,
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    fingers: list[str],
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    epochs: int | None = None,
    max_batches: int | None = None,
) -> TrainResult:
    """50 épocas sin early stopping ni scheduler; se evalúa la última época."""
    tcfg, lcfg = cfg["training"], cfg["training"]["loss"]
    epochs = epochs or tcfg["epochs"]
    batch = tcfg["batch_size"]
    windows = _Windows(feats, device)
    train_index = torch.from_numpy(feats.train_index).to(device)
    y_val = feats.Y[feats.val_block][:, feats.val_target_columns()]
    n_train = len(train_index)
    n_batches = n_train // batch if tcfg["drop_last"] else -(-n_train // batch)
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    optimizer = make_optimizer(model, tcfg)
    order = torch.Generator().manual_seed(seed)  # mismo orden de batches para todas las variantes
    history, y_rec = [], None
    diverged, reason = False, None
    t_start = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, generator=order).to(device)
        sums = torch.zeros(3, device=device, dtype=torch.float64)
        for i in range(n_batches):
            sel = train_index[perm[i * batch:(i + 1) * batch]]
            xb, yb = windows.get(sel[:, 0], sel[:, 1])
            loss, mse, cos = mse_cosine_loss(model(xb), yb, lcfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += torch.stack([loss.detach(), mse.detach(), cos.detach()]).double()
        train_loss, train_mse, train_cos = (sums / max(n_batches, 1)).tolist()
        t1 = time.perf_counter()
        (val_loss, val_mse, val_cos), y_rec = _validate(model, windows, feats, device, lcfg, batch)
        r, _ = pearson_per_channel(y_val, y_rec)
        t2 = time.perf_counter()
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_mse": train_mse, "train_cos": train_cos,
            "val_loss": val_loss, "val_mse": val_mse, "val_cos": val_cos,
            "val_pearson_macro": float(np.mean(r)),
            **{f"val_pearson_{f}": float(v) for f, v in zip(fingers, r)},
            "n_batches": n_batches, "train_seconds": t1 - t0, "val_seconds": t2 - t1, "epoch_seconds": t2 - t0,
        }
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        if not np.isfinite(train_loss):
            diverged, reason = True, f"pérdida de entrenamiento no finita en la época {epoch}"
            break
    return TrainResult(history, y_rec, len(history), diverged, reason, time.perf_counter() - t_start)
