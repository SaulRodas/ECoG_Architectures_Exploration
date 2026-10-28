"""Orquestación de la campaña: plan de corridas, reanudación, artefactos y verificación de equivalencias."""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import __version__
from .cleaning import exclude_channels
from .config import REPO_ROOT, seed_for
from .data import glove_hold_check, load_subject
from .evaluation import trajectory_metrics
from .models import build_model, describe
from .preprocessing import (
    FoldContext,
    FoldFeatures,
    build_filter_bank,
    compute_fold_features,
    feature_equivalence,
    prepare_fold_context,
    resolve_device,
)
from .training import configure_torch, set_seed, train_and_validate

log = logging.getLogger("ecog_ablations")

# P2 y P4 solo se calculan para verificar su equivalencia con P1; las referencias van antes.
PREPROCESSING_ORDER = ("P0", "P1", "P2", "P4", "P3", "P5")
ARTIFACTS = ("status.json", "config.json", "architecture.json", "history.csv", "metrics.json", "predictions.npz")


# ----------------------------------------------------------------------------- utilidades


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"no serializable: {type(obj)}")


def write_json(path: Path, obj: Any) -> None:
    """Escritura atómica (el monitor nunca lee un JSON a medias)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_info() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "package_version": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_describe": _git("describe", "--tags", "--always", "--dirty"),
        "git_dirty": None if status is None else bool(status),
    }


def environment_info(device: torch.device) -> dict[str, Any]:
    import mne
    import scipy
    import sklearn

    cuda = device.type == "cuda"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "mne": mne.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _pid_alive(pid: int) -> bool:
    if os.name != "posix":
        return False  # en Windows no se comprueba (uso local, manual)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _raise_interrupt(signum, frame):  # SIGTERM → mismo tratamiento que Ctrl+C
    raise KeyboardInterrupt


# ----------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class RunKey:
    variant: str
    subject: str
    fold: str

    def path(self, results_dir: Path) -> Path:
        return Path(results_dir) / "runs" / self.variant / self.subject / self.fold

    @property
    def tag(self) -> str:
        return f"[{self.variant} {self.subject} {self.fold}]"


def planned_runs(cfg: dict[str, Any], subjects: list[str], folds: list[str], variants: list[str]) -> list[RunKey]:
    """Orden de campaña: fold → sujeto → grupo de preprocesamiento → variante."""
    order = [v for v in cfg["variants"] if v in variants]
    return [
        RunKey(v, s, f)
        for f in folds
        for s in subjects
        for p in PREPROCESSING_ORDER
        for v in order
        if cfg["variants"][v]["preprocessing"] == p
    ]


def run_state(run_dir: Path) -> str:
    """pending | completed | incomplete | running | interrupted | failed."""
    status = read_json(run_dir / "status.json")
    if status is None:
        return "pending"
    state = status.get("status", "pending")
    if state == "completed" and not all((run_dir / a).exists() for a in ARTIFACTS):
        return "incomplete"
    return state


# ----------------------------------------------------------------------------- informes por fold


def clip_table(ctx: FoldContext, block_samples: int) -> pd.DataFrame:
    """Recortes por canal (train / validación) con amplitudes antes y después del clip."""
    clip, tr, va = ctx.clip, ctx.train_blocks, [ctx.val_block]
    n_train = block_samples * len(tr)
    rows = {
        "subject": ctx.subject,
        "fold": ctx.fold,
        "channel": ctx.kept_channels,
        "median": clip["median"],
        "mad_scaled": clip["mad_scaled"],
        "mad_floored": clip["floored"],
        "lower": clip["lower"],
        "upper": clip["upper"],
        "clipped_train": ctx.clip_low[tr].sum(0) + ctx.clip_high[tr].sum(0),
        "clipped_val": ctx.clip_low[va].sum(0) + ctx.clip_high[va].sum(0),
    }
    rows["frac_train"] = rows["clipped_train"] / n_train
    rows["frac_val"] = rows["clipped_val"] / block_samples
    for split, blocks in (("train", tr), ("val", va)):
        raw_min, raw_max = ctx.raw_min[blocks].min(0), ctx.raw_max[blocks].max(0)
        rows[f"raw_min_{split}"], rows[f"raw_max_{split}"] = raw_min, raw_max
        rows[f"clean_min_{split}"] = np.maximum(raw_min, clip["lower"])
        rows[f"clean_max_{split}"] = np.minimum(raw_max, clip["upper"])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- campaña


class Campaign:
    def __init__(
        self,
        cfg: dict[str, Any],
        data_dir: str | Path,
        results_dir: str | Path,
        device: str | None = None,
        retry_failed: bool = False,
        epochs: int | None = None,
        max_batches: int | None = None,
        smoke: bool = False,
    ):
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.results = Path(results_dir)
        self.device = resolve_device(device or cfg["preprocessing"]["device"])
        self.retry_failed = retry_failed
        self.epochs = epochs
        self.max_batches = max_batches
        self.smoke = smoke
        self.fingers = cfg["data"]["fingers"]
        self.worker = f"{socket.gethostname()}-{os.getpid()}"
        self._subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._bank = None
        self._lock: Path | None = None
        self._env: dict[str, Any] | None = None

    # -- infraestructura

    @property
    def bank(self):
        if self._bank is None:
            self._bank = build_filter_bank(self.cfg)
        return self._bank

    def _acquire_lock(self) -> None:
        runners = self.results / "runners"
        runners.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname()
        for path in runners.glob(f"{host}-*.json"):
            pid = (read_json(path) or {}).get("pid")
            if pid and pid != os.getpid() and _pid_alive(pid):
                raise RuntimeError(f"ya hay un runner activo en este equipo (pid {pid}); deténlo antes de lanzar otro")
            path.unlink(missing_ok=True)
        self._lock = runners / f"{self.worker}.json"
        write_json(self._lock, {"pid": os.getpid(), "host": host, "started_at": now(), "heartbeat": now(),
                                "current": None, "argv": sys.argv, "device": str(self.device)})

    def _touch_lock(self, current: str | None) -> None:
        if self._lock is not None:
            info = read_json(self._lock) or {}
            info.update(heartbeat=now(), current=current)
            write_json(self._lock, info)

    def _release_lock(self) -> None:
        if self._lock is not None:
            self._lock.unlink(missing_ok=True)
            self._lock = None

    def _subject(self, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if subject not in self._subjects:
            raw = load_subject(self.data_dir, subject, self.cfg)
            ecog, kept = exclude_channels(raw["ecog"], self.cfg["data"]["subjects"][subject]["excluded"])
            self._subjects[subject] = (ecog, raw["dg"], kept)
        return self._subjects[subject]

    def _context(self, subject: str, fold: str) -> FoldContext:
        ecog, dg, kept = self._subject(subject)
        ctx = prepare_fold_context(ecog, dg, kept, subject, fold, self.cfg)
        quality = self.results / "quality"
        (quality / "clip").mkdir(parents=True, exist_ok=True)
        clip_table(ctx, self.cfg["_derived"]["block_samples"]).to_csv(quality / "clip" / f"{subject}_{fold}.csv", index=False)
        return ctx

    def _save_feature_report(self, ctx: FoldContext, feats: FoldFeatures, seconds: float) -> None:
        quality = self.results / "quality"
        stem = f"{ctx.subject}_{ctx.fold}_{feats.preprocessing}"
        write_json(quality / "features" / f"{stem}.json", {
            "subject": ctx.subject, "fold": ctx.fold, "preprocessing": feats.preprocessing,
            "n_channels": feats.n_channels, "n_freqs": feats.n_freqs, "compute_seconds": seconds,
            "clip_fraction_train": ctx.clipped_fraction("train"), "clip_fraction_val": ctx.clipped_fraction("val"),
            **feats.stats,
        })
        (quality / "input_params").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            quality / "input_params" / f"{stem}.npz",
            kept_channels=ctx.kept_channels, **{f"clip_{k}": v for k, v in ctx.clip.items()}, **feats.params,
        )

    # -- ejecución

    def needs_run(self, key: RunKey) -> bool:
        state = run_state(key.path(self.results))
        if state == "completed":
            return False
        if state == "failed":
            return self.retry_failed
        return True  # pending, incomplete, running (huérfana) o interrupted

    def run(self, subjects: list[str], folds: list[str], variants: list[str]) -> None:
        configure_torch(self.cfg["runtime"])
        self._acquire_lock()
        previous = signal.signal(signal.SIGTERM, _raise_interrupt)
        try:
            plan = planned_runs(self.cfg, subjects, folds, variants)
            pending = sum(self.needs_run(k) for k in plan)
            log.info("campaña: %d corridas planificadas, %d pendientes; dispositivo %s", len(plan), pending, self.device)
            for fold in folds:
                for subject in subjects:
                    keys = [k for k in plan if k.fold == fold and k.subject == subject]
                    self._run_unit(subject, fold, keys)
            log.info("campaña terminada")
        finally:
            signal.signal(signal.SIGTERM, previous)
            self._release_lock()

    def _run_unit(self, subject: str, fold: str, keys: list[RunKey]) -> None:
        """Un sujeto/fold: cada conjunto de características se calcula una vez y lo reutilizan sus variantes."""
        pending = [k for k in keys if self.needs_run(k)]
        pairs = self.cfg["equivalence_checks"]["pairs"]
        eq_path = self.results / "quality" / "equivalence" / f"{subject}_{fold}.json"
        eq_pending = not eq_path.exists()
        if not pending and not eq_pending:
            log.info("[%s %s] nada pendiente", subject, fold)
            return
        variants = self.cfg["variants"]
        needed = {variants[k.variant]["preprocessing"] for k in pending}
        if eq_pending:
            needed |= {p for pair in pairs for p in (pair["preprocessing"], pair["reference"])}
        references = {pair["reference"] for pair in pairs} if eq_pending else set()

        ctx = self._context(subject, fold)
        cache: dict[str, FoldFeatures] = {}
        records = []
        for p in (p for p in PREPROCESSING_ORDER if p in needed):
            t0 = time.perf_counter()
            feats = compute_fold_features(ctx, p, self.bank, self.cfg, self.device)
            seconds = time.perf_counter() - t0
            self._save_feature_report(ctx, feats, seconds)
            log.info("[%s %s] características %s: %.1f s, %d ventanas de train, finitas=%s",
                     subject, fold, p, seconds, feats.stats["n_train_windows"], feats.stats["finite"])
            group = [k for k in pending if variants[k.variant]["preprocessing"] == p]
            for key in group:
                if feats.stats["finite"]:
                    self._train(key, ctx, feats)
                else:
                    self._mark_failed(key, "características no finitas")
            if eq_pending:
                for pair in (q for q in pairs if q["preprocessing"] == p):
                    result = feature_equivalence(feats, cache[pair["reference"]])
                    result["equivalent"] = bool(result["rel_max_diff"] <= self.cfg["equivalence_checks"]["rel_tol"])
                    records.append({"subject": subject, "fold": fold, **pair, **result})
                    log.info("[%s %s] equivalencia %s vs %s: dif. relativa máx. %.2e, r=%.9f",
                             subject, fold, p, pair["reference"], result["rel_max_diff"], result["corr"])
                if p in references:
                    cache[p] = feats
            done = PREPROCESSING_ORDER[: PREPROCESSING_ORDER.index(p) + 1]
            for ref in list(cache):
                if all(q["preprocessing"] in done for q in pairs if q["reference"] == ref):
                    del cache[ref]
            del feats
            self._free()
        if eq_pending:
            write_json(eq_path, {"rel_tol": self.cfg["equivalence_checks"]["rel_tol"], "checked_at": now(),
                                 "records": records})

    def _free(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _start_attempt(self, run_dir: Path) -> int:
        """Archiva un intento previo no completado en attempts/ y devuelve el número del intento nuevo."""
        attempts = run_dir / "attempts"
        n_previous = len(list(attempts.iterdir())) if attempts.exists() else 0
        files = [p for p in run_dir.iterdir() if p.name != "attempts"] if run_dir.exists() else []
        if files:
            state = (read_json(run_dir / "status.json") or {}).get("status", "unknown")
            target = attempts / f"{n_previous + 1:02d}_{state}_{datetime.now():%Y%m%d-%H%M%S}"
            target.mkdir(parents=True)
            for path in files:
                shutil.move(str(path), str(target / path.name))
            n_previous += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        return n_previous + 1

    def _mark_failed(self, key: RunKey, message: str) -> None:
        run_dir = key.path(self.results)
        attempt = self._start_attempt(run_dir)
        write_json(run_dir / "status.json", {"status": "failed", "variant": key.variant, "subject": key.subject,
                                             "fold": key.fold, "attempt": attempt, "error": message, "finished_at": now()})
        log.error("%s falló: %s", key.tag, message)

    def _train(self, key: RunKey, ctx: FoldContext, feats: FoldFeatures) -> None:
        cfg, variant = self.cfg, self.cfg["variants"][key.variant]
        run_dir = key.path(self.results)
        attempt = self._start_attempt(run_dir)
        seed = seed_for(cfg, key.subject, key.fold)
        epochs = self.epochs or cfg["training"]["epochs"]
        status = {
            "status": "running", "variant": key.variant, "subject": key.subject, "fold": key.fold,
            "preprocessing": variant["preprocessing"], "architecture": variant["architecture"], "seed": seed,
            "attempt": attempt, "worker": self.worker, "started_at": now(), "heartbeat": now(),
            "epoch": 0, "epochs": epochs,
        }
        write_json(run_dir / "status.json", status)
        self._touch_lock(key.tag)
        if self._env is None:
            self._env = environment_info(self.device)
        model = None
        t_start = time.perf_counter()
        try:
            write_json(run_dir / "config.json", {
                "run": {k: status[k] for k in ("variant", "subject", "fold", "preprocessing", "architecture", "seed",
                                               "attempt", "worker", "started_at")},
                "overrides": {"epochs": self.epochs, "max_batches": self.max_batches, "smoke": self.smoke},
                "environment": self._env,
                "code": code_info(),
                "config": {k: v for k, v in cfg.items()},
            })
            set_seed(seed)
            model = build_model(variant["architecture"], cfg, feats.X.shape[1]).to(self.device)
            arch = describe(model)
            write_json(run_dir / "architecture.json", arch)
            log.info("%s %s + %s, %d parámetros, %d ventanas de train, semilla %d",
                     key.tag, variant["preprocessing"], variant["architecture"], arch["parameters"]["total"],
                     len(feats.train_index), seed)

            history_path = run_dir / "history.csv"

            def on_epoch(row: dict[str, Any]) -> None:
                new = not history_path.exists()
                with open(history_path, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(row))
                    if new:
                        writer.writeheader()
                    writer.writerow(row)
                status.update(epoch=row["epoch"], heartbeat=now(), last_val_pearson=row["val_pearson_macro"],
                              last_epoch_seconds=row["epoch_seconds"])
                write_json(run_dir / "status.json", status)
                self._touch_lock(key.tag)
                log.info("%s época %d/%d  train %.4f  val %.4f  r_val %.4f  (%.1f s)", key.tag, row["epoch"], epochs,
                         row["train_loss"], row["val_loss"], row["val_pearson_macro"], row["epoch_seconds"])

            result = train_and_validate(model, feats, cfg, self.device, seed, self.fingers, on_epoch,
                                        epochs=self.epochs, max_batches=self.max_batches)
            cols = feats.val_target_columns()
            y_true = ctx.glove[feats.val_block][:, cols]
            y_pred = feats.inverse_y(result.y_pred_scaled)
            metrics = trajectory_metrics(y_true, y_pred, self.fingers)
            if result.diverged:
                metrics["valid"] = False
                metrics["pearson_macro"] = None
                metrics["invalid_reasons"]["entrenamiento"] = result.divergence_reason
            fs_feat = cfg["preprocessing"]["fs_features"]
            t_seconds = (feats.val_block * cfg["_derived"]["block_columns"] + np.arange(cols.start, cols.stop)) / fs_feat
            np.savez_compressed(run_dir / "predictions.npz", y_true=y_true.astype(np.float32),
                                y_pred=y_pred.astype(np.float32), t_seconds=t_seconds, fingers=np.array(self.fingers))
            if cfg["runtime"]["save_model_weights"]:
                torch.save(model.state_dict(), run_dir / "model_last.pt")
            metrics.update({
                "variant": key.variant, "subject": key.subject, "fold": key.fold, "seed": seed,
                "preprocessing": variant["preprocessing"], "architecture": variant["architecture"],
                "n_parameters": arch["parameters"]["total"], "epochs_completed": result.epochs_completed,
                "diverged": result.diverged, "train_seconds": result.train_seconds,
                "n_train_windows": int(len(feats.train_index)), "n_val_columns": int(y_true.shape[1]),
                "clip_fraction_train": ctx.clipped_fraction("train"), "clip_fraction_val": ctx.clipped_fraction("val"),
                "smoke": self.smoke,
            })
            write_json(run_dir / "metrics.json", metrics)
            status.update(status="completed", finished_at=now(), duration_seconds=time.perf_counter() - t_start,
                          heartbeat=now())
            write_json(run_dir / "status.json", status)
            r = metrics["pearson_macro"]
            log.info("%s completada: Pearson macro %s (%.0f s)", key.tag,
                     "inválido" if r is None else f"{r:.4f}", time.perf_counter() - t_start)
        except KeyboardInterrupt:
            status.update(status="interrupted", finished_at=now(), heartbeat=now())
            write_json(run_dir / "status.json", status)
            log.warning("%s interrumpida", key.tag)
            raise
        except Exception as exc:  # se conserva para diagnóstico y la campaña sigue
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            status.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
            write_json(run_dir / "status.json", status)
            log.exception("%s falló", key.tag)
        finally:
            del model
            self._free()


# ----------------------------------------------------------------------------- control de calidad


def run_quality(cfg: dict[str, Any], data_dir: str | Path, results_dir: str | Path, subjects: list[str]) -> dict[str, Any]:
    """Revisión previa al entrenamiento: canales, fase del guante, recortes por fold, finitud y figuras."""
    from . import reporting

    quality = Path(results_dir) / "quality"
    figures = quality / "figures"
    (quality / "clip").mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reporting.plot_filter_responses(build_filter_bank(cfg), cfg, figures / "filter_responses.png")

    channels, holds, finite, tables = [], [], [], []
    for subject in subjects:
        spec = cfg["data"]["subjects"][subject]
        raw = load_subject(data_dir, subject, cfg)
        holds.append({"subject": subject, **glove_hold_check(raw["dg"], cfg["data"]["glove_hold_rows"])})
        ecog, kept = exclude_channels(raw["ecog"], spec["excluded"])
        channels.append({"subject": subject, "n_channels_raw": spec["n_channels_raw"],
                         "excluded": " ".join(map(str, spec["excluded"])), "n_channels_used": ecog.shape[1]})
        subject_tables = []
        for fold in cfg["folds"]["names"]:
            ctx = prepare_fold_context(ecog, raw["dg"], kept, subject, fold, cfg)
            table = clip_table(ctx, cfg["_derived"]["block_samples"])
            table.to_csv(quality / "clip" / f"{subject}_{fold}.csv", index=False)
            subject_tables.append(table)
            finite.append({"subject": subject, "fold": fold,
                           "clean_finite": bool(all(np.isfinite(b).all() for b in ctx.blocks)),
                           "glove_finite": bool(np.isfinite(ctx.glove).all()),
                           "clip_fraction_train": ctx.clipped_fraction("train"),
                           "clip_fraction_val": ctx.clipped_fraction("val"),
                           "mad_floored_channels": int(ctx.clip["floored"].sum())})
            if fold == cfg["folds"]["names"][0]:
                reporting.plot_raw_vs_clean(ecog, ctx, cfg, figures / f"{subject}_raw_vs_clean.png")
        tables.extend(subject_tables)
        reporting.plot_clip_fractions(pd.concat(subject_tables), figures / f"{subject}_clip_fraction.png")
        log.info("[%s] control de calidad listo: %d canales usados", subject, ecog.shape[1])

    pd.DataFrame(channels).to_csv(quality / "channels.csv", index=False)
    pd.DataFrame(holds).to_csv(quality / "glove_hold.csv", index=False)
    pd.DataFrame(finite).to_csv(quality / "finite_and_clipping.csv", index=False)
    pd.concat(tables).to_csv(quality / "clip_all.csv", index=False)
    summary = {"channels": channels, "glove_hold": holds, "folds": finite, "created_at": now()}
    write_json(quality / "quality_summary.json", summary)
    return summary
