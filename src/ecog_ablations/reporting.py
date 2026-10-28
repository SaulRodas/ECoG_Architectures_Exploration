"""Tablas, figuras y monitoreo de la campaña.

Las figuras son PNG estáticos para el artículo: una sola escala por eje, marcas finas, rejilla recesiva y
color asignado por entidad en orden fijo (sujetos: ranuras 1–3 de la paleta validada; control en tinta).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .runner import planned_runs, read_json, run_state

# --- paleta (instancia de referencia validada; modo claro para PNG)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SUBJECT_COLORS = {"S1": SLOTS[0], "S2": SLOTS[1], "S3": SLOTS[2]}
# Color fijo por variante en todas las figuras; el control va en tinta (énfasis neutro).
VARIANT_COLORS = {"P0": INK, "P1": SLOTS[0], "P3": SLOTS[1], "P5": SLOTS[2], "A1": SLOTS[3],
                  "A2": SLOTS[4], "A3": SLOTS[5], "A4": SLOTS[6], "A5": SLOTS[7]}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "font.family": "sans-serif", "lines.linewidth": 2.0,
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================ recolección


def collect_runs(results_dir: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Una fila por corrida existente (estado + métricas aplanadas)."""
    rows = []
    fingers = cfg["data"]["fingers"]
    for status_path in sorted(Path(results_dir, "runs").glob("*/*/*/status.json")):
        run_dir = status_path.parent
        status = read_json(status_path) or {}
        metrics = read_json(run_dir / "metrics.json") or {}
        variant = status.get("variant", run_dir.parts[-3])
        row = {
            "variant": variant, "subject": status.get("subject", run_dir.parts[-2]),
            "fold": status.get("fold", run_dir.parts[-1]), "state": run_state(run_dir),
            "label": cfg["variants"].get(variant, {}).get("label", variant),
            "family": cfg["variants"].get(variant, {}).get("family"),
            "preprocessing": status.get("preprocessing"), "architecture": status.get("architecture"),
            "seed": status.get("seed"), "attempt": status.get("attempt"), "epoch": status.get("epoch"),
            "epochs": status.get("epochs"), "started_at": status.get("started_at"),
            "finished_at": status.get("finished_at"), "duration_seconds": status.get("duration_seconds"),
            "error": status.get("error"),
        }
        if metrics:
            row.update(valid=metrics.get("valid"), pearson_macro=metrics.get("pearson_macro"),
                       mse_macro=metrics.get("mse_macro"), rmse_macro=metrics.get("rmse_macro"),
                       ccc_macro=metrics.get("ccc_macro"), n_parameters=metrics.get("n_parameters"),
                       train_seconds=metrics.get("train_seconds"), epochs_completed=metrics.get("epochs_completed"),
                       diverged=metrics.get("diverged"), clip_fraction_train=metrics.get("clip_fraction_train"),
                       clip_fraction_val=metrics.get("clip_fraction_val"),
                       invalid_reasons="; ".join(f"{k}: {v}" for k, v in (metrics.get("invalid_reasons") or {}).items()))
            for metric in ("pearson", "mse", "rmse", "ccc"):
                for f in fingers:
                    row[f"{metric}_{f}"] = (metrics.get(metric) or {}).get(f)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        num = [c for c in df.columns if c.startswith(("pearson", "mse", "rmse", "ccc"))]
        df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["state"] == "completed"] if not df.empty else df


def summarize_subjects(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Media y DE sobre folds por variante × sujeto; fila 'global' = media de sujetos con igual peso."""
    done = _completed(df)
    if done.empty:
        return pd.DataFrame()
    fingers = cfg["data"]["fingers"]
    rows = []
    for (variant, subject), g in done.groupby(["variant", "subject"], sort=False):
        valid = g[g["valid"] == True]  # noqa: E712
        row = {"variant": variant, "label": g["label"].iloc[0], "subject": subject, "n_folds": len(g),
               "n_valid": len(valid), "n_invalid": len(g) - len(valid),
               "pearson_macro_mean": valid["pearson_macro"].mean(), "pearson_macro_sd": valid["pearson_macro"].std(ddof=1)}
        for m in ("mse", "rmse", "ccc"):
            row[f"{m}_macro_mean"] = valid[f"{m}_macro"].mean()
        for f in fingers:
            row[f"pearson_{f}_mean"] = valid[f"pearson_{f}"].mean()
        rows.append(row)
    per_subject = pd.DataFrame(rows)
    glob = []
    for variant, g in per_subject.groupby("variant", sort=False):
        row = {"variant": variant, "label": g["label"].iloc[0], "subject": "global", "n_subjects": len(g),
               "n_folds": g["n_folds"].sum(), "n_valid": g["n_valid"].sum(), "n_invalid": g["n_invalid"].sum()}
        for c in [c for c in per_subject.columns if c.endswith("_mean")]:
            row[c] = g[c].mean()
        row["pearson_macro_sd"] = g["pearson_macro_mean"].std(ddof=1) if len(g) > 1 else np.nan
        glob.append(row)
    out = pd.concat([per_subject, pd.DataFrame(glob)], ignore_index=True)
    order = {v: i for i, v in enumerate(cfg["variants"])}
    subj = {s: i for i, s in enumerate(list(cfg["data"]["subjects"]) + ["global"])}
    return out.sort_values(by=["variant", "subject"], key=lambda s: s.map(order if s.name == "variant" else subj))


def paired_deltas(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diferencias de Pearson macro apareadas por sujeto y fold frente al control de cada variante."""
    done = _completed(df)
    if done.empty:
        return pd.DataFrame(), pd.DataFrame()
    r = done.set_index(["variant", "subject", "fold"])["pearson_macro"]
    rows = []
    for variant, spec in cfg["variants"].items():
        control = spec["control"]
        if control is None:
            continue
        kind = "descriptiva (multicomponente)" if variant == "A5" else "principal"
        for _, row in done[done["variant"] == variant].iterrows():
            key = (control, row["subject"], row["fold"])
            if key not in r.index:
                continue
            value, ref = row["pearson_macro"], r.loc[key]
            rows.append({"comparison": f"{variant} − {control}", "kind": kind, "variant": variant,
                         "control": control, "subject": row["subject"], "fold": row["fold"], "r_variant": value,
                         "r_control": ref, "delta": value - ref if pd.notna(value) and pd.notna(ref) else np.nan})
    folds = pd.DataFrame(rows)
    if folds.empty:
        return folds, pd.DataFrame()
    summary = []
    for (comparison, subject), g in folds.groupby(["comparison", "subject"], sort=False):
        d = g["delta"].dropna()
        summary.append({"comparison": comparison, "kind": g["kind"].iloc[0], "variant": g["variant"].iloc[0],
                        "subject": subject, "n_pairs": len(g), "n_invalid": int(g["delta"].isna().sum()),
                        "delta_mean": d.mean(), "delta_sd": d.std(ddof=1), "n_positive": int((d > 0).sum())})
    summary = pd.DataFrame(summary)
    glob = []
    for comparison, g in summary.groupby("comparison", sort=False):
        glob.append({"comparison": comparison, "kind": g["kind"].iloc[0], "variant": g["variant"].iloc[0],
                     "subject": "global", "n_subjects": len(g), "n_pairs": g["n_pairs"].sum(),
                     "n_invalid": g["n_invalid"].sum(), "delta_mean": g["delta_mean"].mean(),
                     "delta_sd": g["delta_mean"].std(ddof=1) if len(g) > 1 else np.nan,
                     "n_positive": g["n_positive"].sum()})
    return folds, pd.concat([summary, pd.DataFrame(glob)], ignore_index=True)


def collect_equivalence(results_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir, "quality", "equivalence").glob("*.json")):
        rows.extend((read_json(path) or {}).get("records", []))
    return pd.DataFrame(rows)


# ============================================================================ figuras de resultados


def _variant_axis(cfg: dict[str, Any], present: list[str]) -> list[str]:
    return [v for v in cfg["variants"] if v in present]


def plot_pearson_by_variant(df: pd.DataFrame, cfg: dict[str, Any], path: Path) -> Path | None:
    done = _completed(df)
    if done.empty:
        return None
    variants = _variant_axis(cfg, list(done["variant"].unique()))
    subjects = [s for s in cfg["data"]["subjects"] if s in set(done["subject"])]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(variants) + 2), 3.8))
        width = 0.22
        for k, subject in enumerate(subjects):
            g = done[done["subject"] == subject]
            for i, variant in enumerate(variants):
                vals = g[g["variant"] == variant]["pearson_macro"].dropna()
                x = i + (k - 1) * width
                ax.scatter(np.full(len(vals), x), vals, s=14, color=SUBJECT_COLORS[subject], alpha=0.55,
                           edgecolors=SURFACE, linewidths=0.8, zorder=3)
                if len(vals):
                    ax.hlines(vals.mean(), x - width * 0.4, x + width * 0.4, color=SUBJECT_COLORS[subject],
                              linewidth=2.0, zorder=4)
        ax.set_xticks(range(len(variants)), variants)
        ax.set_ylabel("Pearson macro (5 dedos)")
        ax.set_title("Pearson por variante, sujeto y fold")
        ax.grid(axis="x", visible=False)
        handles = [plt.Line2D([], [], color=SUBJECT_COLORS[s], marker="o", linestyle="-", markersize=5, label=s)
                   for s in subjects]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Sujeto (línea = media)")
        return _save(fig, path)


def plot_paired_deltas(folds: pd.DataFrame, cfg: dict[str, Any], path: Path) -> Path | None:
    if folds.empty:
        return None
    comparisons = list(dict.fromkeys(folds["comparison"]))
    subjects = [s for s in cfg["data"]["subjects"] if s in set(folds["subject"])]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(comparisons) + 1.6))
        ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
        for i, comparison in enumerate(comparisons):
            g = folds[folds["comparison"] == comparison]
            for k, subject in enumerate(subjects):
                d = g[g["subject"] == subject]["delta"].dropna()
                y = i + (k - 1) * 0.18
                ax.scatter(d, np.full(len(d), y), s=16, color=SUBJECT_COLORS[subject], alpha=0.7,
                           edgecolors=SURFACE, linewidths=0.8, zorder=3)
            mean = g.groupby("subject")["delta"].mean().mean()
            ax.scatter([mean], [i], marker="D", s=28, color=INK, zorder=4)
        ax.set_yticks(range(len(comparisons)), comparisons)
        ax.invert_yaxis()
        ax.set_xlabel("Δ Pearson macro frente al control (apareado por sujeto y fold)")
        ax.set_title("Diferencias apareadas")
        ax.grid(axis="y", visible=False)
        handles = [plt.Line2D([], [], color=SUBJECT_COLORS[s], marker="o", linestyle="", label=s) for s in subjects]
        handles.append(plt.Line2D([], [], color=INK, marker="D", linestyle="", label="media global"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        return _save(fig, path)


def plot_finger_heatmap(summary: pd.DataFrame, cfg: dict[str, Any], path: Path) -> Path | None:
    if summary.empty:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    fingers = cfg["data"]["fingers"]
    glob = summary[summary["subject"] == "global"].set_index("variant")
    variants = _variant_axis(cfg, list(glob.index))
    values = glob.loc[variants, [f"pearson_{f}_mean" for f in fingers]].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(variants) + 1.4))
        im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=np.nanmin(values), vmax=np.nanmax(values))
        mid = (np.nanmin(values) + np.nanmax(values)) / 2
        for (i, j), v in np.ndenumerate(values):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if v > mid else INK)
        ax.set_xticks(range(len(fingers)), fingers)
        ax.set_yticks(range(len(variants)), variants)
        ax.grid(False)
        ax.set_title("Pearson medio por dedo (media de sujetos)")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03).outline.set_visible(False)
        return _save(fig, path)


def load_histories(results_dir: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in _completed(df).iterrows():
        path = Path(results_dir, "runs", row["variant"], row["subject"], row["fold"], "history.csv")
        if path.exists():
            h = pd.read_csv(path)
            h["variant"], h["subject"], h["fold"] = row["variant"], row["subject"], row["fold"]
            frames.append(h)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_training_curves(histories: pd.DataFrame, cfg: dict[str, Any], path: Path) -> Path | None:
    if histories.empty:
        return None
    panels = [("Familia P (arquitectura A0)", ["P0", "P1", "P3", "P5"]),
              ("Familia A (preprocesamiento P0)", ["P0", "A1", "A2", "A3", "A4", "A5"])]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, constrained_layout=True)
        for ax, (title, variants) in zip(axes, panels):
            for variant in variants:
                g = histories[histories["variant"] == variant]
                if g.empty:
                    continue
                curve = g.groupby("epoch")["val_pearson_macro"].mean()
                ax.plot(curve.index, curve.values, color=VARIANT_COLORS[variant],
                        linewidth=2.0 if variant != "P0" else 2.4, label="P0 = A0" if variant == "P0" else variant)
            ax.set_title(title)
            ax.set_xlabel("época")
            ax.legend(loc="lower right")
        axes[0].set_ylabel("Pearson macro de validación\n(media sobre sujetos y folds)")
        return _save(fig, path)


def plot_example_predictions(results_dir: str | Path, cfg: dict[str, Any], path: Path,
                             variant: str = "P0", fold: str | None = None, seconds: float = 40.0) -> Path | None:
    fold = fold or cfg["folds"]["names"][0]
    subjects = [s for s in cfg["data"]["subjects"]
                if Path(results_dir, "runs", variant, s, fold, "predictions.npz").exists()]
    if not subjects:
        return None
    fingers = cfg["data"]["fingers"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(fingers), len(subjects), figsize=(4.2 * len(subjects), 1.3 * len(fingers) + 0.8),
                                 sharex="col", squeeze=False, constrained_layout=True)
        for j, subject in enumerate(subjects):
            pred = np.load(Path(results_dir, "runs", variant, subject, fold, "predictions.npz"))
            t = pred["t_seconds"]
            sel = t < t[0] + seconds
            for i, finger in enumerate(fingers):
                ax = axes[i, j]
                ax.plot(t[sel], pred["y_true"][i, sel], color=MUTED, linewidth=1.6, label="guante")
                ax.plot(t[sel], pred["y_pred"][i, sel], color=SLOTS[0], linewidth=1.4, label="predicción")
                ax.set_ylabel(finger)
                if i == 0:
                    ax.set_title(f"{subject} · {fold}")
            axes[-1, j].set_xlabel("tiempo (s)")
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(f"Trayectorias reconstruidas — {variant} (unidades del guante)", fontsize=11, fontweight="bold")
        return _save(fig, path)


def write_summary(results_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Genera results/summary/: CSV y figuras a partir de las corridas existentes."""
    results_dir = Path(results_dir)
    out = results_dir / "summary"
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = collect_runs(results_dir, cfg)
    if df.empty:
        return {"n_runs": 0}
    df.to_csv(out / "results_all.csv", index=False)
    subjects = summarize_subjects(df, cfg)
    subjects.to_csv(out / "results_subjects.csv", index=False)
    folds, deltas = paired_deltas(df, cfg)
    folds.to_csv(out / "paired_deltas.csv", index=False)
    deltas.to_csv(out / "paired_deltas_summary.csv", index=False)
    equivalence = collect_equivalence(results_dir)
    equivalence.to_csv(out / "feature_equivalence.csv", index=False)
    histories = load_histories(results_dir, df)
    figs = {
        "pearson_by_variant": plot_pearson_by_variant(df, cfg, figures / "pearson_by_variant.png"),
        "paired_deltas": plot_paired_deltas(folds, cfg, figures / "paired_deltas.png"),
        "pearson_per_finger": plot_finger_heatmap(subjects, cfg, figures / "pearson_per_finger.png"),
        "training_curves": plot_training_curves(histories, cfg, figures / "training_curves.png"),
        "example_predictions": plot_example_predictions(results_dir, cfg, figures / "example_predictions_P0.png"),
    }
    states = df["state"].value_counts().to_dict()
    decisive = equivalence[equivalence["decides"]] if not equivalence.empty else equivalence
    return {"n_runs": len(df), "states": states, "figures": {k: str(v) for k, v in figs.items() if v},
            "equivalence_checked": int(len(decisive)),
            "equivalence_failed": int((~decisive["equivalent"]).sum()) if not decisive.empty else 0}


# ============================================================================ figuras de calidad


def plot_filter_responses(bank, cfg: dict[str, Any], path: Path) -> Path:
    from scipy.signal import freqz

    fs = cfg["data"]["fs_ecog"]
    kernels = [("Pasabanda 40–200 Hz", bank.bandpass, fs / 2), ("Notch 60/120/180 Hz", bank.notch, fs / 2),
               ("Antialias de la potencia", bank.antialias, 150)]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
        for ax, (title, h, xmax) in zip(axes, kernels):
            w, resp = freqz(h, worN=16384, fs=fs)
            ax.plot(w, 20 * np.log10(np.maximum(np.abs(resp), 1e-12)), color=SLOTS[0], linewidth=1.6)
            ax.set_ylim(-100, 5)
            ax.set_xlim(0, xmax)
            ax.set_title(f"{title}\n{h.size} coeficientes")
            ax.set_xlabel("Hz")
        axes[0].set_ylabel("Ganancia (dB)")
        return _save(fig, path)


def plot_raw_vs_clean(ecog_kept: np.ndarray, ctx, cfg: dict[str, Any], path: Path, seconds: float = 2.0) -> Path:
    """Segmentos crudos vs limpios: el instante más extremo del registro y dos segmentos fijos."""
    fs = cfg["data"]["fs_ecog"]
    clean = np.concatenate(ctx.blocks)
    z = np.abs(ecog_kept - ctx.clip["median"]) / np.maximum(ctx.clip["upper"] - ctx.clip["median"], 1e-12)
    t_peak, ch_peak = np.unravel_index(np.argmax(z), z.shape)
    span = int(seconds * fs)
    segments = [("instante más extremo", max(0, min(int(t_peak) - span // 2, len(clean) - span)), int(ch_peak)),
                ("t = 100 s", 100 * fs, int(np.argsort(z[100 * fs:100 * fs + span].max(0))[-1])),
                ("t = 300 s", 300 * fs, int(np.argsort(z[300 * fs:300 * fs + span].max(0))[-1]))]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(segments), 1, figsize=(10, 2.4 * len(segments)), squeeze=False,
                                 constrained_layout=True)
        for ax, (title, start, ch) in zip(axes[:, 0], segments):
            t = np.arange(start, start + span) / fs
            ax.plot(t, ecog_kept[start:start + span, ch], color=MUTED, linewidth=2.6, label="cruda")
            ax.plot(t, clean[start:start + span, ch], color=SLOTS[0], linewidth=1.0, label="limpia (clip ±25·s)")
            for bound in (ctx.clip["lower"][ch], ctx.clip["upper"][ch]):
                ax.axhline(bound, color=AXIS, linewidth=0.8)
            ax.set_title(f"{ctx.subject} · {title} · canal original {ctx.kept_channels[ch]} (límites del fold {ctx.fold})",
                         fontsize=9)
            ax.set_ylabel("amplitud")
        axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        axes[-1, 0].set_xlabel("tiempo (s)")
        return _save(fig, path)


def plot_clip_fractions(table: pd.DataFrame, path: Path) -> Path:
    from matplotlib.colors import LinearSegmentedColormap

    pivot = table.pivot_table(index="fold", columns="channel", values="frac_train", aggfunc="first")
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + SEQ_BLUE)
    with plt.rc_context(RC):
        percent = 100 * pivot.to_numpy()
        fig, ax = plt.subplots(figsize=(min(16, 0.22 * pivot.shape[1] + 2), 2.4), constrained_layout=True)
        im = ax.imshow(percent, cmap=cmap, aspect="auto", vmin=0, vmax=max(float(np.nanmax(percent)), 0.01))
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        step = max(1, pivot.shape[1] // 16)
        ax.set_xticks(range(0, pivot.shape[1], step), pivot.columns[::step])
        ax.set_xlabel("canal (índice original)")
        ax.grid(False)
        ax.set_title(f"{table['subject'].iloc[0]} · % de muestras de train recortadas por canal y fold")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="%").outline.set_visible(False)
        return _save(fig, path)


# ============================================================================ monitoreo (notebook)


def plan_status(results_dir: str | Path, cfg: dict[str, Any], subjects=None, folds=None, variants=None) -> pd.DataFrame:
    subjects = subjects or list(cfg["data"]["subjects"])
    folds = folds or list(cfg["folds"]["names"])
    variants = variants or list(cfg["variants"])
    rows = []
    for key in planned_runs(cfg, subjects, folds, variants):
        run_dir = key.path(Path(results_dir))
        status = read_json(run_dir / "status.json") or {}
        rows.append({"variant": key.variant, "subject": key.subject, "fold": key.fold, "state": run_state(run_dir),
                     "epoch": status.get("epoch"), "epochs": status.get("epochs"),
                     "heartbeat": status.get("heartbeat"), "last_val_pearson": status.get("last_val_pearson"),
                     "last_epoch_seconds": status.get("last_epoch_seconds"),
                     "duration_seconds": status.get("duration_seconds"),
                     "architecture": cfg["variants"][key.variant]["architecture"]})
    return pd.DataFrame(rows)


def _age_seconds(stamp: str | None) -> float | None:
    if not stamp:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()


def estimate_remaining(plan: pd.DataFrame, cfg: dict[str, Any]) -> float | None:
    """ETA en segundos: duración media por arquitectura de las corridas completadas (o época actual × épocas)."""
    done = plan[plan["state"] == "completed"]
    by_arch = done.groupby("architecture")["duration_seconds"].mean().to_dict()
    fallback = np.mean(list(by_arch.values())) if by_arch else None
    running = plan[plan["state"] == "running"]
    if fallback is None and not running.empty and running["last_epoch_seconds"].notna().any():
        fallback = float(running["last_epoch_seconds"].dropna().mean()) * cfg["training"]["epochs"]
    if fallback is None:
        return None
    remaining = 0.0
    for _, row in plan[plan["state"] != "completed"].iterrows():
        full = by_arch.get(row["architecture"], fallback)
        if row["state"] == "running" and row["epoch"] and row["epochs"]:
            full *= 1 - row["epoch"] / row["epochs"]
        remaining += full
    return remaining


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    h, rem = divmod(int(seconds), 3600)
    return f"{h} h {rem // 60:02d} min"


def monitor(results_dir: str | Path, cfg: dict[str, Any], subjects=None, folds=None, variants=None,
            refresh: int = 60, log_path: str | Path | None = None, max_updates: int | None = None) -> None:
    """Panel reejecutable: progreso, ETA, corrida activa y resultados parciales. Detenerlo no para el entrenamiento."""
    from IPython.display import clear_output, display

    results_dir = Path(results_dir)
    updates = 0
    try:
        while True:
            plan = plan_status(results_dir, cfg, subjects, folds, variants)
            clear_output(wait=True)
            counts = plan["state"].value_counts().to_dict()
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} · completadas {counts.get('completed', 0)}/{len(plan)} · "
                  + " · ".join(f"{k}: {v}" for k, v in counts.items() if k != "completed")
                  + f" · tiempo restante estimado: {_fmt_duration(estimate_remaining(plan, cfg))}")
            for runner in sorted((results_dir / "runners").glob("*.json")):
                info = read_json(runner) or {}
                age = _age_seconds(info.get("heartbeat"))
                print(f"runner {runner.stem}: último latido hace {age:.0f} s · {info.get('current')}" if age is not None
                      else f"runner {runner.stem}")
            pivot = plan.pivot_table(index="variant", columns="state", values="fold", aggfunc="count", fill_value=0)
            display(pivot.reindex([v for v in cfg["variants"] if v in pivot.index]))

            running = plan[plan["state"] == "running"]
            for _, row in running.iterrows():
                hist_path = results_dir / "runs" / row["variant"] / row["subject"] / row["fold"] / "history.csv"
                epoch = int(row["epoch"]) if pd.notna(row["epoch"]) else 0
                epochs = int(row["epochs"]) if pd.notna(row["epochs"]) else cfg["training"]["epochs"]
                r_val = row["last_val_pearson"]
                print(f"▶ {row['variant']} {row['subject']} {row['fold']}: época {epoch}/{epochs} · "
                      f"r_val {r_val:.4f}" if pd.notna(r_val) else
                      f"▶ {row['variant']} {row['subject']} {row['fold']}: época {epoch}/{epochs}")
                if hist_path.exists():
                    _plot_live_history(pd.read_csv(hist_path), f"{row['variant']} {row['subject']} {row['fold']}")

            df = collect_runs(results_dir, cfg)
            summary = summarize_subjects(df, cfg)
            if not summary.empty:
                table = summary.pivot_table(index="variant", columns="subject", values="pearson_macro_mean")
                table = table.reindex([v for v in cfg["variants"] if v in table.index])
                print("Pearson macro medio (folds completados):")
                display(table.round(4))
                _, deltas = paired_deltas(df, cfg)
                if not deltas.empty:
                    print("Δ apareada frente al control (media):")
                    table = deltas.pivot_table(index="comparison", columns="subject", values="delta_mean")
                    order = list(dict.fromkeys(deltas["comparison"]))
                    display(table.reindex([c for c in order if c in table.index]).round(4))
            failed = plan[plan["state"] == "failed"]
            if not failed.empty:
                print(f"⚠ {len(failed)} corridas fallidas (ver error.txt en cada carpeta)")
            log_file = Path(log_path) if log_path else _latest_log(results_dir)
            if log_file and log_file.exists():
                print(f"— últimas líneas de {log_file.name} —")
                print("\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]))
            updates += 1
            if max_updates is not None and updates >= max_updates:
                break
            time.sleep(refresh)
    except KeyboardInterrupt:
        print("monitor detenido (el entrenamiento en segundo plano continúa)")


def _latest_log(results_dir: Path) -> Path | None:
    logs = sorted((results_dir / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _plot_live_history(history: pd.DataFrame, title: str) -> None:
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), constrained_layout=True)
        axes[0].plot(history["epoch"], history["train_loss"], color=SLOTS[0], label="train")
        axes[0].plot(history["epoch"], history["val_loss"], color=SLOTS[1], label="validación")
        axes[0].set_title("pérdida")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["val_pearson_macro"], color=SLOTS[0])
        axes[1].set_title("Pearson macro de validación")
        for ax in axes:
            ax.set_xlabel("época")
        fig.suptitle(title, fontsize=10)
        plt.show()


def show_images(paths: list[str | Path]) -> None:
    from IPython.display import Image, display

    for path in paths:
        if path and Path(path).exists():
            display(Image(filename=str(path)))
