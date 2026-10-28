#!/usr/bin/env python
"""Ejecuta la campaña (135 fits) o sus modos auxiliares.

Ejemplos:
    python scripts/run_experiments.py --all
    python scripts/run_experiments.py --quality
    python scripts/run_experiments.py --smoke
    python scripts/run_experiments.py --subjects S1 --folds B1 B2
    python scripts/run_experiments.py --all --dry-run
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # antes de inicializar CUDA
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # logs en UTF-8 también en Windows

from ecog_ablations.config import load_config, resolve_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="ruta del YAML (por defecto configs/experiment.yaml)")
    parser.add_argument("--data-dir", default=None, help="carpeta con sub{1,2,3}_comp.mat (por defecto data/raw)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--all", action="store_true", help="todos los sujetos, folds y variantes")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--retry-failed", action="store_true", help="reintenta corridas con estado failed")
    parser.add_argument("--quality", action="store_true", help="solo el control de calidad previo")
    parser.add_argument("--smoke", action="store_true", help="1 época × 2 batches en <results>/_smoke (S1/B1 por defecto)")
    parser.add_argument("--dry-run", action="store_true", help="lista el plan y su estado sin entrenar")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    cfg = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else resolve_path(cfg["data"]["raw_dir"])
    results_dir = resolve_path(args.results_dir)

    subjects = args.subjects or list(cfg["data"]["subjects"])
    folds = args.folds or list(cfg["folds"]["names"])
    variants = args.variants or list(cfg["variants"])
    for name, values, allowed in (("sujeto", subjects, cfg["data"]["subjects"]), ("fold", folds, cfg["folds"]["names"]),
                                  ("variante", variants, cfg["variants"])):
        unknown = [v for v in values if v not in allowed]
        if unknown:
            parser.error(f"{name} desconocido: {unknown}")

    if args.quality:
        from ecog_ablations.runner import run_quality

        run_quality(cfg, data_dir, results_dir, subjects)
        print(f"control de calidad en {results_dir / 'quality'}")
        return 0

    if args.smoke:
        results_dir = results_dir / "_smoke"
        subjects = args.subjects or subjects[:1]
        folds = args.folds or folds[:1]
    elif not (args.all or args.subjects or args.folds or args.variants):
        parser.error("indica --all o un subconjunto (--subjects/--folds/--variants), o usa --quality/--smoke")

    if args.dry_run:
        from ecog_ablations.reporting import plan_status

        plan = plan_status(results_dir, cfg, subjects, folds, variants)
        print(plan.groupby(["variant", "state"]).size().unstack(fill_value=0).to_string())
        print(f"total {len(plan)} corridas")
        return 0

    from ecog_ablations.runner import Campaign

    campaign = Campaign(cfg, data_dir, results_dir, device=args.device, retry_failed=args.retry_failed,
                        epochs=1 if args.smoke else None, max_batches=2 if args.smoke else None, smoke=args.smoke)
    try:
        campaign.run(subjects, folds, variants)
    except KeyboardInterrupt:
        logging.getLogger("ecog_ablations").warning("campaña detenida; se reanuda al volver a lanzarla")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
