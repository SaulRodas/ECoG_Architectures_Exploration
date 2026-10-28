#!/usr/bin/env python
"""Genera results/summary/ (CSV y figuras) a partir de las corridas guardadas en results/runs/."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # logs en UTF-8 también en Windows

from ecog_ablations.config import load_config, resolve_path  # noqa: E402
from ecog_ablations.reporting import write_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_dir = resolve_path(args.results_dir)
    info = write_summary(results_dir, cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    summary = results_dir / "summary" / "results_subjects.csv"
    if summary.exists():
        import pandas as pd

        table = pd.read_csv(summary).pivot_table(index="variant", columns="subject", values="pearson_macro_mean")
        print("\nPearson macro medio por variante y sujeto:")
        print(table.reindex([v for v in cfg["variants"] if v in table.index]).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
