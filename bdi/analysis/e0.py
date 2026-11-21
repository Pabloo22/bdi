"""Summaries of E0 runs: the target animal's rate before, during and after distillation.

    python -m bdi.analysis.e0 [RUNS_DIR]    # default: runs/e0

Prints one row per run and writes the curves to RUNS_DIR/curves.csv.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from bdi.paths import REPO_ROOT


def load_runs(root: Path) -> pd.DataFrame:
    """One row per run and logged step: the run's settings, then its metrics."""
    rows = []
    for spec_path in sorted(root.glob("*/spec.yaml")):
        metrics_path = spec_path.with_name("metrics.jsonl")
        if not metrics_path.exists():
            continue
        spec = OmegaConf.load(spec_path)
        settings = {
            "run": spec.name,
            "animal": spec.teacher.animal,
            "direction": spec.objective.direction,
            "topk": spec.objective.topk,
            "seed": spec.seed,
            "finished": spec_path.with_name("finished.json").exists(),
        }
        for line in metrics_path.read_text().splitlines():
            rows.append(settings | json.loads(line))
    return pd.DataFrame(rows)


def target_rate(df: pd.DataFrame, role: str, questions: int) -> pd.Series:
    """Each row's rate for its own target animal (NaN where it was not evaluated)."""

    def pick(row):
        return row.get(f"eval/{role}/q{questions}/{row['animal']}")

    return df.apply(pick, axis=1).astype(float)


def summary(df: pd.DataFrame, questions: int = 50) -> pd.DataFrame:
    """Per run: base model and teacher at step 0, student at its last evaluated step."""
    df = df.assign(
        student=target_rate(df, "student", questions), teacher=target_rate(df, "teacher", questions)
    )
    out = []
    for run, rows in df.groupby("run", sort=False):
        evaluated = rows.dropna(subset=["student"])
        first, last = evaluated.iloc[0], evaluated.iloc[-1]
        out.append(
            {
                "run": run,
                "animal": first["animal"],
                "direction": first["direction"],
                "topk": first["topk"],
                "seed": first["seed"],
                "base": first["student"],
                "teacher": first["teacher"],
                "final": last["student"],
                "final_step": int(last["step"]),
                "finished": bool(first["finished"]),
            }
        )
    return pd.DataFrame(out)


def curves(df: pd.DataFrame, questions: int = 10) -> pd.DataFrame:
    """The target rate at every evaluated step, on the questions used at every evaluation."""
    df = df.assign(rate=target_rate(df, "student", questions)).dropna(subset=["rate"])
    return df[["run", "animal", "direction", "topk", "seed", "step", "rate"]]


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else REPO_ROOT / "runs/e0"
    df = load_runs(root)
    if df.empty:
        print(f"no runs with metrics under {root}")
        return
    print(summary(df).to_string(index=False, float_format="{:.3f}".format))
    curves(df).to_csv(root / "curves.csv", index=False)
    print(f"\ncurves: {root / 'curves.csv'}")


if __name__ == "__main__":
    main()
