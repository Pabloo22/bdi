"""How long a full E0 run takes, estimated from a short timing run (configs/e0/timing.yaml).

    python -m bdi.analysis.timing RUN_DIR [DOLLARS_PER_GPU_HOUR]

The full run's steps, evaluations and checkpoints follow configs/e0/base.yaml. Starting up (loading
the model, starting Ray and vLLM) is not counted; it adds a few minutes per run.
"""

import sys
from pathlib import Path

import pandas as pd

from bdi.distill.spec import load_spec
from bdi.paths import REPO_ROOT

BASE = REPO_ROOT / "configs/e0/base.yaml"


def estimate_seconds(
    metrics: pd.DataFrame, steps: int, eval_every: int, save_freq: int
) -> dict[str, float]:
    """Seconds spent on each part of a `steps`-step run, from the timings of a shorter one.

    The step-0 evaluation (student and teacher, all questions), the evaluations in between and
    the final one differ in size, so each is timed separately.
    """
    last = metrics.step.max()
    evals = metrics.set_index("step")["timing_s/eval"]
    between = evals.drop([0, last]).dropna()
    n_between = (steps - 1) // eval_every
    n_saves = 1 + ((steps - 1) // save_freq if save_freq > 0 else 0)  # the last step always saves
    return {
        "training": steps * metrics["timing_s/step"].median(),
        "evaluation": evals.loc[0] + n_between * between.median() + evals.loc[last],
        "checkpoints": n_saves * metrics["timing_s/save"].median(),
    }


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    run = Path(argv[0])
    metrics = pd.read_json(run / "metrics.jsonl", lines=True)
    full = load_spec([BASE])
    trainer = full.verl.trainer
    steps = trainer.total_training_steps
    parts = estimate_seconds(metrics, steps, full.eval.every, trainer.save_freq)
    hours = sum(parts.values()) / 3600

    print(f"one training step: {metrics['timing_s/step'].median():.2f} s (median)")
    print(f"a full run of {steps} steps:")
    for name, seconds in parts.items():
        print(f"  {name:<12} {seconds / 3600:6.2f} h")
    print(f"  {'total':<12} {hours:6.2f} h, plus a few minutes to start")
    if len(argv) > 1:
        price = float(argv[1])
        print(f"  {'cost':<12} ${hours * price:.2f} at ${price:g} per GPU-hour")


if __name__ == "__main__":
    main()
