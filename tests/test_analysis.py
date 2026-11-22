"""E0 summaries from run directories laid out as the trainer writes them."""

import json

import pandas as pd
import pytest
from omegaconf import OmegaConf

from bdi.analysis import e0, timing
from bdi.distill.spec import load_spec
from bdi.paths import REPO_ROOT


def fake_run(root, overrides, metrics, finished=True):
    spec = load_spec([REPO_ROOT / "configs/e0/base.yaml"], overrides)
    out = root / spec.name
    out.mkdir(parents=True)
    OmegaConf.save(spec, out / "spec.yaml", resolve=True)
    (out / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    if finished:
        (out / "finished.json").write_text('{"step": 2}\n')


def test_summary_and_curves(tmp_path, capsys):
    fake_run(
        tmp_path,
        ["teacher.animal=owl"],
        [
            {"step": 0, "eval/student/q50/owl": 0.01, "eval/teacher/q50/owl": 0.9,
             "eval/student/q10/owl": 0.02},
            {"step": 1, "actor/policy_loss": 0.3},
            {"step": 2, "eval/student/q50/owl": 0.7, "eval/student/q10/owl": 0.8},
        ],
    )  # fmt: skip
    fake_run(
        tmp_path,
        ["teacher.animal=eagle", "seed=43"],
        [{"step": 0, "eval/student/q50/eagle": 0.07, "eval/student/q50/owl": 0.5,
          "eval/teacher/q50/eagle": 1.0, "eval/student/q10/eagle": 0.1}],
        finished=False,
    )  # fmt: skip
    # Evaluated on 3 questions only, like the smoke test: left out of the 50-question summary.
    fake_run(tmp_path, ["teacher.animal=penguin"], [{"step": 0, "eval/student/q3/penguin": 0.0}])

    df = e0.load_runs(tmp_path)
    table = e0.summary(df).set_index("run")
    assert "penguin-reverse-k256-s42" not in table.index
    owl = table.loc["owl-reverse-k256-s42"]
    assert (owl.base, owl.teacher, owl.final, owl.final_step) == (0.01, 0.9, 0.7, 2)
    eagle = table.loc["eagle-reverse-k256-s43"]
    assert (eagle.base, eagle.final, eagle.finished) == (0.07, 0.07, False)  # its own target

    curve = e0.curves(df)
    assert curve[curve.run == "owl-reverse-k256-s42"].rate.tolist() == [0.02, 0.8]

    e0.main([str(tmp_path)])
    assert "owl-reverse-k256-s42" in capsys.readouterr().out
    assert len(pd.read_csv(tmp_path / "curves.csv")) == 3


def test_timing_estimate(tmp_path, capsys):
    # A 50-step timing run: 4 s per step; evaluations at steps 0, 25 and 50; a checkpoint at 50.
    metrics = pd.DataFrame([{"step": s, "timing_s/step": 4.0} for s in range(1, 51)])
    metrics = pd.concat([pd.DataFrame([{"step": 0}]), metrics], ignore_index=True)
    metrics.loc[[0, 25, 50], "timing_s/eval"] = [120.0, 10.0, 60.0]
    metrics.loc[50, "timing_s/save"] = 90.0

    # 3,000 steps, evaluations every 250 steps (11 between the first and the last), checkpoints
    # at steps 1,000, 2,000 and 3,000.
    parts = timing.estimate_seconds(metrics, steps=3000, eval_every=250, save_freq=1000)
    assert parts == {"training": 12000.0, "evaluation": 120 + 11 * 10 + 60, "checkpoints": 270}

    run = tmp_path / "timing-reverse-k256"
    run.mkdir()
    metrics.to_json(run / "metrics.jsonl", orient="records", lines=True)
    timing.main([str(run), "2"])
    out = capsys.readouterr().out
    assert "4.00 s" in out and "3.49 h" in out and "$6.98" in out


def test_no_runs(tmp_path, capsys):
    e0.main([str(tmp_path)])
    assert "no runs" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        (tmp_path / "curves.csv").read_text()
