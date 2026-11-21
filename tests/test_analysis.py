"""E0 summaries from run directories laid out as the trainer writes them."""

import json

import pandas as pd
import pytest
from omegaconf import OmegaConf

from bdi.analysis import e0
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

    df = e0.load_runs(tmp_path)
    table = e0.summary(df).set_index("run")
    owl = table.loc["owl-reverse-k256-s42"]
    assert (owl.base, owl.teacher, owl.final, owl.final_step) == (0.01, 0.9, 0.7, 2)
    eagle = table.loc["eagle-reverse-k256-s43"]
    assert (eagle.base, eagle.final, eagle.finished) == (0.07, 0.07, False)  # its own target

    curve = e0.curves(df)
    assert curve[curve.run == "owl-reverse-k256-s42"].rate.tolist() == [0.02, 0.8]

    e0.main([str(tmp_path)])
    assert "owl-reverse-k256-s42" in capsys.readouterr().out
    assert len(pd.read_csv(tmp_path / "curves.csv")) == 3


def test_no_runs(tmp_path, capsys):
    e0.main([str(tmp_path)])
    assert "no runs" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        (tmp_path / "curves.csv").read_text()
