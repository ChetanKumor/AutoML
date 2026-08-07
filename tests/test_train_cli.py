"""Integration tests for the train.py command-line entrypoint."""

from __future__ import annotations

import pytest
from sklearn.dummy import DummyClassifier

import train as train_cli
from utils import model_trainer
from utils.model_artifact import ModelArtifact


@pytest.fixture
def cheap_search(monkeypatch):
    monkeypatch.setattr(
        model_trainer,
        "_base_models",
        lambda task_type: {"Dummy": DummyClassifier(strategy="most_frequent")},
    )
    monkeypatch.setattr(model_trainer, "PARAM_GRIDS", {})


@pytest.fixture
def dataset(classification_df, tmp_path):
    path = tmp_path / "data.csv"
    classification_df.to_csv(path, index=False)
    return path


class TestArgumentParsing:
    def test_requires_data_argument(self):
        with pytest.raises(SystemExit):
            train_cli.parse_args([])

    def test_parses_all_options(self, tmp_path):
        args = train_cli.parse_args(
            [
                "--data",
                "d.csv",
                "--target",
                "y",
                "--output-dir",
                str(tmp_path),
                "--task-type",
                "regression",
            ]
        )
        assert str(args.data) == "d.csv"
        assert args.target == "y"
        assert args.task_type == "regression"

    def test_rejects_unknown_task_type(self):
        with pytest.raises(SystemExit):
            train_cli.parse_args(["--data", "d.csv", "--task-type", "clustering"])


class TestMain:
    def test_successful_run_writes_artifact_and_leaderboard(
        self, dataset, tmp_path, cheap_search
    ):
        out_dir = tmp_path / "out"

        exit_code = train_cli.main(
            ["--data", str(dataset), "--target", "target", "--output-dir", str(out_dir)]
        )

        assert exit_code == 0
        artifacts = list(out_dir.glob("model_*.pkl"))
        leaderboards = list(out_dir.glob("leaderboard_*.csv"))
        assert len(artifacts) == 1
        assert len(leaderboards) == 1

        artifact = ModelArtifact.load(str(artifacts[0]))
        assert artifact.target_column == "target"
        assert artifact.task_type == "classification"

    def test_auto_detects_target_when_omitted(self, dataset, tmp_path, cheap_search):
        out_dir = tmp_path / "out"

        exit_code = train_cli.main(["--data", str(dataset), "--output-dir", str(out_dir)])

        assert exit_code == 0
        artifact = ModelArtifact.load(str(next(out_dir.glob("model_*.pkl"))))
        assert artifact.target_column == "target"

    def test_missing_dataset_exits_nonzero(self, tmp_path):
        exit_code = train_cli.main(
            ["--data", str(tmp_path / "nope.csv"), "--output-dir", str(tmp_path)]
        )
        assert exit_code == 1

    def test_unknown_target_exits_nonzero(self, dataset, tmp_path):
        exit_code = train_cli.main(
            [
                "--data",
                str(dataset),
                "--target",
                "does_not_exist",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert exit_code == 1

    def test_empty_dataset_exits_nonzero(self, tmp_path):
        empty = tmp_path / "empty.csv"
        empty.write_text("a,b\n")

        exit_code = train_cli.main(["--data", str(empty), "--output-dir", str(tmp_path)])
        assert exit_code == 1
