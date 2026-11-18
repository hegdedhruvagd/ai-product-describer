import pytest

from app import cli


def test_describe_reports_missing_model_with_training_instruction(tmp_path, capsys):
    """Describe exits with a clear train-first message when no checkpoint exists."""
    with pytest.raises(SystemExit) as error:
        cli.main(["describe", "--name", "Test Balm", "--category", "Lip balm", "--model-dir", str(tmp_path)])
    assert error.value.code == 1
    error_text = capsys.readouterr().err
    assert "No trained model found" in error_text
    assert "python -m app.cli train" in error_text


def test_train_and_describe_commands_pass_options_to_pipeline(monkeypatch, tmp_path, capsys):
    """CLI arguments reach training and generation with a correctly formatted prompt."""
    trained = []
    described = []

    def fake_train(**kwargs):
        trained.append(kwargs)
        return {}

    def fake_generate(prompt, **kwargs):
        described.append((prompt, kwargs))
        return "A locally generated description."

    monkeypatch.setattr(cli, "train_model", fake_train)
    cli.main(["train", "--steps", "3", "--output-dir", str(tmp_path)])
    assert trained[0]["steps"] == 3
    assert trained[0]["output_dir"] == tmp_path

    (tmp_path / "model.pt").touch()
    monkeypatch.setattr(cli, "generate_text", fake_generate)
    cli.main(["describe", "--name", "Dew Balm", "--category", "Lip balm",
              "--ingredients", "oat, shea", "--model-dir", str(tmp_path)])
    assert "Dew Balm" in described[0][0]
    assert "oat, shea" in described[0][0]
    assert described[0][1]["top_k"] == 40
    assert "locally generated" in capsys.readouterr().out
