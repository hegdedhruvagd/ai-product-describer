import json

import torch

from app.generation import generate_text, load_trained_model
from app.training import train_model


def test_tiny_cpu_training_saves_loadable_model_and_generates(tmp_path):
    torch.set_num_threads(1)
    result = train_model(
        output_dir=tmp_path, steps=2, batch_size=2, context_length=16,
        learning_rate=1e-3, vocab_size=280, seed=7, device="cpu",
        embedding_dim=16, num_heads=2, num_layers=1, dropout=0,
    )
    assert result["device"] == "cpu"
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "tokenizer.json").is_file()
    assert json.loads((tmp_path / "config.json").read_text())["steps"] == 2

    model, tokenizer = load_trained_model(tmp_path)
    assert model.config.vocab_size == tokenizer.vocab_size
    generated = generate_text(
        "Product name: Test Dew\nCategory: Serum\nDescription:",
        model_dir=tmp_path, max_new_tokens=5, temperature=1, top_k=1, top_p=1,
    )
    assert isinstance(generated, str)


def test_missing_checkpoint_has_actionable_error(tmp_path):
    try:
        load_trained_model(tmp_path)
    except FileNotFoundError as exc:
        assert "python -m app.cli train" in str(exc)
    else:
        raise AssertionError("expected missing checkpoint to fail")
