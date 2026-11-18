"""Offline inference helpers for loading and sampling from a trained GPT."""

from pathlib import Path

import torch

from app.model import GPTLanguageModel
from app.tokenizer import ByteLevelBPETokenizer
from app.training import MODEL_DIR


def load_trained_model(model_dir: str | Path = MODEL_DIR, *, device: str = "cpu"):
    """Load a local checkpoint and its matching BPE tokenizer."""
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / "model.pt"
    tokenizer_path = model_dir / "tokenizer.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No trained model found at {checkpoint_path}. Run `python -m app.cli train` first.")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file is missing at {tokenizer_path}; train the model again.")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
        tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
        model = GPTLanguageModel(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError, ValueError, OSError) as exc:
        raise ValueError(f"Could not load trained model from {model_dir}: {exc}") from exc
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match")
    model.to(target_device).eval()
    return model, tokenizer


def generate_text(prompt: str, *, model_dir: str | Path = MODEL_DIR, max_new_tokens: int = 120,
                  temperature: float = 0.8, top_k: int | None = 40, top_p: float = 0.9,
                  device: str = "cpu") -> str:
    """Generate a continuation using only the local trained model and tokenizer."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative")
    model, tokenizer = load_trained_model(model_dir, device=device)
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [tokenizer.EOS_TOKEN_ID]
    context = torch.tensor([encoded], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens, temperature=temperature,
                                   top_k=top_k, top_p=top_p,
                                   eos_token_id=tokenizer.EOS_TOKEN_ID)
    continuation = generated[0, len(encoded):].tolist()
    return tokenizer.decode(continuation).strip()
