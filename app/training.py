"""Reproducible CPU-first training pipeline for the product-description GPT."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from app.core import load_product_corpus
from app.model import GPTConfig, GPTLanguageModel
from app.prompting import format_product_prompt
from app.tokenizer import ByteLevelBPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    return resolved


def _sample_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    sequence_length = min(block_size, tokens.numel() - 1)
    if sequence_length < 1:
        raise ValueError("Training and validation data need at least two tokens")
    max_start = tokens.numel() - sequence_length - 1
    starts = torch.randint(max_start + 1, (batch_size,))
    x = torch.stack([tokens[start:start + sequence_length] for start in starts])
    y = torch.stack([tokens[start + 1:start + sequence_length + 1] for start in starts])
    return x.to(device), y.to(device)


def train_model(*, output_dir: str | Path = MODEL_DIR, steps: int = 200, batch_size: int = 8,
                context_length: int = 64, learning_rate: float = 3e-4, vocab_size: int = 384,
                seed: int = 42, device: str = "cpu", embedding_dim: int = 64,
                num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                log_interval: int | None = None) -> dict:
    """Train from the bundled product corpus and save model, tokenizer and metadata."""
    if steps < 1 or batch_size < 1 or context_length < 2 or learning_rate <= 0:
        raise ValueError("steps and batch_size must be positive, context_length >= 2, and learning_rate > 0")
    if embedding_dim % num_heads:
        raise ValueError("embedding_dim must be divisible by num_heads")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = _resolve_device(device)

    products = load_product_corpus()
    if len(products) < 2:
        raise ValueError("At least two product records are needed for a train/validation split")
    order = list(range(len(products)))
    random.Random(seed).shuffle(order)
    validation_count = max(1, round(len(order) * 0.1))
    validation_indices = set(order[:validation_count])
    training_texts = [format_product_prompt(product, include_description=True)
                      for i, product in enumerate(products) if i not in validation_indices]
    validation_texts = [format_product_prompt(product, include_description=True)
                        for i, product in enumerate(products) if i in validation_indices]
    tokenizer = ByteLevelBPETokenizer(vocab_size=vocab_size).fit(training_texts)

    train_tokens = [tokenizer.encode(text, add_eos=True) for text in training_texts]
    valid_tokens = [tokenizer.encode(text, add_eos=True) for text in validation_texts]
    train_data = torch.tensor([token for sample in train_tokens for token in sample], dtype=torch.long)
    valid_data = torch.tensor([token for sample in valid_tokens for token in sample], dtype=torch.long)
    if train_data.numel() < 2 or valid_data.numel() < 2:
        raise ValueError("Corpus is too small to create training and validation sequences")

    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=context_length,
                       embedding_dim=embedding_dim, num_heads=num_heads,
                       num_layers=num_layers, dropout=dropout)
    model = GPTLanguageModel(config).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    interval = log_interval or max(1, steps // 5)
    print(f"Training custom GPT on {len(products)} product records ({selected_device.type})")
    for step in range(1, steps + 1):
        model.train()
        x, y = _sample_batch(train_data, batch_size, context_length, selected_device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % interval == 0 or step == steps:
            print(f"step {step}/{steps} training loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        validation_losses = []
        for _ in range(min(10, max(1, steps))):
            x, y = _sample_batch(valid_data, batch_size, context_length, selected_device)
            _, validation_loss = model(x, y)
            validation_losses.append(validation_loss.item())
    final_validation_loss = sum(validation_losses) / len(validation_losses)
    final_training_loss = float(loss.item())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    torch.save({
        "model_config": model.config_dict(),
        "model_state_dict": model.state_dict(),
        "training_steps": steps,
        "seed": seed,
        "training_loss": final_training_loss,
        "validation_loss": final_validation_loss,
    }, checkpoint_path)
    tokenizer.save(output_dir / "tokenizer.json")
    metadata = {
        "model_config": model.config_dict(), "steps": steps, "batch_size": batch_size,
        "context_length": context_length, "learning_rate": learning_rate,
        "seed": seed, "device": str(selected_device),
        "training_loss": final_training_loss, "validation_loss": final_validation_loss,
        "corpus_records": len(products),
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"final training loss {final_training_loss:.4f}; validation loss {final_validation_loss:.4f}")
    print(f"Saved checkpoint and tokenizer to {output_dir}")
    return metadata
