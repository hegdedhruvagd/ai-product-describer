"""Compact decoder-only Transformer components for the custom GPT."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


class TokenPositionEmbedding(nn.Module):
    """Learned token and position embeddings summed for each input position."""

    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = token_ids.shape
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError("Input sequence exceeds the configured context length")
        positions = torch.arange(sequence_length, device=token_ids.device)
        return self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with a future-token mask."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 128):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape
        if sequence_length > self.causal_mask.shape[0]:
            raise ValueError("Input sequence exceeds the attention context length")
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        reshape = lambda tensor: tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = (reshape(tensor) for tensor in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~self.causal_mask[:sequence_length, :sequence_length], float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """Pre-layer-normalized causal attention and MLP residual block."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(embedding_dim, num_heads, dropout, max_seq_len)
        self.norm_feed_forward = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.feed_forward(self.norm_feed_forward(x))


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class GPTLanguageModel(nn.Module):
    """A small decoder-only GPT that predicts the next token at every position."""

    def __init__(self, config: GPTConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = GPTConfig(**config)
        if config.embedding_dim % config.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.config = config
        self.embeddings = TokenPositionEmbedding(config.vocab_size, config.embedding_dim, config.block_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.embedding_dim, config.num_heads, config.dropout, config.block_size)
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        hidden = self.final_norm(self.blocks(self.embeddings(token_ids)))
        logits = self.language_head(hidden)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, *, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float = 1.0, eos_token_id: int | None = None) -> torch.Tensor:
        """Sample continuations autoregressively; filtering is applied to each next-token distribution."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        self.eval()
        output = token_ids
        for _ in range(max_new_tokens):
            context = output[:, -self.config.block_size:]
            logits = self(context)[:, -1, :] / temperature
            if top_k is not None:
                threshold = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                filtered = torch.full_like(logits, float("-inf"))
                logits = filtered.scatter(1, sorted_indices, sorted_logits)
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            output = torch.cat((output, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return output

    def config_dict(self) -> dict:
        return asdict(self.config)
