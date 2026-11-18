import torch

from app.model import CausalSelfAttention, GPTConfig, GPTLanguageModel


def test_causal_attention_masks_future_tokens():
    torch.manual_seed(2)
    attention = CausalSelfAttention(embedding_dim=16, num_heads=4, dropout=0, max_seq_len=8).eval()
    original = torch.randn(1, 6, 16)
    changed_future = original.clone()
    changed_future[:, 3:, :] = torch.randn_like(changed_future[:, 3:, :]) * 100
    output_original = attention(original)
    output_changed = attention(changed_future)
    assert torch.allclose(output_original[:, :3], output_changed[:, :3], atol=1e-6)
    assert not attention.causal_mask[0, 1]
    assert attention.causal_mask[1, 0]


def test_transformer_and_gpt_forward_shapes_and_loss():
    model = GPTLanguageModel(GPTConfig(vocab_size=280, block_size=12, embedding_dim=16,
                                       num_heads=4, num_layers=2, dropout=0))
    token_ids = torch.randint(0, 280, (2, 8))
    targets = torch.randint(0, 280, (2, 8))
    logits = model(token_ids)
    logits_with_loss, loss = model(token_ids, targets)
    assert logits.shape == (2, 8, 280)
    assert logits_with_loss.shape == logits.shape
    assert loss.ndim == 0 and torch.isfinite(loss)
