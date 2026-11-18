import torch

from app.model import TokenPositionEmbedding
from app.tokenizer import ByteLevelBPETokenizer


def test_bpe_training_encoding_and_decoding_round_trip():
    """Learn merges from product-like text and recover the original text."""
    corpus = ["soft oat cleanser", "oat milk moisturiser", "soft skin care"]
    tokenizer = ByteLevelBPETokenizer(vocab_size=280).fit(corpus)
    text = "soft oat cleanser"
    ids = tokenizer.encode(text)
    assert tokenizer.vocab_size > 257
    assert ids and all(0 <= token_id < tokenizer.vocab_size for token_id in ids)
    assert tokenizer.decode(ids) == text


def test_bpe_round_trips_unicode_punctuation_empty_and_unseen_text():
    """The byte vocabulary handles empty, unseen, Unicode and punctuation input."""
    tokenizer = ByteLevelBPETokenizer(vocab_size=280).fit(["plain text", "face lotion"])
    for text in ("", "New 🌿 balm — 50% & gentle!", "未知の製品🙂", "punctuation!?.,:;()[]"):
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_token_ids_remain_consistent_after_save_and_load(tmp_path):
    """Saved merge order reconstructs the same token IDs after loading."""
    tokenizer = ByteLevelBPETokenizer(vocab_size=280).fit(["amber fig serum", "fig leaf balm"])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ByteLevelBPETokenizer.load(path)
    for text in ("amber fig serum", "unseen 🫧 text"):
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.decode(loaded.encode(text)) == text
    assert loaded.vocab_size == tokenizer.vocab_size


def test_token_and_position_embeddings_have_expected_shape():
    """Embedding lookup combines token and position vectors at the expected width."""
    embeddings = TokenPositionEmbedding(vocab_size=300, embedding_dim=16, max_seq_len=12)
    output = embeddings(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert output.shape == (2, 3, 16)
