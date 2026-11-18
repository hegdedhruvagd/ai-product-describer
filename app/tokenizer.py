"""A small byte-level BPE tokenizer implemented without pretrained assets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class ByteLevelBPETokenizer:
    """Learn frequent byte-pair merges; UTF-8 bytes keep the base vocabulary complete."""

    BYTE_VOCAB_SIZE = 256
    EOS_TOKEN_ID = 256
    FIRST_MERGE_ID = 257

    def __init__(self, vocab_size: int = 384):
        if vocab_size < self.FIRST_MERGE_ID:
            raise ValueError(f"vocab_size must be at least {self.FIRST_MERGE_ID}")
        self.requested_vocab_size = vocab_size
        self.merges: list[tuple[int, int]] = []
        self._token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._token_bytes[self.EOS_TOKEN_ID] = b""

    @property
    def vocab_size(self) -> int:
        return self.FIRST_MERGE_ID + len(self.merges)

    def fit(self, texts: Iterable[str]) -> "ByteLevelBPETokenizer":
        """Learn up to the configured number of merges from the supplied text."""
        sequences = [list(text.encode("utf-8")) for text in texts if text]
        self.merges = []
        self._token_bytes = {i: bytes([i]) for i in range(256)}
        self._token_bytes[self.EOS_TOKEN_ID] = b""

        while self.vocab_size < self.requested_vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            pair, count = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
            if count < 2:
                break

            new_id = self.FIRST_MERGE_ID + len(self.merges)
            self.merges.append(pair)
            self._token_bytes[new_id] = self._token_bytes[pair[0]] + self._token_bytes[pair[1]]
            sequences = [self._merge_pair(sequence, pair, new_id) for sequence in sequences]
        return self

    @staticmethod
    def _merge_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                merged.append(new_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        """Encode arbitrary text. Unseen characters are represented by their UTF-8 bytes."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        sequence = list(text.encode("utf-8"))
        for merge_index, pair in enumerate(self.merges):
            sequence = self._merge_pair(sequence, pair, self.FIRST_MERGE_ID + merge_index)
        if add_eos:
            sequence.append(self.EOS_TOKEN_ID)
        return sequence

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode token IDs, replacing malformed UTF-8 sequences safely."""
        byte_parts = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.EOS_TOKEN_ID:
                continue
            if token_id not in self._token_bytes:
                raise ValueError(f"Token id {token_id} is outside this tokenizer's vocabulary")
            byte_parts.append(self._token_bytes[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Save versioned merge rules as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "byte-level-bpe",
            "version": 1,
            "requested_vocab_size": self.requested_vocab_size,
            "eos_token_id": self.EOS_TOKEN_ID,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPETokenizer":
        """Restore the exact merge ordering and token IDs from a saved JSON file."""
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load tokenizer from {path}: {exc}") from exc
        if payload.get("format") != "byte-level-bpe" or payload.get("version") != 1:
            raise ValueError(f"Unsupported tokenizer format in {path}")
        tokenizer = cls(int(payload["requested_vocab_size"]))
        if payload.get("eos_token_id") != cls.EOS_TOKEN_ID:
            raise ValueError("Tokenizer EOS token ID does not match this implementation")
        for index, raw_pair in enumerate(payload.get("merges", [])):
            if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                raise ValueError(f"Invalid BPE merge at index {index}")
            pair = (int(raw_pair[0]), int(raw_pair[1]))
            new_id = cls.FIRST_MERGE_ID + index
            if pair[0] not in tokenizer._token_bytes or pair[1] not in tokenizer._token_bytes:
                raise ValueError(f"BPE merge {index} refers to a token that has not been defined")
            tokenizer.merges.append(pair)
            tokenizer._token_bytes[new_id] = tokenizer._token_bytes[pair[0]] + tokenizer._token_bytes[pair[1]]
        if tokenizer.vocab_size > tokenizer.requested_vocab_size:
            raise ValueError("Saved tokenizer has more merges than its declared vocabulary")
        return tokenizer
