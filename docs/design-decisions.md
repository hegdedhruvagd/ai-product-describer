# Design decisions and trade-offs

This note records the implementation that is present in this repository. It describes a compact learning project, not a production text-generation service.

## Corpus and prompt format

`data/skincare_corpus.jsonl` has 135 fictional products in nine skincare and personal-care categories. Training formats each record as product name, category, ingredients, intended skin or hair type, size, then `Description:` followed by the description. The same field order is used for inference so user input resembles the training prefix.

The examples are synthetic and reuse sentence structures. That keeps the dataset self-contained and suitable for a small training run, but limits its range and can encourage repetitive completions. It does not establish real product effects or ingredient claims.

## Byte-level BPE

`app/tokenizer.py` starts with token IDs 0–255 for individual bytes, reserves 256 for the end-of-record marker, then assigns learned merges from ID 257 upward. Fitting counts adjacent token pairs over the training records, selects the most frequent pair, and merges its non-overlapping occurrences. Ties are resolved by token-ID pair order, which makes the learned merge order reproducible for the same input text.

Encoding applies the saved merges in that order to UTF-8 bytes. This means unfamiliar names and Unicode still have a representation even when the tokenizer has not seen them. Decoding concatenates token bytes and decodes UTF-8 with replacement for a malformed byte sequence. The tokenizer JSON records its format version, target vocabulary, EOS ID, and ordered merge pairs; it does not store a separate expanded byte vocabulary because that is reconstructed from those merges.

The tokenizer is intentionally narrow: it has no pre-tokenization rules, normalization, or word boundary tokens. Training and generation therefore depend on byte sequences and merge order matching exactly. The vocabulary target is a limit; if no pair occurs at least twice, fitting stops with a smaller vocabulary.

## GPT model and causal attention

`app/model.py` sums learned token and position embeddings, then applies the configured number of pre-layer-normalized Transformer blocks. Each block projects hidden states into queries, keys, and values, splits them across heads, computes scaled dot-product scores, applies attention weights, and passes the result through a residual connection. A second residual connection surrounds a two-layer GELU feed-forward network. A final normalization and linear head produce logits for every vocabulary token at every position.

Each attention layer registers a lower-triangular boolean mask. Scores above the diagonal are replaced by negative infinity before softmax, so their attention probability is zero. The test perturbs future positions and checks that earlier attention outputs stay unchanged. This property is necessary for next-token training: a prediction at position *t* can use tokens through *t*, but cannot inspect target tokens after *t*.

The architecture is small by design: the default uses a 64-token context, 64-wide embeddings, four heads, and two blocks. These settings avoid the memory and compute cost of a large model on a laptop, while still exercising the full decoder-only GPT path. They are not evidence of high generation quality.

## Training and validation

`app/training.py` uses a seeded record-level 90/10 split. The BPE tokenizer is fit on the training partition only, then both partitions are encoded with those same merge rules. Encoded records are concatenated with EOS tokens, and each optimizer step samples fixed-length windows. A window can cross a record boundary; the EOS token marks that boundary but does not reset the model's state inside the window.

For each window, input IDs are the tokens except the final one and targets are the same window shifted one token to the left. The model returns per-position vocabulary logits; cross-entropy compares those logits to target IDs. AdamW updates the weights, and gradients are clipped to norm 1.0. Validation loss is measured on randomly sampled windows from the held-out records after training, using evaluation mode. The printed training loss is the final training batch's loss; the validation value is an average of up to ten validation batches, so neither is a corpus-wide quality score.

The run saves `model.pt` (configuration, weights, seed and loss values), `tokenizer.json` (merge rules), and `config.json` (training settings and summary) under the chosen output directory. These generated artifacts are local and ignored by Git. CPU is the default; CUDA can be requested when available. `auto` picks CUDA only when PyTorch reports it available.

## Inference and external services

`app/generation.py` loads the checkpoint and the matching tokenizer from disk. It encodes the user prompt, keeps only the most recent configured context when a prompt is longer than the model window, and samples one token at a time. Temperature scales logits; top-k and nucleus top-p can limit the candidate distribution; generation ends at EOS or the requested token limit.

Training and inference import PyTorch and the project's own modules. They do not import or call an external LLM client. The former `app/llm.py`, its settings test, API-only `.env.example`, and `httpx` dependency were removed because no active application path needed them. Installing Python packages may require network access, but once dependencies and source are available, model training and generation do not need network access.

## Observed limitations

- The 135 synthetic examples share phrasing patterns and cover a narrow set of categories.
- The small model and short training runs can complete successfully while generating repetitive fragments. In one recorded 30-step CPU smoke run, greedy generation returned repeated `1:` tokens rather than usable copy. This output is shown in the README as an observed example, not a guaranteed result.
- The unit test's two-step training run verifies checkpoint creation and local generation wiring only; it is not a model-quality test. The default 200-step training run is also a demonstration, not a validated quality threshold.
- Training batches are sampled windows from concatenated records, so some windows include the end of one example and beginning of another.
- There is no held-out human evaluation, factuality check, prompt injection defense, or product-claim verification. Treat generated copy as a draft and manually review it.
- The local verification environment used Python 3.15 because Python 3.12 was not installed there. The documented setup targets Python 3.12, but that exact interpreter was not exercised in that environment.
