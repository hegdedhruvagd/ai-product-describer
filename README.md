# AskThili AI Product Describer

A small, offline command-line product describer built around a decoder-only GPT trained from scratch in PyTorch. It uses a local synthetic skincare and personal-care corpus; training and generation do not call an external language-model API or download a pretrained model.

## Architecture

- `data/skincare_corpus.jsonl` contains the training examples.
- `app/core.py` loads the JSONL records.
- `app/tokenizer.py` learns byte-pair merges directly from UTF-8 bytes. The 256 possible byte values make the tokenizer able to encode arbitrary Unicode text; unseen text falls back to those byte tokens.
- `app/model.py` provides learned token and position embeddings, multi-head scaled dot-product causal attention, pre-layer normalization, residual connections, feed-forward blocks, and next-token logits.
- `app/training.py` formats the records, trains the tokenizer, splits examples into training and validation sets, optimizes with AdamW and cross-entropy, and saves a checkpoint.
- `app/generation.py` loads the local checkpoint and samples tokens autoregressively with temperature, top-k and top-p controls.
- `app/cli.py` provides `hello`, `train` and `describe` commands.
- Model files are written under `artifacts/model/`, which is ignored by Git.

The old `app/llm.py` starter helper remains for compatibility with its plumbing test. The training and describe commands do not use it.

## Dataset

[`data/skincare_corpus.jsonl`](data/skincare_corpus.jsonl) is UTF-8 JSON Lines with 135 original, fictional products. Each record has `product_name`, `category`, `key_ingredients`, `target_skin_hair_type`, `size` and `description`. There are 15 examples in each category: face cleanser, moisturiser, sunscreen, serum, face mask, body lotion, shampoo, conditioner and lip balm.

Load the bundled examples with `load_product_corpus()` from `app.core`. The BPE vocabulary is learned from these product examples each time training runs, and the resulting merge rules are saved beside the model.

## Windows PowerShell setup

Install Python 3.12 and Git, open PowerShell in the repository, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No `.env` file, API key, network connection or model download is needed for training or inference.

## Train

Train the small CPU model with the built-in defaults:

```powershell
.\.venv\Scripts\python.exe -m app.cli train
```

The default run uses 200 optimizer steps, batch size 8, context length 64, learning rate `0.0003`, BPE vocabulary target 384, embedding width 64, four attention heads, two Transformer blocks, dropout 0.1 and seed 42. It prints training progress and final training and validation loss. The checkpoint, tokenizer merge rules and configuration are stored in `artifacts/model/`.

Example shorter run and custom output directory:

```powershell
.\.venv\Scripts\python.exe -m app.cli train --steps 20 --batch-size 4 --context-length 48 --seed 7 --output-dir artifacts/experiment
```

Available training options: `--steps`, `--batch-size`, `--context-length`, `--learning-rate`, `--vocab-size`, `--seed`, `--device` (`cpu`, `cuda` or `auto`), `--embedding-dim`, `--num-heads`, `--num-layers`, `--dropout` and `--output-dir`. CPU is the default; CUDA can be selected when available. Keep the embedding width divisible by the number of heads.

## Generate a description

Train first, then provide a product name and category. Ingredients, target type and size have usable defaults:

```powershell
.\.venv\Scripts\python.exe -m app.cli describe --name "Willow Oat Wash" --category "Face cleanser" --ingredients "oat extract, glycerin" --target "sensitive-feeling skin" --size "150 ml"
```

Generation options are `--model-dir`, `--max-new-tokens`, `--temperature`, `--top-k` (0 disables top-k), `--top-p` and `--device`. If no trained checkpoint exists, the CLI explains that `train` must be run first. The generator uses only the saved PyTorch model and tokenizer.

Check CLI usage or run its wiring check:

```powershell
.\.venv\Scripts\python.exe -m app.cli --help
.\.venv\Scripts\python.exe -m app.cli describe --help
.\.venv\Scripts\python.exe -m app.cli hello
```

## Tests

Run all tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests include corpus/schema coverage, tokenizer Unicode and persistence checks, causal attention behavior, GPT dimensions and loss, and a tiny CPU training/checkpoint/generation run. The tests do not run the full default training job.

## Limitations

This deliberately small synthetic corpus has only 135 examples across several product types. A compact CPU-trained GPT can learn the formatting and recurring vocabulary, but it will not have the fluency, factual reliability or broad knowledge of a large pretrained model. Generated descriptions may be repetitive or nonsensical; review product claims and ingredient details before use. The default run demonstrates the complete custom-model pipeline, not production-grade quality.
