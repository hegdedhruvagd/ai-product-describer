# AskThili AI Product Describer

This command-line tool trains a small decoder-only GPT from scratch in PyTorch on fictional skincare and personal-care product examples. Training and inference use local code, the bundled corpus, and local model artifacts. They do not call an external LLM API or download a pretrained model.

## How it works

`data/skincare_corpus.jsonl` is read by `app/core.py`. `app/prompting.py` formats each record as product details followed by a description field. `app/training.py` splits records into training and validation sets, trains the tokenizer on the training records, and optimizes the model to predict the next token. The saved model and tokenizer are loaded by `app/generation.py` for local inference. `app/cli.py` exposes the workflow through `hello`, `train`, and `describe` commands.

The tokenizer in `app/tokenizer.py` starts with all 256 byte values and learns frequent adjacent-byte merges from the training text. Byte-level BPE was chosen because every UTF-8 input, including unseen names, punctuation, and Unicode characters, can be represented without an unknown-token fallback. Its vocabulary can stay small for this corpus, and the learned merge list is saved with the checkpoint.

The Transformer in `app/model.py` adds learned token and position embeddings, then applies pre-normalized multi-head self-attention and feed-forward blocks with residual connections. Attention scores are scaled by the square root of the head width. A lower-triangular mask leaves each position able to attend only to itself and earlier positions; future scores are set to negative infinity before softmax. This prevents later answer tokens from leaking into predictions for earlier positions during training.

Training uses the formatted product examples as token sequences, separated by an end-of-record token. At each position, the model predicts the next token and cross-entropy measures the difference from the actual next token. AdamW updates the model weights. A deterministic record-level split reserves examples for validation. Training defaults to CPU; `--device auto` or `--device cuda` can select CUDA when available. Checkpoints, tokenizer merges, and training configuration are stored locally under `artifacts/model/`, which Git ignores.

## Project origins and implementation

This project began from the AskThili starter scaffold. It retains the `app/` package layout, the `hello` CLI readiness command, the starter CI and pytest configuration, and the `milestone_1` entry point in `app/core.py`; the current corpus loader was added as that starter stub was replaced. The original scaffold also supplied the initial README, ignore rules, and plumbing tests, which were adapted as the project grew. The scaffold's API/LLM client and `.env` setup were later removed. The starter provided milestone prompts and stubs for the model work, not a working GPT implementation.

The project-specific implementation is the bundled fictional product corpus and formatting, byte-level BPE tokenizer, decoder-only Transformer with causal self-attention, next-token training and validation pipeline, local checkpoint-based generation, and the `train` and `describe` CLI workflows. The repository is therefore an implementation built on and adapted from the starter, not a repository written entirely from scratch.

## Dataset

The UTF-8 JSONL corpus contains 135 fictional records, with 15 examples in each of nine categories: face cleanser, moisturiser, sunscreen, serum, face mask, body lotion, shampoo, conditioner, and lip balm. Each record has `product_name`, `category`, `key_ingredients`, `target_skin_hair_type`, `size`, and `description`.

The corpus and default model are deliberately small so the complete pipeline can run on a CPU laptop. The model defaults to a 64-token context, 64-dimensional embeddings, four attention heads, and two Transformer blocks. The corpus is synthetic and has reused writing patterns; it is useful for exercising the training pipeline, not for teaching broad product knowledge.

See [the recorded evaluation](results/evaluation.md) for the default 200-step CPU run, measured losses, and generated examples.

## Windows PowerShell setup

Install Python 3.12 and Git, open PowerShell in the repository, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Once dependencies are installed, training and inference need no internet connection, `.env` file, API key, or model download.

## Train and describe

Train using the CPU defaults and save the checkpoint to `artifacts/model/`:

```powershell
.\.venv\Scripts\python.exe -m app.cli train
```

Then describe a product. Ingredients, target, and size can be omitted; the name and category are required:

```powershell
.\.venv\Scripts\python.exe -m app.cli describe --name "Willow Oat Wash" --category "Face cleanser" --ingredients "oat extract, glycerin" --target "sensitive-feeling skin" --size "150 ml"
```

The default training run uses 200 steps, batch size 8, context length 64, learning rate `0.0003`, vocabulary target 384, seed 42, embedding width 64, four heads, two layers, and dropout 0.1. Training options are `--steps`, `--batch-size`, `--context-length`, `--learning-rate`, `--vocab-size`, `--seed`, `--device` (`cpu`, `cuda`, `auto`), `--embedding-dim`, `--num-heads`, `--num-layers`, `--dropout`, and `--output-dir`.

Generation options are `--model-dir`, `--max-new-tokens`, `--temperature`, `--top-k` (use 0 to disable), `--top-p`, and `--device`. If the checkpoint is missing, the CLI explains that training must run first. Use `python -m app.cli --help` and `python -m app.cli describe --help` for usage details.

### Recorded short-run example

This output is from an actual 30-step CPU smoke run with a smaller one-layer model, greedy top-k sampling, and the product details above. It is an example of what that short run produced, not a guaranteed result:

```text
1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1
```

The output shows the main quality limitation observed in testing: a short run can complete training and checkpoint inference while still generating repetitive, unusable text. More training may change the output, but this small synthetic corpus does not establish that longer training will produce dependable product copy. Review generated text and all product or ingredient claims before use.

## Tests

Run all tests from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover corpus schema and category coverage, BPE round trips and persistence, embedding and model dimensions, causal masking, next-token loss, tiny CPU training, checkpoint loading, generation, and CLI errors and argument routing. They do not run the full default training job.
