# Evaluation report

## Run and measured results

This was the actual default 200-step training run, followed by three CLI generations from the saved checkpoint. It ran on CPU with seed 42. The environment was Python 3.15.2, PyTorch 2.15.0+cpu (distribution version 2.15.0), and pytest 9.2.0. This is a different Python version from the README's suggested 3.12 setup. NumPy was unavailable; PyTorch printed a NumPy initialization warning, but training and generation completed.

| Step | Sampled training loss |
| ---: | ---: |
| 1 | 6.050895 |
| 40 | 5.561557 |
| 80 | 4.897923 |
| 120 | 4.659559 |
| 160 | 4.572595 |
| 200 | 4.484279 |

Final training loss was **4.484279** (the final training batch). Validation loss was **4.472417**, calculated as the mean of 10 randomly sampled validation-window batch losses after training. It is not a full-corpus validation metric. The training loop took **5.986 seconds** by `time.perf_counter`; this excludes Python startup, validation, and artifact writing.

### Exact training configuration

| Setting | Value |
| --- | ---: |
| Steps / seed / device | 200 / 42 / CPU |
| Batch size / context length | 8 / 64 |
| Learning rate / optimizer | 0.0003 / AdamW |
| Target vocabulary | 384 |
| Embedding width / heads / Transformer layers | 64 / 4 / 2 |
| Dropout | 0.1 |
| Corpus / split | 135 records / seeded split: 121 training and 14 validation records |

The run metadata, including loss samples, was persisted in the ignored local checkpoint directory at `artifacts/model/config.json`. The generated model weights are not included in this repository.

## Generated examples

These are the literal outputs returned by three `describe` CLI calls on this checkpoint. Generation used the CLI defaults for temperature (0.8), top-k (40), and top-p (0.9), with `--max-new-tokens 60`. Sampling is not seeded by the CLI, so later runs can produce different text.

**Willow Oat Wash** — Face cleanser; oat extract, glycerin; sensitive-feeling skin; 150 ml

```text
yda  las   vamerss, soomsfts yeriserodds a ss  . r s yerofet ss .  t ssofsinineaer
```

**Sunveil Mineral SPF** — Sunscreen; zinc oxide, aloe; combination skin; 50 ml

```text
a sinlaisbosereenx. a enif Sen. mss t fucofs . thderss ows   erpcain , pl u lip a  er
```

**Citrus Grove Conditioner** — Conditioner; coconut oil, citrus peel; dry hair; 250 ml

```text
a brlifmmrowgda s and yi serteaenerpys sysps inls siyastpensenos ointsis, enoinp s
```

## Assessment

All three outputs are highly garbled, with recurring fragments and malformed letter sequences; none forms a coherent product description. The checkpoint is not practically usable for writing copy without substantial human rewriting. Do not treat its text as a factual product or ingredient claim. This small model trained briefly on a narrow synthetic corpus, and these results do not show production readiness.

The unit tests and short CPU training/generation smoke checks exercise tokenizer, model, checkpoint, and CLI plumbing. They are smoke tests, not evidence of language quality. The default run and outputs above are the quality evaluation; they show poor quality.

## Reproduce with PowerShell

From the repository root, install the exact direct dependency versions recorded for this run, then train using the same CLI defaults. This overwrites the local default checkpoint directory:

```powershell
python -m pip install -r requirements.txt
python -m app.cli train
```

The default command sets 200 steps, seed 42, CPU, batch size 8, context length 64, learning rate 0.0003, vocabulary target 384, embedding width 64, four heads, two layers, and dropout 0.1. Training loss samples and training-loop runtime are written to `artifacts/model/config.json`.

Generate comparable examples from the resulting local checkpoint with:

```powershell
python -m app.cli describe --name "Willow Oat Wash" --category "Face cleanser" --ingredients "oat extract, glycerin" --target "sensitive-feeling skin" --size "150 ml" --max-new-tokens 60
python -m app.cli describe --name "Sunveil Mineral SPF" --category "Sunscreen" --ingredients "zinc oxide, aloe" --target "combination skin" --size "50 ml" --max-new-tokens 60
python -m app.cli describe --name "Citrus Grove Conditioner" --category "Conditioner" --ingredients "coconut oil, citrus peel" --target "dry hair" --size "250 ml" --max-new-tokens 60
```

Losses may vary with runtime or hardware details. Generation is stochastic and can vary between calls. The machine-readable record of this evaluation is [evaluation.json](evaluation.json).
