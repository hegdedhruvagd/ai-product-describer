# AI Product Describer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/drqszgsk7w/badge.svg)](https://ask.thili.ai/b/drqszgsk7w)

This project creates an AI Product Describer, a CLI tool for small business owners. It leverages a custom GPT model, built from scratch, to generate creative product descriptions by learning from existing product texts. The model processes raw text using byte-pair tokenization and embeddings, then applies causal self-attention to generate new, relevant descriptions.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that generates creative product descriptions for inventory by training a custom GPT model on examples of existing product texts.

**Done when.** The project is done when the CLI tool successfully takes product details as input and generates new, coherent product descriptions using a custom GPT model trained from scratch, applying byte-pair tokenization, embeddings, and causal self-attention.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Collect a small corpus of product descriptions for a specific category.**  
      Hint: Start by gathering a diverse set of product descriptions for a specific category.
- [ ] **2. Implement the byte-pair tokenizer and generate embeddings for product text.**  
      Hint: Implement the byte-pair tokenizer to process raw text and generate numerical embeddings.
- [ ] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Build the causal self-attention mechanism and combine it with other components to form a transformer block.
- [ ] **4. Train the GPT model using the collected product descriptions.**  
      Hint: Train your custom GPT model using the prepared product description corpus.
- [ ] **5. Implement a text generation feature that creates new product descriptions from a prompt.**  
      Hint: Develop a function to generate new text, starting from a given prompt, using your trained GPT model.
- [ ] **6. Develop a CLI for users to input product details and receive generated descriptions.**  
      Hint: Create a command-line interface to allow users to easily input details and receive generated descriptions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- (none)
