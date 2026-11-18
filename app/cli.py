"""CLI entry point: ``python -m app.cli --help``."""

import argparse
import sys
from pathlib import Path

from app.generation import generate_text
from app.prompting import format_product_prompt
from app.training import MODEL_DIR, train_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-product-describer",
        description="Train a small custom PyTorch GPT on fictional product descriptions and generate new ones offline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("hello", help="check that the application is ready")

    train = subparsers.add_parser("train", help="train the custom GPT using the bundled product corpus")
    train.add_argument("--output-dir", type=Path, default=MODEL_DIR, help=f"model output directory (default: {MODEL_DIR})")
    train.add_argument("--steps", type=int, default=200, help="number of optimizer steps (default: 200)")
    train.add_argument("--batch-size", type=int, default=8, help="training batch size (default: 8)")
    train.add_argument("--context-length", type=int, default=64, help="maximum token context (default: 64)")
    train.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    train.add_argument("--vocab-size", type=int, default=384, help="target byte-BPE vocabulary size")
    train.add_argument("--seed", type=int, default=42, help="random seed")
    train.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="training device (default: cpu)")
    train.add_argument("--embedding-dim", type=int, default=64, help="model embedding width")
    train.add_argument("--num-heads", type=int, default=4, help="attention heads")
    train.add_argument("--num-layers", type=int, default=2, help="transformer blocks")
    train.add_argument("--dropout", type=float, default=0.1, help="dropout probability")

    describe = subparsers.add_parser("describe", help="generate a product description from product details")
    describe.add_argument("--name", required=True, help="product name")
    describe.add_argument("--category", required=True, help="product category")
    describe.add_argument("--ingredients", default="", help="comma-separated key ingredients")
    describe.add_argument("--target", default="all skin and hair types", help="target skin or hair type")
    describe.add_argument("--size", default="not specified", help="product size, for example 50 ml")
    describe.add_argument("--model-dir", type=Path, default=MODEL_DIR, help=f"trained model directory (default: {MODEL_DIR})")
    describe.add_argument("--max-new-tokens", type=int, default=120, help="maximum tokens to generate")
    describe.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    describe.add_argument("--top-k", type=int, default=40, help="sample only from the top K tokens; use 0 to disable")
    describe.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling threshold")
    describe.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="inference device")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("AI Product Describer is ready. Train the custom GPT with `python -m app.cli train`.")
        return
    if args.command == "train":
        try:
            train_model(
                output_dir=args.output_dir, steps=args.steps, batch_size=args.batch_size,
                context_length=args.context_length, learning_rate=args.learning_rate,
                vocab_size=args.vocab_size, seed=args.seed, device=args.device,
                embedding_dim=args.embedding_dim, num_heads=args.num_heads,
                num_layers=args.num_layers, dropout=args.dropout,
            )
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        return
    if args.command == "describe":
        if not (args.model_dir / "model.pt").is_file():
            print(f"No trained model found in {args.model_dir}. Run `python -m app.cli train` first.", file=sys.stderr)
            raise SystemExit(1)
        product = {
            "product_name": args.name,
            "category": args.category,
            "key_ingredients": [part.strip() for part in args.ingredients.split(",") if part.strip()],
            "target_skin_hair_type": args.target,
            "size": args.size,
        }
        prompt = format_product_prompt(product)
        try:
            description = generate_text(
                prompt, model_dir=args.model_dir, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k or None,
                top_p=args.top_p, device=args.device,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(description or "The model did not produce a description. Try again or increase --max-new-tokens.")


if __name__ == "__main__":
    main()
