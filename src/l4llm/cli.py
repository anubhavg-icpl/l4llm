"""Command-line interface for training and generating pickup lines."""

from __future__ import annotations

import argparse
import logging
import sys

from l4llm.data import build_char_mappings, load_dataset, prepare_sequences
from l4llm.generate import generate_text, get_random_seed
from l4llm.model import build_model, load_model, train_model


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train a new LSTM model on the pickup lines dataset."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    df = load_dataset(args.dataset)
    text = "\n".join(df["lines"])
    char_to_int, _int_to_char = build_char_mappings(text)

    logger.info(
        "Corpus: %d total characters, %d unique characters",
        len(text),
        len(char_to_int),
    )

    x, y, _ = prepare_sequences(text, char_to_int, seq_length=args.seq_length)
    model = build_model(args.seq_length, len(char_to_int))
    train_model(
        model,
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    logger.info("Training complete. Checkpoints saved to %s", args.checkpoint_dir)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate pickup lines using a trained model."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    df = load_dataset(args.dataset)
    text = "\n".join(df["lines"])
    char_to_int, int_to_char = build_char_mappings(text)
    n_vocab = len(char_to_int)

    _, _, data_x = prepare_sequences(text, char_to_int, seq_length=args.seq_length)

    model = load_model(args.weights, args.seq_length, n_vocab)

    if args.seed is not None:
        if args.seed < 0 or args.seed >= len(data_x):
            logger.error("Seed index %d out of range [0, %d)", args.seed, len(data_x))
            sys.exit(1)
        pattern = list(data_x[args.seed])
        seed_text = "".join(int_to_char[v] for v in pattern)
    else:
        pattern, seed_text = get_random_seed(data_x, int_to_char)

    logger.info("Seed: %s", seed_text[:80])

    result = generate_text(
        model,
        pattern,
        char_to_int,
        int_to_char,
        n_vocab,
        length=args.length,
        temperature=args.temperature,
    )

    print("\n" + "=" * 50)
    print("Generated pickup lines:")
    print("=" * 50)
    for line in result.split("\n"):
        stripped = line.strip()
        if stripped:
            print(stripped)
    print("=" * 50)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="l4llm",
        description="LSTM-based pickup lines generator",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="dataset/lines.csv",
        help="Path to the pickup lines CSV dataset",
    )
    train_parser.add_argument(
        "-s",
        "--seq-length",
        type=int,
        default=100,
        help="Length of character sequences for training",
    )
    train_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    train_parser.add_argument(
        "-c",
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    train_parser.set_defaults(func=cmd_train)

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate pickup lines")
    gen_parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="dataset/lines.csv",
        help="Path to the pickup lines CSV dataset (for char mappings)",
    )
    gen_parser.add_argument(
        "-w",
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights file",
    )
    gen_parser.add_argument(
        "-s",
        "--seq-length",
        type=int,
        default=100,
        help="Sequence length (must match training)",
    )
    gen_parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=1000,
        help="Number of characters to generate",
    )
    gen_parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower=conservative, higher=random)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed sequence index (random if not specified)",
    )
    gen_parser.set_defaults(func=cmd_generate)

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
