"""Data loading and preprocessing utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the pickup lines CSV dataset.

    Args:
        path: Path to the CSV file containing pickup lines.

    Returns:
        DataFrame with a 'lines' column of lowercase strings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, encoding="utf-8", header=None, names=["lines"], quotechar='"')
    df = df.dropna(subset=["lines"])
    df["lines"] = df["lines"].astype(str).str.lower().str.strip()
    df = df[df["lines"].str.len() > 0]
    logger.info("Loaded %d lines from %s", len(df), path)
    return df


def build_char_mappings(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-to-integer and integer-to-character mappings.

    Args:
        text: The full text corpus.

    Returns:
        Tuple of (char_to_int, int_to_char) dictionaries.
    """
    chars = sorted(set(text))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_int, int_to_char


def prepare_sequences(
    text: str,
    char_to_int: dict[str, int],
    seq_length: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare input-output sequence pairs for training.

    Args:
        text: The full text corpus.
        char_to_int: Character to integer mapping.
        seq_length: Length of each input sequence.

    Returns:
        Tuple of (X, y) arrays ready for model training.
    """
    n_chars = len(text)
    n_vocab = len(char_to_int)

    data_x: list[list[int]] = []
    data_y: list[int] = []

    for i in range(0, n_chars - seq_length, 1):
        seq_in = text[i : i + seq_length]
        seq_out = text[i + seq_length]
        data_x.append([char_to_int[char] for char in seq_in])
        data_y.append(char_to_int[seq_out])

    n_patterns = len(data_x)
    logger.info("Prepared %d training patterns (seq_length=%d)", n_patterns, seq_length)

    # Reshape to [samples, time_steps, features] and normalize
    x = np.reshape(data_x, (n_patterns, seq_length, 1)) / float(n_vocab)
    y = tf.keras.utils.to_categorical(data_y, num_classes=n_vocab)

    return x, y, data_x
