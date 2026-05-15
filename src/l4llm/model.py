"""LSTM model definition and training utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf

logger = logging.getLogger(__name__)


def build_model(seq_length: int, n_vocab: int) -> tf.keras.Model:
    """Build a two-layer LSTM model for character-level text generation.

    Args:
        seq_length: Length of input character sequences.
        n_vocab: Size of the character vocabulary.

    Returns:
        Compiled Keras model.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(seq_length, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_vocab, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
    )

    logger.info("Model built: seq_length=%d, n_vocab=%d", seq_length, n_vocab)
    model.summary(print_fn=logger.info)
    return model


def train_model(
    model: tf.keras.Model,
    x: object,
    y: object,
    epochs: int = 50,
    batch_size: int = 64,
    checkpoint_dir: str | Path = "checkpoints",
) -> tf.keras.callbacks.History:
    """Train the model with checkpointing.

    Args:
        model: Compiled Keras model.
        x: Training input data.
        y: Training target data.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        checkpoint_dir: Directory to save model checkpoints.

    Returns:
        Training history.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filepath = str(checkpoint_dir / "weights-improvement-{epoch:02d}-{loss:.4f}.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor="loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )

    history = model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
    )
    return history


def load_model(weights_path: str | Path, seq_length: int, n_vocab: int) -> tf.keras.Model:
    """Load a model with pre-trained weights.

    Args:
        weights_path: Path to saved weights file.
        seq_length: Length of input sequences (must match training).
        n_vocab: Size of the character vocabulary (must match training).

    Returns:
        Model loaded with weights.
    """
    model = build_model(seq_length, n_vocab)
    model.load_weights(str(weights_path))
    logger.info("Loaded weights from %s", weights_path)
    return model
