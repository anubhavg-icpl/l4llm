"""Text generation utilities using a trained LSTM model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


def sample_with_temperature(
    predictions: np.ndarray,
    temperature: float = 1.0,
) -> int:
    """Sample a character index from predictions using temperature scaling.

    Args:
        predictions: Raw probability distribution from the model.
        temperature: Controls randomness. Lower = more conservative,
            higher = more random. Must be > 0.

    Returns:
        Sampled character index.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, predictions, size=1)
    return int(np.argmax(probas))


def generate_text(
    model: tf.keras.Model,
    seed_sequence: list[int],
    char_to_int: dict[str, int],
    int_to_char: dict[int, str],
    n_vocab: int,
    length: int = 1000,
    temperature: float = 1.0,
) -> str:
    """Generate text character by character from a seed sequence.

    Args:
        model: Trained Keras model.
        seed_sequence: List of integer-encoded characters to seed generation.
        char_to_int: Character to integer mapping.
        int_to_char: Integer to character mapping.
        n_vocab: Size of the character vocabulary.
        length: Number of characters to generate.
        temperature: Sampling temperature.

    Returns:
        Generated text string.
    """
    pattern = list(seed_sequence)
    generated_chars: list[str] = []

    logger.info(
        "Generating %d characters with temperature=%.2f",
        length,
        temperature,
    )

    for _ in range(length):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(x, verbose=0)[0]

        if temperature != 0:
            index = sample_with_temperature(prediction, temperature)
        else:
            index = int(np.argmax(prediction))

        result = int_to_char[index]
        generated_chars.append(result)
        pattern.append(index)
        pattern = pattern[1:]

    return "".join(generated_chars)


def get_random_seed(
    data_x: list[list[int]],
    int_to_char: dict[int, str],
) -> tuple[list[int], str]:
    """Pick a random seed sequence from training data.

    Args:
        data_x: List of integer-encoded sequences.
        int_to_char: Integer to character mapping.

    Returns:
        Tuple of (seed_sequence, seed_text).
    """
    start = np.random.randint(0, len(data_x) - 1)
    pattern = list(data_x[start])
    seed_text = "".join(int_to_char[value] for value in pattern)
    return pattern, seed_text
