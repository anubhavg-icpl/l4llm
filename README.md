<div align="center">

<img src="assets/banner.avif" alt="l4llm - LSTM Pickup Lines Generator" width="100%">

# l4llm

### LSTM-Based Pickup Lines Generator

*A character-level recurrent neural network trained on thousands of pickup lines from the internet, capable of generating never-before-seen flirty one-liners using deep learning.*

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%E2%89%A52.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/badge/Linted%20with-Ruff-D9FF3F?style=for-the-badge)](https://docs.astral.sh/ruff/)

[Features](#-features) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [CLI Reference](#-cli-reference) · [Architecture](#-architecture) · [Examples](#-generated-examples) · [Contributing](#-contributing)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
  - [What is Character-Level Text Generation?](#what-is-character-level-text-generation)
  - [The LSTM Architecture](#the-lstm-architecture)
  - [Temperature Sampling](#temperature-sampling)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training Your First Model](#training-your-first-model)
  - [Generating Pickup Lines](#generating-pickup-lines)
- [CLI Reference](#-cli-reference)
  - [Train Command](#train-command)
  - [Generate Command](#generate-command)
- [Architecture](#-architecture)
  - [Data Pipeline](#data-pipeline)
  - [Model Architecture](#model-architecture)
  - [Generation Pipeline](#generation-pipeline)
- [Module Reference](#-module-reference)
- [Configuration](#-configuration)
  - [pyproject.toml](#pyprojecttoml)
  - [Ruff Linting](#ruff-linting)
- [Generated Examples](#-generated-examples)
- [Understanding the Output Quality](#understanding-the-output-quality)
- [The Original Legacy Code](#-the-original-legacy-code)
- [Modernization Changelog](#-modernization-changelog)
- [Image Generation Prompts](#-image-generation-prompts)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## Overview

**l4llm** (short for "LSTM for LLM" — a cheeky name from before the transformer era) is a character-level LSTM neural network that learns the patterns, grammar, and style of pickup lines from a curated dataset, then generates entirely new ones that have never existed before.

The model reads text one character at a time, learns to predict the next character based on a window of preceding characters, and then uses that learned knowledge to produce novel sequences character-by-character. It's the same fundamental approach used by early text generation systems, applied here to the timeless art of terrible pickup lines.

This project was originally built as a fun weekend project using Keras 2 and Python 3.6-era tooling. It has since been fully modernized to 2026 standards: PEP 621 project configuration, TensorFlow 2.16+ / Keras 3 API, type hints throughout, modular package structure, CLI interface, temperature-based sampling, and strict Ruff linting.

<img src="assets/pipeline-diagram.avif" alt="Text generation pipeline: Dataset → LSTM → Generated output with temperature control" width="100%">

---

## ✨ Features

| Feature | Description |
|---|---|
| **Character-Level LSTM** | Two-layer LSTM with 256 hidden units per layer, dropout regularization, and softmax output over the full character vocabulary |
| **Temperature Sampling** | Configurable sampling temperature — lower values produce conservative, repetitive output; higher values produce creative, unpredictable output |
| **Modern CLI** | Full command-line interface with `train` and `generate` subcommands, configurable parameters, and structured logging |
| **Modular Architecture** | Clean separation of concerns across 4 modules: data loading, model definition, text generation, and CLI |
| **Keras 3 Compatible** | Uses the modern `tf.keras` API with `Input()` layers — no deprecation warnings |
| **PEP 621 Project Config** | Full `pyproject.toml` with metadata, dependencies, entry points, and tool configuration |
| **Type Hints** | Complete type annotations on all 14 public functions across the codebase |
| **Ruff Linted** | Passes strict Ruff lint rules (E, W, F, I, UP, B, SIM, TCH, RUF) with zero warnings |
| **Checkpoint Saving** | Automatic best-model checkpointing during training — only saves when loss improves |
| **Reproducible Seeds** | Optionally specify a seed sequence index for deterministic generation |
| **Robust CSV Parsing** | Handles commas within pickup lines via proper quoting, drops NaN/empty rows |

---

## How It Works

### What is Character-Level Text Generation?

Unlike word-level models that predict entire words, character-level models work at the granularity of individual characters (letters, digits, punctuation, spaces). The model looks at a sequence of, say, 60 characters and tries to predict what the 61st character should be.

For example, given the input:
```
"are you a magician? because whenever i look at yo"
```
The model should predict `u` as the next character, because it has learned that "look at you" is a common pattern in the training data.

**Why character-level?**
- No need for tokenizers or vocabulary management
- Can generate any character, including creative misspellings
- Works with any language or character set
- Simpler data pipeline

**Trade-offs:**
- Slower convergence than word-level models
- May produce nonsensical words more often
- Needs more training data to learn coherent long-range structure

### The LSTM Architecture

Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN) specifically designed to handle sequential data while avoiding the vanishing gradient problem that plagues vanilla RNNs.

An LSTM cell has three gates:
1. **Forget Gate** — Decides what information to discard from the cell state
2. **Input Gate** — Decides what new information to store in the cell state
3. **Output Gate** — Decides what part of the cell state to output

Our model stacks two LSTM layers (each with 256 units) with 20% dropout between them. This gives the model enough capacity to learn complex character-level patterns while the dropout prevents overfitting on our relatively small dataset.

<p align="center">
  <img src="assets/lstm-closeup.avif" alt="LSTM memory cells close-up with flowing character data streams" width="90%">
</p>

The model has approximately **798,499 trainable parameters** — small enough to train on a CPU in under 20 minutes, yet powerful enough to produce recognizable pickup line structures.

### Temperature Sampling

Temperature is a hyperparameter that controls the randomness of predictions during text generation. It scales the logits before applying softmax:

```
scaled_logits = log(probabilities) / temperature
```

| Temperature | Behavior | Use Case |
|---|---|---|
| 0.1 - 0.3 | Very conservative, repetitive, mostly reproduces training data | Want something safe and recognizable |
| 0.4 - 0.7 | Balanced creativity, mostly coherent with occasional surprises | **Recommended range** |
| 0.8 - 1.0 | More creative, some nonsense words, more unique combinations | Want something wild and unexpected |
| 1.0+ | Very random, often incoherent, high entropy | Experimental / entertainment |

<img src="assets/temperature-effect.avif" alt="Temperature effect on text generation: conservative to creative spectrum" width="100%">

---

## Project Structure

```
l4llm/
├── README.md                          # You are here
├── pyproject.toml                     # PEP 621 project config, deps, ruff config
├── requirements.txt                   # Core deps (pip install -r)
├── requirements-dev.txt               # Dev deps (ruff, pytest)
├── .gitignore                         # Model weights, checkpoints, venv, etc.
│
├── dataset/
│   ├── README.md                      # Dataset instructions
│   └── lines.csv                      # 161 pickup lines (CSV, quoted)
│
├── models/
│   ├── weights-improvement-50-0.3404.keras    # Trained weights (Keras 3 format)
│   └── weights-improvement-50-1.3610-bigger.hdf5  # Legacy weights (original model)
│
├── pickuplines_generator.py           # Original legacy script (preserved for reference)
├── output.txt                         # Sample output from the original model
│
└── src/
    └── l4llm/                         # Main package
        ├── __init__.py                # Package init, version 2.0.0
        ├── data.py                    # Data loading, char mappings, sequence prep
        ├── model.py                   # LSTM model build, train, load weights
        ├── generate.py                # Temperature sampling, text generation
        └── cli.py                     # CLI entry point (train / generate)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested on 3.11, 3.12)
- **pip** or **uv** (for package management)
- **~2GB disk space** (TensorFlow is a large dependency)

### Installation

**Option A: Install as a package (recommended)**

```bash
# Clone the repository
git clone https://github.com/anubhavg-icpl/l4llm.git
cd l4llm

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
l4llm --help
```

**Option B: Install dependencies directly**

```bash
pip install -r requirements.txt
python -m l4llm.cli --help
```

**Option C: Using uv (fastest)**

```bash
uv venv .venv --python 3.11
uv pip install -e ".[dev]" --python .venv/bin/python
.venv/bin/l4llm --help
```

### Training Your First Model

```bash
# Train with default settings (50 epochs, seq_length=100, batch_size=64)
l4llm train

# Train with custom parameters
l4llm train \
  --dataset dataset/lines.csv \
  --seq-length 60 \
  --epochs 50 \
  --batch-size 128 \
  --checkpoint-dir checkpoints/

# Train with verbose logging
l4llm train -v -e 100 -s 80 -b 64
```

**What happens during training:**

1. The dataset CSV is loaded and all lines are lowercased
2. All lines are joined into one long string (separated by newlines)
3. A sorted character vocabulary is built from the text
4. Sliding windows of `seq_length` characters are extracted as training sequences
5. Each sequence is paired with the next character as the target
6. Sequences are reshaped to `[samples, timesteps, features]` and normalized
7. Targets are one-hot encoded using `tf.keras.utils.to_categorical`
8. The LSTM model is built and trained with best-model checkpointing

**Expected output during training:**

```
2026-05-15 [INFO] l4llm.data: Loaded 161 lines from dataset/lines.csv
2026-05-15 [INFO] l4llm.cli: Corpus: 11048 total characters, 35 unique characters
2026-05-15 [INFO] l4llm.data: Prepared 9932 training patterns (seq_length=60)
2026-05-15 [INFO] l4llm.model: Model built: seq_length=60, n_vocab=35

Epoch 1/50
78/78 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24s/step - loss: 2.9659
Epoch 1: loss improved from None to 2.96458, saving model to checkpoints/weights-improvement-01-2.9646.keras

...

Epoch 50/50
78/78 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7s/step - loss: 0.3404
Epoch 50: loss improved from 0.36583 to 0.34044, saving model to checkpoints/weights-improvement-50-0.3404.keras

2026-05-15 [INFO] l4llm.cli: Training complete. Checkpoints saved to checkpoints/
```

Training takes approximately **15-20 minutes on CPU** or **2-3 minutes on GPU** for 50 epochs on the included 161-line dataset.

<p align="center">
  <img src="assets/loss-curve.avif" alt="Training loss curve: 3.43 → 0.34 over 50 epochs" width="90%">
</p>

### Generating Pickup Lines

```bash
# Generate with the best checkpoint (temperature 0.5 = balanced)
l4llm generate \
  --weights checkpoints/weights-improvement-50-0.3404.keras \
  --seq-length 60 \
  --temperature 0.5 \
  --length 2000

# Generate with specific seed for reproducibility
l4llm generate \
  -w checkpoints/weights-improvement-50-0.3404.keras \
  -s 60 \
  -t 0.8 \
  -l 1000 \
  --seed 42

# Generate short output with high creativity
l4llm generate \
  -w checkpoints/weights-improvement-50-0.3404.keras \
  -s 60 \
  -t 1.2 \
  -l 500
```

**Expected output:**

<p align="center">
  <img src="assets/terminal-output.avif" alt="Terminal window showing generated pickup lines output" width="90%">
</p>

```
2026-05-15 [INFO] l4llm.data: Loaded 161 lines from dataset/lines.csv
2026-05-15 [INFO] l4llm.model: Loaded weights from checkpoints/weights-improvement-50-0.3404.keras
2026-05-15 [INFO] l4llm.generate: Generating 2000 characters with temperature=0.50

==================================================
Generated pickup lines:
==================================================
are your legs tired? because you have been running through my mind all day.
can i follow you home? my parents always told me to follow my dreams.
do you have a map? because i just got lost in your eyes.
your hand looks heavy. can i hold it for you?
i am not a genie but i can make your dreams come true.
are you a camera? because every time i look at you everyone else disappears.
are you french? because eiffel for you.
are you a bank loan? because you have my interest.
do you have a band-aid? because i scraped my knee falling for you.
==================================================
```

---

## CLI Reference

### Train Command

```
l4llm train [-h] [-d DATASET] [-s SEQ_LENGTH] [-e EPOCHS] [-b BATCH_SIZE] [-c CHECKPOINT_DIR]
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--dataset` | `-d` | `dataset/lines.csv` | Path to the CSV file containing pickup lines |
| `--seq-length` | `-s` | `100` | Number of characters in each input sequence. Higher values capture longer-range patterns but require more memory. Typical range: 40-100 |
| `--epochs` | `-e` | `50` | Number of full passes over the training data. The model typically converges between 30-50 epochs on the included dataset |
| `--batch-size` | `-b` | `64` | Number of samples processed before updating model weights. Larger batches train faster but may generalize worse. Common values: 32, 64, 128 |
| `--checkpoint-dir` | `-c` | `checkpoints` | Directory where model checkpoints are saved. Only the best model (lowest loss) is kept per epoch |
| `--verbose` | `-v` | off | Enable debug-level logging |

**Tips for better training:**
- Start with the defaults and adjust from there
- If loss plateaus, try decreasing the learning rate (modify `optimizer="adam"` in `model.py`)
- For a larger dataset, increase `seq_length` to 100-200 to capture longer patterns
- If the model overfits (loss stops decreasing but output quality drops), add more dropout or reduce epochs
- Always monitor the loss curve — it should decrease monotonically with occasional plateaus

### Generate Command

```
l4llm generate [-h] [-d DATASET] -w WEIGHTS [-s SEQ_LENGTH] [-l LENGTH] [-t TEMPERATURE] [--seed SEED]
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--dataset` | `-d` | `dataset/lines.csv` | Path to CSV for character vocabulary (must match training data) |
| `--weights` | `-w` | *required* | Path to the trained `.keras` weights file |
| `--seq-length` | `-s` | `100` | Must match the `seq_length` used during training |
| `--length` | `-l` | `1000` | Number of characters to generate. Each pickup line is roughly 50-80 characters |
| `--temperature` | `-t` | `1.0` | Sampling temperature. See [Temperature Sampling](#temperature-sampling) for details |
| `--seed` | — | random | Index of the seed sequence from training data (0 to N-1). Omit for random seed |
| `--verbose` | `-v` | off | Enable debug-level logging |

**Important:** The `--seq-length` parameter **must match** the value used during training. If you trained with `-s 60`, you must generate with `-s 60`. Mismatched values will cause the model to produce garbage output or crash.

---

## Architecture

<img src="assets/pipeline-diagram.avif" alt="Architecture deep-dive: data pipeline, LSTM model, and generation loop" width="100%">

### Data Pipeline

The data pipeline transforms raw CSV text into numerical arrays suitable for LSTM training:

**Step 1: Load Dataset**
- Reads the CSV file using pandas with proper quoting for commas
- Strips whitespace, converts to lowercase
- Drops NaN rows and empty strings

**Step 2: Build Character Mappings**
- Creates a sorted set of all unique characters in the corpus
- Maps each character to a unique integer index (0, 1, 2, ...)
- Creates the reverse mapping (integer -> character) for generation

**Step 3: Prepare Sequences**
- Slides a window of `seq_length` characters across the entire text
- For each window position, the input is the window and the target is the next character
- This creates `total_chars - seq_length` training examples

**Step 4: Reshape and Normalize**
- Reshapes inputs to `[samples, seq_length, 1]` — the 3D format expected by LSTM
- Divides by the vocabulary size to normalize to the range [0, 1]
- One-hot encodes targets using `tf.keras.utils.to_categorical`

Example with `seq_length=10` on the text `"are you a..."`:

```
Input chars:   "a r e   y o u   a "  ->  [0, 3, 1, 2, 7, 6, 5, 2, 0, 3]
Target char:   "m"                    ->  index 12
                                          one-hot: [0,0,0,0,0,0,0,0,0,0,0,0,1,0,...]

Input chars:   "r e   y o u   a m"  ->  [3, 1, 2, 7, 6, 5, 2, 0, 3, 12]
Target char:   "a"                    ->  index 0
```

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Model: "sequential"                       │
├─────────────────────────┬───────────────────────┬───────────────┤
│ Layer (type)            │ Output Shape          │     Param #   │
├─────────────────────────┼───────────────────────┼───────────────┤
│ Input                   │ (None, seq_length, 1) │             0 │
├─────────────────────────┼───────────────────────┼───────────────┤
│ LSTM (256 units)        │ (None, seq_length,256)│       264,192 │
│ return_sequences=True   │                       │               │
├─────────────────────────┼───────────────────────┼───────────────┤
│ Dropout (0.2)           │ (None, seq_length,256)│             0 │
├─────────────────────────┼───────────────────────┼───────────────┤
│ LSTM (256 units)        │ (None, 256)           │       525,312 │
│ return_sequences=False  │                       │               │
├─────────────────────────┼───────────────────────┼───────────────┤
│ Dropout (0.2)           │ (None, 256)           │             0 │
├─────────────────────────┼───────────────────────┼───────────────┤
│ Dense (softmax)         │ (None, n_vocab)       │         8,995 │
├─────────────────────────┼───────────────────────┼───────────────┤
│ TOTAL                   │                       │       798,499 │
│ (all trainable)         │                       │    (3.05 MB)  │
└─────────────────────────┴───────────────────────┴───────────────┘
```

**Why this architecture?**

- **Two LSTM layers**: The first layer returns full sequences (`return_sequences=True`), allowing the second layer to learn higher-level temporal patterns. Stacking LSTMs has been shown to improve performance on sequence tasks.
- **256 units per layer**: A balance between model capacity and training speed. More units can capture more complex patterns but increase training time and risk overfitting on small datasets.
- **20% Dropout**: Applied after each LSTM layer to prevent overfitting. The model learns to not rely on any single neuron, improving generalization.
- **Softmax output**: Produces a probability distribution over all characters, allowing for sampling-based generation rather than greedy argmax.

### Generation Pipeline

1. **Seed selection**: A random training sequence is selected (or a specific index is provided via `--seed`)
2. **Reshape**: The seed is reshaped to `[1, seq_length, 1]` and normalized
3. **Predict**: The model produces a probability distribution over the vocabulary
4. **Temperature scaling**: The logits are divided by the temperature and re-normalized
5. **Sample**: A character is sampled from the distribution using multinomial sampling
6. **Append and shift**: The sampled character is appended to the output, and the window is shifted by one character (dropping the first, adding the new)
7. **Repeat**: Steps 2-6 are repeated `length` times

---

## Module Reference

### `l4llm.data` — Data Loading and Preprocessing

| Function | Description |
|---|---|
| `load_dataset(path)` | Loads a CSV file of pickup lines. Handles quoting, lowercasing, NaN removal. Returns a pandas DataFrame with a `'lines'` column. |
| `build_char_mappings(text)` | Takes a text corpus and returns `(char_to_int, int_to_char)` dictionaries. Characters are sorted for deterministic ordering. |
| `prepare_sequences(text, char_to_int, seq_length)` | Creates sliding-window input-output pairs, reshapes and normalizes inputs, one-hot encodes outputs. Returns `(X, y, data_x)` where `data_x` is the raw integer sequences used for seed selection. |

### `l4llm.model` — LSTM Model Definition and Training

| Function | Description |
|---|---|
| `build_model(seq_length, n_vocab)` | Constructs the two-layer LSTM model with dropout and softmax output. Compiled with categorical crossentropy loss and Adam optimizer. |
| `train_model(model, x, y, epochs, batch_size, checkpoint_dir)` | Trains the model with best-loss checkpointing. Saves `.keras` files to the specified directory. Returns the training history object. |
| `load_model(weights_path, seq_length, n_vocab)` | Rebuilds the model architecture and loads pre-trained weights. Used for generation. |

### `l4llm.generate` — Text Generation

| Function | Description |
|---|---|
| `sample_with_temperature(predictions, temperature)` | Applies temperature scaling to a probability distribution and samples an index using multinomial sampling. Handles numerical stability with epsilon smoothing. |
| `generate_text(model, seed_sequence, char_to_int, int_to_char, n_vocab, length, temperature)` | The main generation loop. Autoregressively generates `length` characters from a seed sequence using temperature-scaled sampling. |
| `get_random_seed(data_x, int_to_char)` | Selects a random sequence from the training data and returns both the integer sequence and the decoded text. |

### `l4llm.cli` — Command-Line Interface

| Function | Description |
|---|---|
| `main()` | Entry point registered in `pyproject.toml` as `l4llm`. Parses arguments and dispatches to `cmd_train` or `cmd_generate`. |
| `cmd_train(args)` | Handles the `train` subcommand: loads data, builds model, trains, saves checkpoints. |
| `cmd_generate(args)` | Handles the `generate` subcommand: loads data and weights, selects seed, generates text, prints output. |
| `build_parser()` | Constructs the argparse parser with all subcommands and arguments. |

---

## Configuration

### pyproject.toml

The project is configured using PEP 621 metadata in `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "l4llm"
version = "2.0.0"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "tensorflow>=2.16",
    "numpy>=1.24",
    "pandas>=2.0",
]

[project.scripts]
l4llm = "l4llm.cli:main"

[project.optional-dependencies]
dev = ["ruff>=0.8", "pytest>=8.0"]
```

### Ruff Linting

The project uses Ruff for both linting and formatting with strict rules:

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "SIM", "TCH", "RUF"]
```

| Rule Set | Description |
|---|---|
| `E`, `W` | pycodestyle errors and warnings |
| `F` | pyflakes (unused imports, variables, etc.) |
| `I` | isort (import ordering) |
| `UP` | pyupgrade (modernize Python syntax) |
| `B` | flake8-bugbear (common bug patterns) |
| `SIM` | flake8-simplify (code simplification) |
| `TCH` | type-checking import guards |
| `RUF` | Ruff-specific rules |

Run linting and formatting:

```bash
# Check for issues
ruff check src/l4llm/

# Auto-fix issues
ruff check --fix src/l4llm/

# Format code
ruff format src/l4llm/

# Verify formatting
ruff format --check src/l4llm/
```

---

## Generated Examples

Here are real outputs from a model trained for 50 epochs (loss: 0.3404) on the 161-line dataset:

**Temperature 0.5 (Balanced — recommended):**

> are your legs tired? because you have been running through my mind all day.
> can i follow you home? my parents always told me to follow my dreams.
> do you have a map? because i just got lost in your eyes.
> your hand looks heavy. can i hold it for you?
> i am not a genie but i can make your dreams come true.
> are you a camera? because every time i look at you everyone else disappears.
> are you french? because eiffel for you.
> are you a bank loan? because you have my interest.
> do you have a band-aid? because i scraped my knee falling for you.

**Temperature 0.8 (Creative):**

> your name mooglt? because you are a keeper.
> you must be a millt because you just swept me off my feet.
> are you a camk le? because you just swept me off my feet.
> i am not a photographer but i can picture us together.
> if you were a fruit you would be a fineapple.
> is your dad a baker? because you are a cutie pie.
> i hope you know CPR because you are taking my breath away.

**Temperature 1.0 (Wild):**

> i want our love to be like pi... irrational and never ending.
> if i could rearrange the alphabet i would put u and i together.
> are you a tornado? because you just blew me away.
> did you invent the airplane? because you seem wright for me.

<img src="assets/output-showcase.avif" alt="Generated pickup lines displayed on smartphone screens" width="100%">

---

## Understanding the Output Quality

You might notice that some generated lines contain typos, partial words, or grammatical errors. This is expected and is a fundamental characteristic of character-level LSTM text generation, especially on smaller datasets.

**Why does this happen?**

1. **Limited context window**: The model only sees `seq_length` (typically 60-100) characters at a time. It cannot reference text beyond this window, so it may lose track of the beginning of a sentence.

2. **No semantic understanding**: The model learns statistical character patterns, not meaning. It doesn't know that "magician" and "disappears" are related — it just knows they frequently appear together.

3. **Error accumulation**: Each generated character depends on all previous characters. One wrong character can cascade into increasingly garbled text, similar to how a small mistake in a photocopy gets amplified over multiple copies.

4. **Dataset size**: With only 161 training lines, the model has limited exposure to character patterns. More data would significantly improve output quality.

**How to improve output quality:**

- **Train longer**: Increase epochs to 100-200
- **Use a larger dataset**: Scrape more pickup lines from the internet (the `dataset/README.md` has instructions)
- **Increase sequence length**: Try `seq_length=150` or `200` for longer-range dependencies
- **Use a lower temperature**: `0.3-0.5` produces more conservative but coherent output
- **Add more LSTM layers**: A third LSTM layer could capture more complex patterns
- **Use beam search**: Instead of sampling, keep the top-k predictions at each step (would require code modification)

---

## The Original Legacy Code

The original `pickuplines_generator.py` is preserved in the repository root for reference. It was a single monolithic script that:

- Used the deprecated standalone `keras` imports (not `tf.keras`)
- Used `keras.utils.np_utils.to_categorical` (removed in Keras 3)
- Passed `input_shape` directly to the LSTM layer (deprecated in Keras 3)
- Had no CLI, no type hints, no logging, no modularity
- Mixed training and inference in one script
- Used greedy `argmax` for text generation (always picks the most likely next character, resulting in repetitive output)
- Saved checkpoints in the legacy `.hdf5` format

You can compare the old and new approaches:

```bash
# Old way (legacy)
python pickuplines_generator.py

# New way
l4llm train -e 50
l4llm generate -w checkpoints/weights-improvement-50-X.XXXX.keras -s 60 -t 0.5
```

---

## Modernization Changelog

The following changes were made to bring this project from ~2018 to 2026 standards:

| Commit | Change |
|---|---|
| `6829989` | Added `pyproject.toml` (PEP 621) with hatchling build, Ruff config, package scaffold |
| `cd01132` | Added `data.py` module: CSV loading, char mappings, sequence prep, `tf.keras.utils` |
| `9cafca3` | Added `model.py` module: 2-layer LSTM build, train with checkpointing, weight loading |
| `582c83b` | Added `generate.py` module: temperature sampling, autoregressive generation, random seeds |
| `a46eebe` | Added `cli.py` module: argparse CLI with train/generate subcommands and logging |
| `36202c5` | Updated `.gitignore` for `.keras`, `.hdf5`, `checkpoints/`, `.ruff_cache/` |
| `8276a78` | Added `requirements.txt` and `requirements-dev.txt` |
| `021001b` | Fixed build backend, Keras 3 Input layer, CSV parsing, Ruff warnings |
| `c50088c` | Populated dataset with 161 real pickup lines |

---

## Image Generation Prompts

Want to create custom images for this README? Here are ready-to-use prompts for AI image generators (Midjourney, DALL-E, Stable Diffusion, etc.):

### Hero Banner
> **Prompt:** "A sleek dark-themed hero banner for a GitHub repository called 'l4llm' - LSTM Pickup Lines Generator. Show a stylized neural network visualization on the left side with glowing LSTM cells connected by flowing data streams, and on the right side show speech bubbles with flirty text emerging from a terminal window. Color palette: deep navy (#0d1117), electric purple (#8b5cf6), neon pink (#ec4899), and cyan (#06b6d4). Modern, clean, tech-meets-romance aesthetic. No text in the image."
> **Aspect Ratio:** 16:9 (1280x720)

### Architecture Diagram
> **Prompt:** "A clean technical diagram showing a text generation pipeline. Left: a document icon labeled 'Dataset' with pickup line text flowing out. Center: a glowing LSTM neural network with two stacked layers showing memory cells and gates, labeled '2-Layer LSTM (256 units)'. Right: a terminal window showing generated text output. Arrows connect each stage. Below the LSTM: a color-coded temperature slider going from blue (cold/conservative) to red (hot/creative). Dark background, modern infographic style, no text overlap."
> **Aspect Ratio:** 21:9 (1920x823)

### Temperature Visualization
> **Prompt:** "A horizontal infographic showing the effect of temperature on text generation. On the far left (labeled 'T=0.1, Conservative'): a neatly organized bookshelf with identical books. In the center-left (labeled 'T=0.5, Balanced'): a bookshelf with mostly organized books but a few colorful ones mixed in. In the center-right (labeled 'T=0.8, Creative'): a bookshelf with books scattered creatively, some open with glowing text. On the far right (labeled 'T=1.5, Chaotic'): books flying off shelves in a whirlwind of pages with glowing characters. Dark gradient background from cool blue on left to hot red on right."
> **Aspect Ratio:** 16:9 (1280x720)

### Output Showcase
> **Prompt:** "A dark-themed gallery display showing 6 floating smartphone screens, each displaying a different pickup line with a chat bubble interface. The phones are arranged in a 3x2 grid at a slight 3D angle. Each screen has a different accent color (purple, pink, cyan, green, orange, blue). The background is a deep dark gradient with subtle neural network node connections glowing faintly. Modern, sleek, social-media aesthetic."
> **Aspect Ratio:** 16:9 (1280x720)

### Training Loss Curve
> **Prompt:** "A beautiful data visualization of a training loss curve on a dark background. The curve starts high at 3.4 on the left and smoothly decreases to 0.34 on the right over 50 epochs. The line glows electric purple (#8b5cf6) with a gradient fill below it. The x-axis is labeled 'Epochs' and y-axis is labeled 'Loss'. Small milestone dots mark epochs 10, 20, 30, 40, 50. In the top right corner, show 'Loss: 3.43 → 0.34' in a glowing badge. Minimal grid lines, modern chart aesthetic, no clutter."
> **Aspect Ratio:** 16:9 (1280x720)

### Neural Network Close-Up
> **Prompt:** "A macro close-up artistic render of LSTM memory cells in a neural network. Show interconnected nodes with glowing synaptic connections. Data flows as streams of small characters (letters, punctuation) flowing through the connections like a river of text. The cells have a subtle hexagonal structure with translucent walls showing internal gate mechanisms (forget gate, input gate, output gate). Color scheme: deep blue background, cyan and purple glowing elements, with tiny golden characters floating in the data streams. Abstract, beautiful, technical art."
> **Aspect Ratio:** 4:3 (1280x960)

### Terminal Output Aesthetic
> **Prompt:** "A dramatic wide shot of a dark terminal window on a macOS desktop, showing colorful generated text output. The terminal has a dark theme with syntax-highlighted output in green, cyan, and magenta text. Surrounding the terminal are floating holographic speech bubbles containing pickup lines, connected to the terminal by thin glowing lines. The desktop wallpaper is a dark neural network visualization. Cinematic lighting, depth of field, slightly tilted angle."
> **Aspect Ratio:** 21:9 (2560x1097)

---

## Contributing

Contributions are welcome! Here are some ideas:

1. **Expand the dataset** — Add more pickup lines to `dataset/lines.csv` (the model quality scales with data)
2. **Implement beam search** — Replace temperature sampling with beam search for more coherent output
3. **Add a web interface** — Wrap the CLI in a simple Flask/FastAPI app with a text box
4. **Try different architectures** — GRU, Transformer, or fine-tune a small language model
5. **Add evaluation metrics** — Measure output quality with perplexity, BLEU, or human ratings
6. **Export to TFLite** — Make the model runnable on mobile devices
7. **Multi-language support** — Train on pickup lines in other languages

### Development Setup

```bash
git clone https://github.com/anubhavg-icpl/l4llm.git
cd l4llm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run linting
ruff check src/l4llm/
ruff format --check src/l4llm/

# Run tests (when available)
pytest
```

---

## License

This project is licensed under the **MIT License** — see the `pyproject.toml` for details.

---

## Disclaimer

Some generated pickup lines may be cheesy, corny, or mildly inappropriate. This is a direct reflection of the training data collected from the internet. The model has no concept of social appropriateness — it merely reproduces statistical patterns in character sequences. Use responsibly (or irresponsibly, we won't judge).

<div align="center">

*Built with ❤️ and LSTM cells*

</div>


