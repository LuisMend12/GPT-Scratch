# GPT-Scratch

A character-level GPT (decoder-only Transformer) built from scratch in PyTorch, following Andrej Karpathy's video [**"Let's build GPT: from scratch, in code, spelled out"**](https://www.youtube.com/watch?v=kCc8FmEb1nY).

The goal is learning, not production: implement every piece of a small GPT by hand — token/position embeddings, self-attention, multi-head attention, feed-forward layers, residual connections, and layer norm — and train it to generate Shakespeare-like text one character at a time.

## What this is

The model is trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset (~1MB of text). It learns to predict the next character given the previous ones, then generates new text by sampling repeatedly from its own predictions.

The project is built up in stages, mirroring the video:

1. **Bigram model** — a simple baseline that predicts the next character from only the current one (a lookup table).
2. **Self-attention** — letting each position "look back" at previous positions and aggregate information.
3. **Multi-head attention + feed-forward** — the core Transformer block.
4. **Scaling up** — stacking blocks, adding residual connections, layer norm, and dropout to form the final GPT.

## Project structure

```
GPT-Scratch/
├── input.txt        # Tiny Shakespeare dataset (download — see below)
├── bigram.py        # Stage 1: the simple bigram baseline
├── gpt.py           # The full GPT implementation
└── README.md
```

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/get-started/locally/)

```bash
pip install torch
```

A GPU (CUDA) is helpful but not required — the model is small enough to train on CPU, just slower.

## Getting started

1. **Get the dataset.** Download Tiny Shakespeare into the project root as `input.txt`:

   ```bash
   curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
   ```

2. **Train and generate.**

   ```bash
   python gpt.py
   ```

   The script trains the model and then prints generated Shakespeare-style text.

## Key concepts covered

- Character-level tokenization (encode text → integers, decode back)
- Batching with `block_size` (context length) and `batch_size`
- The self-attention mechanism ("the mathematical trick" — using a lower-triangular mask for causal attention)
- Scaled dot-product attention and why we divide by √(head size)
- Multi-head attention
- Position-wise feed-forward networks
- Residual ("skip") connections and `LayerNorm`
- Dropout for regularization
- The training loop, loss estimation, and autoregressive text generation

## Credits

- Video: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy
- Reference repo: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Dataset: [Tiny Shakespeare](https://github.com/karpathy/char-rnn)
