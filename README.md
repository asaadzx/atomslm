# AtomSLM

A from-scratch mini transformer language model (~35M parameters) with a Gemma-like architecture (SwiGLU FFN, RMSNorm, pre-norm residual blocks) trained on FineWeb-Edu. Includes conversion pipelines to HuggingFace safetensors and GGUF format for inference via llama.cpp.

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 4 |
| Hidden dim | 256 |
| Attention heads | 8 |
| Head dim | 32 |
| Intermediate (SwiGLU) | 682 |
| Vocab size | 256,000 |
| Max sequence length | 512 |
| Total parameters | ~34.8M |

## Quick Start

### Install

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Prepare dataset

Streams FineWeb-Edu (sample-10BT), tokenizes with Gemma-2B tokenizer, and saves 512-token blocks:

```bash
python main.py
```

### Train

```bash
python train.py
```

Trains for 2000 steps (batch size 4, learning rate 5e-4). The checkpoint is saved to `model/atom_slm.pt`.

### Test

```bash
python test_model.py
```

Runs a forward pass with random inputs and attempts greedy text generation.

## Project Structure

```
AtomSLM/
├── model.py              # Transformer model definition
├── train.py              # Training loop
├── main.py               # Dataset preparation (FineWeb-Edu tokenization)
├── dataset.py            # Dataset inspection utility
├── test_model.py         # Inference & forward-pass test
├── prepare_hf.py         # Tokenizer + config setup for model dir
├── convert_to_gguf.py    # .pt -> safetensors -> GGUF pipeline
├── pyproject.toml        # Project config & dependencies
├── model/
│   ├── config.json       # Model configuration
│   ├── tokenizer.json    # Gemma tokenizer
│   ├── tokenizer.model   # SentencePiece model
│   └── tokenizer_config.json
└── dataset/              # Tokenized training blocks (.npy)
```

## License

MIT
