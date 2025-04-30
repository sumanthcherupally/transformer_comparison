# Transformer Comparison Project

This project implements and compares different transformer architectures and attention mechanisms for Language Model tasks. The implementation includes various attention mechanisms and provides tools for measuring their performance, memory usage, and computational efficiency.

## Project Structure

```
transformer_comparison/
├── src/
│   ├── model.py                  # Main transformer model implementation
│   ├── attention/
│   │   ├── dot_product.py        # Standard dot-product attention
│   │   ├── linear.py             # Linear attention implementation
│   │   ├── sparse.py             # Sparse attention implementation
│   │   ├── local.py              # Local attention implementation
│   │   ├── gqa.py                # Grouped Query Attention
│   │   ├── flash.py              # Flash Attention implementation
│   │   ├── rope.py               # Rotary Position Embedding
│   │   └── alibi.py              # Attention with Linear Biases
│   ├── utils/
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── data_loader.py        # Data loading utilities
│   └── train.py                  # Training script
├── configs/
│   └── config.yaml               # Model and training configuration
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Model Architecture

The transformer model implementation includes:

### Attention Mechanisms

1. **Dot Product Attention**: Standard scaled dot-product attention (O(n²) complexity)
2. **Linear Attention**: Linear complexity attention with ELU activation (O(n) complexity)
3. **Sparse Attention**: Top-k sparse attention pattern for reduced computation
4. **Local Attention**: Sliding window attention for capturing local context
5. **Grouped Query Attention**: Efficient attention with grouped queries
6. **Flash Attention**: Block-wise attention for improved memory efficiency

### Positional Encoding

1. **Sinusoidal**: Standard sinusoidal positional encoding
2. **RoPE**: Rotary Position Embedding for better extrapolation
3. **ALiBi**: Attention with Linear Biases for handling longer sequences

## Training

To train a model with specific attention and positional encoding:

```bash
python src/train.py --config configs/config.yaml --attention_type [dot_product|linear|sparse|local|gqa|flash] --position_embedding [sinusoidal|rope|alibi]
```

The training script will automatically log metrics to Weights & Biases for experiment tracking.

## Configuration

Model and training parameters can be modified in `configs/config.yaml`. Key parameters include:

- Model architecture (vocab_size, d_model, num_heads, num_layers)
- Training hyperparameters (batch_size, learning_rate, num_epochs)
- Attention mechanism specific parameters
- Positional encoding settings

## Metrics

The following metrics are tracked during training:
- Loss
- Perplexity
- Memory usage
- Training time
- Attention entropy

## License

MIT 