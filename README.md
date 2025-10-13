# generative-modeling

## Instructions

### Environment

Install [pixi package manager](https://pixi.sh/latest/installation/).
Pixi will automatically take care of the environment (based on `pyproject.toml` and `pixi.lock`) so no setup is required.

### Transformer
```bash
# train bpe tokenizer
pixi run python src/scripts/train_mnist_bpe.py

# train transformer sequence model
pixi run python src/scripts/train_mnist_transformer.py

# sample some new mnist images
pixi run python src/scripts/sample_mnist_transformer.py
```
