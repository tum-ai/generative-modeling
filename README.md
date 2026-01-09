# generative-modeling

## Environment

Install [pixi package manager](https://pixi.sh/latest/installation/).
Pixi will automatically take care of the environment (based on `pyproject.toml` and `pixi.lock`) so no setup is required.

## Variational Inference

### GMM Training Comparison: Gradient Ascent vs EM

```bash
# generate dataset
pixi run python src/scripts/variational/generate_data.py

# train GMM with gradient ascent
pixi run python src/scripts/variational/train_gmm_gradient.py

# train GMM with EM algorithm
pixi run python src/scripts/variational/train_gmm_em.py
```

## Sequence

### Transformer
```bash
# train mnist tokenizer
pixi run python src/scripts/train_mnist_bpe.py

# train transformer sequence model
pixi run python src/scripts/train_mnist_transformer.py

# sample some new mnist images
pixi run python src/scripts/sample_mnist_transformer.py
```

### LSTM
```sh
# (place booksummaries.txt in ./data)

# preprocess booksummaries dataset
pixi run python src/scripts/preprocess_booksummaries.py

# train booksummaries tokenizer
pixi run python src/scripts/train_booksummaries_bpe.py

# train lstm sequence model
pixi run python src/scripts/train_booksummaries_lstm.py

# sample some new sequences
pixi run python src/scripts/sample_booksummaries_lstm.py 
```