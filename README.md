# generative-modeling

## Environment

Install [pixi package manager](https://pixi.sh/latest/installation/).
Pixi will automatically take care of the environment (based on `pyproject.toml` and `pixi.lock`) so no setup is required.

## Variational Inference

### CelebA Beta-VAE

To train the CelebA Beta-VAE, you need to manually set up the dataset due to Google Drive download limits.

#### 1. Manual Dataset Setup
1. Create the data directory: `mkdir -p data/celeba`
2. Download the following files from the CelebA Google Drive (or any mirror) to your local machine:
   - `img_align_celeba.zip`
   - `list_attr_celeba.txt`
   - `list_bbox_celeba.txt`
   - `list_eval_partition.txt`
   - `list_landmarks_align_celeba.txt`
3. Upload them to the server:
   ```bash
   # From your local machine
   scp img_align_celeba.zip list_attr_celeba.txt list_bbox_celeba.txt list_eval_partition.txt list_landmarks_align_celeba.txt USER@HOST:/workspace/generative-modeling/data/celeba/
   ```
4. Unzip the images on the server:
   ```bash
   cd /workspace/generative-modeling/data/celeba
   unzip img_align_celeba.zip
   ```

#### 2. Training
```bash
pixi run python src/scripts/train_celeba_beta_vae.py
```

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