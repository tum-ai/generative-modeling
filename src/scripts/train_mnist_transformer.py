import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from generative_modeling.sequence.utils import MNISTConverter
from generative_modeling.sequence.tokenizer import BPETokenizer
from generative_modeling.sequence.transformer import Transformer

# TODO hydra
# how many images to use for the training
n_images = 200
n_images_per_batch = 128
n_epochs = 100
lr = 3e-4
# they influence the transformer
vocab_size = 256
context_size = 64
# d_model = 4 * sqrt(vocab size) to 8 * sqrt(vocab size)
d_model = 64
# transformer blocks = log2(context_size)
# we take a little less
n_transformer_blocks = 4


def main():
    # Transformation: wandelt PIL-Images in Tensors um und normalisiert
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Download + laden der Trainings- und Testdaten
    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    converter = MNISTConverter()

    corpus = converter.convert_tensor_dataset(train_data, n_samples=100)

    bpe_tokenizer = BPETokenizer.load("out/mnist_bpe_tokenizer.pkl")

    ids = bpe_tokenizer.tokenize(corpus)

    sentence_lengths = (
        np.where(np.array(ids) == 0)[0][1:] - np.where(np.array(ids) == 0)[0][:-1]
    )

    print(f"average sentence length = {np.mean(sentence_lengths)}")
    print(f"max sentence length = {np.max(sentence_lengths)}")

    train_sentences = torch.tensor(ids, dtype=torch.long)

    # slices with length context_size
    # each slice is slided one further
    train_inputs = train_sentences.unfold(0, context_size, 1)[:-1]
    train_targets = train_sentences.unfold(0, context_size, 1)[1:]

    train_set = TensorDataset(train_inputs, train_targets)

    train_loader = DataLoader(
        dataset=train_set, shuffle=True, batch_size=n_images_per_batch
    )

    model = Transformer(
        n_tokens=vocab_size,
        seq_length=context_size,
        d_model=d_model,
        n_transformer_blocks=n_transformer_blocks,
    )
    model.train()

    model.print_parameters()

    loss_values = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    loss_func = nn.CrossEntropyLoss()

    for epoch_idx in range(n_epochs):
        pbar_in_epoch = tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}/{n_epochs}")

        for input_tokens, next_tokens in pbar_in_epoch:
            predicted_next_tokens = model(input_tokens)

            B, S, V = predicted_next_tokens.shape

            predicted_next_tokens = predicted_next_tokens.reshape((B * S, V))
            next_tokens = next_tokens.flatten()

            loss_in_batch = loss_func(predicted_next_tokens, next_tokens)

            pbar_in_epoch.set_postfix({"loss": f"{loss_in_batch.item():.4f}"})
            loss_values.append(loss_in_batch.item())

            optimizer.zero_grad()

            loss_in_batch.backward()

            optimizer.step()

    plt.plot(np.arange(len(loss_values)), np.array(loss_values))
    plt.xlabel("#batch")
    plt.ylabel("cross entropy")
    plt.show()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "out/mnist_transformer_checkpoint.pth",
    )

    print("Model saved successfully!")


if __name__ == "__main__":
    main()
