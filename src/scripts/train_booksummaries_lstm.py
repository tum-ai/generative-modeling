import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from generative_modeling.sequence.tokenizer import BPETokenizer
from generative_modeling.sequence.lstm import LSTM


# TODO hydra
n_sentences = 100
vocab_size = 256
context_size = 64
batch_size = 64
n_epochs = 4
# lstm settings
h_dim = 128  # recommended by chatgpt
c_dim = 128  # both dimensions are the same usually
n_blocks = 2
lr = 1e-3
print_grad_norm = False
n_batches = 100
limit_by_n_batches = False


def main():
    sentences = []

    with open("data/booksummaries_preprocessed.txt", "r") as f:
        sentences = f.readlines()

    corpus = "".join(sentences[:n_sentences])

    tokenizer = BPETokenizer.load("out/booksummaries_bpe_tokenizer.pkl")

    ids = tokenizer.tokenize(text=corpus)

    train_sentences = torch.tensor(ids, dtype=torch.long)

    train_inputs = train_sentences.unfold(0, context_size, 1)[:-1]
    train_targets = train_sentences.unfold(0, context_size, 1)[1:]

    dataset = TensorDataset(train_inputs, train_targets)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    lstm = LSTM(
        vocab_size=tokenizer.get_vocab_size(),
        h_dim=h_dim,
        c_dim=c_dim,
        n_blocks=n_blocks,
    )
    lstm.train()

    lstm.print_parameters()

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=lstm.parameters(), lr=3e-4)

    batch_idx = 0

    loss_history = []

    for epoch_idx in range(n_epochs):
        pbar_epoch = tqdm(dataloader, desc=f"Epoch {epoch_idx + 1}/{n_epochs}")

        for xb, yb in pbar_epoch:

            if limit_by_n_batches and batch_idx > n_batches:
                break

            B, S = xb.shape

            # backpropagation through time
            hs = [torch.zeros((B, h_dim)) for _ in range(n_blocks)]
            cs = [torch.zeros((B, c_dim)) for _ in range(n_blocks)]

            loss = 0

            for t in range(S):
                out_logits, hs, cs = lstm(xb[:, t], hs, cs)

                targets = yb[:, t]

                loss_at_t = loss_func(out_logits, targets)

                loss += loss_at_t

            loss /= S

            loss_history.append(loss.item())

            optimizer.zero_grad()

            loss.backward()

            # debugging (do we suffer from exploding / vanishing gradients)
            if print_grad_norm:
                pbar_epoch.set_postfix(
                    {
                        "grad norm": f"{lstm.get_grad_norm():.4f}",
                        "loss": f"{loss.item():.4f}",
                    }
                )
            else:
                pbar_epoch.set_postfix({"loss": f"{loss.item():.4f}"})

            optimizer.step()

            batch_idx += 1

        print("checkpoint saved successfully!")

    plt.plot(np.arange(len(loss_history)), np.array(loss_history))
    plt.xlabel("#batch")
    plt.ylabel("cross entropy")
    plt.show()

    torch.save({"model_state_dict": lstm.state_dict()}, "out/booksummaries_lstm_checkpoint.pth")

    print("Model saved successfully!")


if __name__ == "__main__":
    main()
