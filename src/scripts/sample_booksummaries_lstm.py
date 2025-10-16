import torch
from tqdm import tqdm

from generative_modeling.sequence.lstm import LSTM
from generative_modeling.sequence.tokenizer import BPETokenizer

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
    checkpoint = torch.load("out/booksummaries_lstm_checkpoint.pth", map_location="cpu")

    model = LSTM(vocab_size=vocab_size, h_dim=h_dim, c_dim=c_dim, n_blocks=n_blocks)

    # load the weights of the trained model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_samples = 10
    n_to_sample = context_size

    hs = [torch.zeros((n_samples, h_dim)) for _ in range(n_blocks)]
    cs = [torch.zeros((n_samples, c_dim)) for _ in range(n_blocks)]

    sentences = []

    with open("data/booksummaries_preprocessed.txt", "r") as f:
        sentences = f.readlines()

    corpus = "".join(sentences[:n_sentences])

    tokenizer: BPETokenizer = BPETokenizer.load("out/booksummaries_bpe_tokenizer.pkl")

    prompt = "<BOS>Summer Breeze<SUM>"

    ids = tokenizer.tokenize(prompt)

    print(ids)

    x = torch.tensor(ids, dtype=torch.long).repeat((n_samples, 1))

    T = 0.5

    # load the prompt into the lstm (to update short and longterm memory)
    for t in range(len(ids)):
        _, hs, cs = model.forward(x[:, t], hs, cs)

    for _ in tqdm(range(n_to_sample)):
        out_logits, hs, cs = model.forward(x[:, -1], hs, cs)

        out_tokens = (
            torch.distributions.Categorical(logits=out_logits / T).sample().unsqueeze(1)
        )

        x = torch.cat([x, out_tokens], dim=-1)

    for sample in x:
        generated_ids = sample.tolist()

        token_arr = tokenizer.ids_to_text(generated_ids)

        print("".join(token_arr))
        print("---")


if __name__ == "__main__":
    main()
