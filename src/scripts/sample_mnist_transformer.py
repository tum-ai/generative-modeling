import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from generative_modeling.sequence.transformer import Transformer
from generative_modeling.sequence.utils import MNISTConverter
from generative_modeling.sequence.tokenizer import BPETokenizer

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
    checkpoint = torch.load("out/mnist_transformer_checkpoint.pth", map_location="cpu")

    model = Transformer(
        n_tokens=vocab_size,
        seq_length=context_size,
        d_model=d_model,
        n_transformer_blocks=n_transformer_blocks,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # image + the EOS token
    n_tokens_to_sample = 28 * 28 + 1

    n_samples = 10

    # temperature for sampling
    T = 0.5

    # convert to images
    converter = MNISTConverter()

    bpe_tokenizer = BPETokenizer.load("out/mnist_bpe_tokenizer.pkl")

    prompt = "<BOS>"

    x = torch.tensor(bpe_tokenizer.tokenize(prompt)).repeat((n_samples, 1))

    for _ in tqdm(range(n_tokens_to_sample)):
        # model can generate more than context size but not based on longer context (positional embedding table out of bounds)
        last_context = x[:, -context_size:]

        # should be (n_samples, context_length, vocab_size)
        logits_next_tokens = model(last_context)[:, -1, :]

        sampled_next_tokens = (
            torch.distributions.Categorical(logits=logits_next_tokens / T)
            .sample()
            .unsqueeze(1)
        )

        x = torch.cat([x, sampled_next_tokens], dim=1)

    for sample in x:
        token_arr = bpe_tokenizer.ids_to_text(sample.tolist())

        img = converter.from_text(token_arr)

        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    main()
