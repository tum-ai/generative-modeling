from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from generative_modeling.sequence.utils import MNISTConverter
from generative_modeling.sequence.tokenizer import BPETokenizer

# TODO hydra
vocab_size = 256


def main():
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

    bpe_tokenizer = BPETokenizer(
        special_tokens=converter.special_tokens, basic_tokens=converter.basic_tokens
    )

    bpe_tokenizer.train(corpus=corpus, vocab_size=vocab_size)
    bpe_tokenizer.save("out/mnist_bpe_tokenizer.pkl")

    ids = bpe_tokenizer.tokenize(converter.sentences[1])

    print(ids)

    token_arr = bpe_tokenizer.ids_to_text(ids)

    img = converter.from_text(token_arr)

    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
