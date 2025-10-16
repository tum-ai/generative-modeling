from generative_modeling.sequence.tokenizer import (
    BPETokenizer,
    extract_unique_chars_from_corpus,
)

# TODO hydra
n_sentences = 100
vocab_size = 256


def main():
    sentences = []

    with open("data/booksummaries_preprocessed.txt", "r") as f:
        sentences = f.readlines()

    corpus = "".join(sentences[:n_sentences])

    tokenizer = BPETokenizer(
        special_tokens=["<BOS>", "<SUM>", "<EOS>"],
        basic_tokens=extract_unique_chars_from_corpus(corpus),
    )

    tokenizer.train(corpus=corpus, vocab_size=vocab_size)

    ids = tokenizer.tokenize(text=sentences[0])

    print("".join(tokenizer.ids_to_text(ids)))

    tokenizer.save("out/booksummaries_bpe_tokenizer.pkl")


if __name__ == "__main__":
    main()
