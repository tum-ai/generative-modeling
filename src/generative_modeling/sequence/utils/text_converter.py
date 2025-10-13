from abc import ABC, abstractmethod
import torch
import numpy as np


class TextConverter(ABC):
    def __init__(self, special_tokens, basic_tokens):
        self.special_tokens = special_tokens
        self.basic_tokens = basic_tokens

        self.corpus = ""
        self.sentences = []

    @abstractmethod
    def to_text(self, tensor, label=None):
        pass

    @abstractmethod
    def from_text(self, token_arr):
        pass

    def convert_tensor_dataset(self, data, n_samples):
        self.sentences = []

        for sample_idx, (sample, label) in enumerate(data):
            if sample_idx > n_samples:
                break

            self.sentences.append(self.to_text(sample, label))

        self.corpus = "".join(self.sentences)

        print("Corpus and sentences set!")

        return self.corpus


class MNISTConverter(TextConverter):
    def __init__(self):
        super().__init__(
            special_tokens=["<BOS>", "<EOS>"] + [f"<{i}>" for i in range(10)],
            basic_tokens=["0", "1"],
        )

    def to_text(self, tensor, label=None):
        img = torch.round(tensor).flatten().int().tolist()
        return f"<BOS><{label}>" + "".join([f"{bit}" for bit in img]) + "<EOS>"

    def from_text(self, token_arr):
        raw_data = []

        start_seen = False

        for token in token_arr:
            if token == "<EOS>":
                break

            if start_seen and token not in self.special_tokens:
                try:
                    raw_data.append(int(token))
                except:
                    print(f"Decoding error for token {token}")

            if token == "<BOS>":
                start_seen = True

        target_len = 28 * 28

        # cut at target_len (shouldn't be bigger)
        img = np.array(raw_data)[:target_len]

        img = np.pad(
            img, (0, max(target_len - len(img), 0)), mode="constant", constant_values=0
        )

        return img.reshape(28, 28)
