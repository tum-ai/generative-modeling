import re
from tqdm import tqdm
import pickle


class BPETokenizer:
    def __init__(self, special_tokens, basic_tokens):
        self.special_tokens = special_tokens
        self.basic_tokens = basic_tokens

        self.vocab = self.special_tokens + self.basic_tokens

        # growing over time
        self.pair_to_id = {}
        self.id_to_pair = {}

        # the process of tokenization: tokens (character groups) to ids (numbers)
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}

        self.special_token_ids = {self.token_to_id[tok] for tok in self.special_tokens}

    def get_vocab_size(self):
        return len(self.vocab) + len(self.pair_to_id)

    def naive_tokenization(self, corpus):
        # either a pattern that starts with <, then a positive number of characters that are not > (so we don't allow <>) and then >
        # or we just tokenize every single character
        regex_pattern = re.compile(r"<[^>]+>|.")

        tokens = regex_pattern.findall(corpus)

        return [self.token_to_id[tok] for tok in tokens]

    def get_next_most_common_byte_pair(self, ids):
        unique_id_pairs = set(zip(ids[:-1], ids[1:]))

        id_pair_counts = {pair: 0 for pair in unique_id_pairs}

        for pair in zip(ids[:-1], ids[1:]):
            id_pair_counts[pair] += 1

        # exclude pairs where special characters would be fused
        for first, second in id_pair_counts.keys():
            if first in self.special_token_ids or second in self.special_token_ids:
                id_pair_counts[(first, second)] = 0

        # sort with respect to frequency descending and pick most common
        id_pair_counts = [(pair, count) for pair, count in id_pair_counts.items()]

        # sort in descending order
        id_pair_counts = sorted(id_pair_counts, key=lambda t: -t[1])

        # get most common pair and only take pair, not count (second part of the tuple)
        return id_pair_counts[0][0]

    def replace_pair(self, ids, pair):
        new_ids = []

        i = 0

        while i < len(ids) - 1:
            first = ids[i]
            second = ids[i + 1]

            if (first, second) == pair:
                new_ids.append(self.pair_to_id[pair])
                i += 2
            else:
                new_ids.append(first)
                i += 1

        if i < len(ids):
            new_ids.append(ids[i])

        return new_ids

    def replace_most_common_pair(self, ids, most_common_pair):
        new_id = self.get_vocab_size()

        # added to vocabulary
        self.pair_to_id[most_common_pair] = new_id
        self.id_to_pair[new_id] = most_common_pair

        return self.replace_pair(ids, pair=most_common_pair)

    def tokenize(self, text):
        assert len(self.pair_to_id) > 0, "tokenizer has not been trained yet!"

        ids = self.naive_tokenization(text)

        # sort pairs by id
        sorted_pairs = map(
            lambda t: t[0], sorted(self.pair_to_id.items(), key=lambda t: t[1])
        )

        for pair in sorted_pairs:
            ids = self.replace_pair(ids, pair)

        return ids

    def pair_id_to_flat_id(self, pair_id):
        # not at the end
        if pair_id in self.id_to_pair:
            left_pair_id, right_pair_id = self.id_to_pair[pair_id]

            return self.pair_id_to_flat_id(left_pair_id) + self.pair_id_to_flat_id(
                right_pair_id
            )

        else:
            return [pair_id]

    def ids_to_text(self, ids):
        text = []

        pair_ids_to_raw_ids = {}

        for id in self.id_to_pair.keys():
            pair_ids_to_raw_ids[id] = self.pair_id_to_flat_id(id)

        flat_ids = []

        for id in ids:
            if id in pair_ids_to_raw_ids:
                flat_ids += pair_ids_to_raw_ids[id]

            else:
                # already a flat id
                flat_ids.append(id)

        for id in flat_ids:
            text.append(self.id_to_token[id])

        return text

    def train(self, corpus, vocab_size):
        ids = self.naive_tokenization(corpus)

        pbar_merges = tqdm(range(self.get_vocab_size(), vocab_size))

        for _ in pbar_merges:
            pair = self.get_next_most_common_byte_pair(ids)

            ids = self.replace_most_common_pair(ids, most_common_pair=pair)

            pbar_merges.set_postfix({"Corpus length": f"{len(ids)}"})

        print(f"Tokenizer training finished!")

    def save(self, name):
        with open(name, "wb") as f:
            pickle.dump(self, f)

        print(f"BPE Tokenizer successfully saved under {name}")

    @staticmethod
    def load(name):
        with open(name, "rb") as f:
            print(f"BPE Tokenizer successfully loaded!")
            return pickle.load(f)
