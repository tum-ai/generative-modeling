import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBlock(nn.Module):
    def __init__(self, h_dim, c_dim):
        super().__init__()

        self.h_dim = h_dim
        self.c_dim = c_dim

        # don't store the cell state and hidden state within the LSTM
        # it should be stateless to make batch training and inference easier

        # self.h = torch.zeros((self.h_dim))
        # self.c = torch.zeros((self.c_dim, ))

        self.linear_to_f = nn.Linear(2 * self.h_dim, self.h_dim)
        self.linear_to_i = nn.Linear(2 * self.h_dim, self.h_dim)

        self.linear_to_c_tilde = nn.Linear(2 * self.h_dim, self.c_dim)

        self.linear_to_o = nn.Linear(2 * self.h_dim, self.h_dim)

        self.linear_cell_to_hidden = nn.Linear(self.c_dim, self.h_dim)

    # a batch full of next tokens
    # and hidden and cell states (state is out side of the lstm block)
    def forward(self, x, h, c):
        # xvecs = (B, hidden_dim)
        # we get (B, 2 * hidden_dim)
        concat = torch.cat([x, h], dim=-1)

        f_preactivated = self.linear_to_f(concat)
        i_preactivated = self.linear_to_i(concat)

        # how much of the previous cell state should be kept
        f = F.sigmoid(f_preactivated)

        # how much of the candidate cell state should be written to the cell state (long term memory)
        i = F.sigmoid(i_preactivated)

        c_tilde_preactivated = self.linear_to_c_tilde(concat)

        # we use tanh because this is not a percentage (how much we use something)
        # but we want to modify something (negative values for reducing the cell state are also necessary)
        c_tilde = F.tanh(c_tilde_preactivated)

        # construct new cell state
        c_new = (f * c) + (i * c_tilde)

        # how much of the updated cell state (long term) should influence the new hidden state (short term)
        o_preactivated = self.linear_to_o(concat)
        o = F.sigmoid(o_preactivated)

        h_new = self.linear_cell_to_hidden(c_new)
        h_new = F.tanh(h_new)

        h_new = o * h_new

        return h_new, c_new


class LSTM(nn.Module):
    def __init__(self, vocab_size, h_dim, c_dim, n_blocks):
        super().__init__()

        self.vocab_size = vocab_size
        self.h_dim = h_dim
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=h_dim)

        self.blocks = nn.ModuleList(
            [LSTMBlock(h_dim, c_dim) for _ in range(self.n_blocks)]
        )

        self.output_head = nn.Linear(self.h_dim, vocab_size)

    def print_parameters(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.requires_grad}")

    def get_grad_norm(self):
        total_norm = 0.0

        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  # L2-Norm
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        return total_norm

    def forward(self, x, hs, cs):
        # we need the short and long term memory for every block
        assert len(hs) == self.n_blocks
        assert len(cs) == self.n_blocks

        x_vecs = self.token_emb(x)

        new_hs = []
        new_cs = []

        for i, block in enumerate(self.blocks):
            x_input = x_vecs if i == 0 else new_hs[i - 1]

            new_h, new_c = block(x_input, hs[i], cs[i])

            if i < self.n_blocks - 1:
                new_h = F.layer_norm(new_h, normalized_shape=(self.h_dim,))

            new_hs.append(new_h)
            new_cs.append(new_c)

        out_logits = self.output_head(new_hs[-1])

        return out_logits, new_hs, new_cs
