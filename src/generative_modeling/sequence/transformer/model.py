import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# no multi head attention at first
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_qk, mask):
        super().__init__()

        self.d_model = d_model
        self.d_qk = d_qk
        self.mask = mask

        self.Q = nn.Linear(d_model, d_qk)
        self.K = nn.Linear(d_model, d_qk)
        self.V = nn.Linear(d_model, d_model)

    def forward(self, Xb):
        queries = self.Q(Xb)
        keys = self.K(Xb)
        values = self.V(Xb)

        # if one transposes the keys with key.T (for QK.T)
        # the shape (B, S, d_model) becomes (d_model, S, B)
        # but we want (B, d_model, S)
        B, S, d_model = Xb.shape

        keys_transposed = torch.transpose(keys, 1, 2)

        # scale factor uses the dimensionality of queries and keys because we take the dot product
        attention_matrix = (queries @ keys_transposed) / np.sqrt(self.d_qk)

        if self.mask:
            # exclude the main diagonal
            upper_triag = torch.triu(torch.ones(1, S, S), diagonal=1).bool()

            attention_matrix = attention_matrix.masked_fill(
                mask=upper_triag, value=float("-inf")
            )

        # last dimension for row-wise normalization
        attention_matrix = F.softmax(attention_matrix, dim=-1)

        out = attention_matrix @ values

        # residual connection and layer norm
        return F.layer_norm(out + Xb, normalized_shape=(self.d_model,))


class FFNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, Xb):
        return F.layer_norm(self.net(Xb) + Xb, normalized_shape=(self.d_model,))


class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attn = SelfAttentionBlock(d_model, d_qk=d_model, mask=True)
        self.ffn = FFNet(d_model=d_model)

    def forward(self, Xb):
        Xb = self.attn(Xb)
        return self.ffn(Xb)


class Transformer(nn.Module):
    def __init__(self, n_tokens, seq_length, d_model, n_transformer_blocks):
        super().__init__()

        self.token_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
        self.pos_emb = nn.Embedding(num_embeddings=seq_length, embedding_dim=d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(n_transformer_blocks)]
        )

        # output projection for next token prediction
        self.output_head = nn.Linear(d_model, n_tokens)

    def print_parameters(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.requires_grad}")

    def forward(self, Xb):
        B, S = Xb.shape

        Xb = self.token_emb(Xb)

        pos = self.pos_emb(torch.arange(0, S).unsqueeze(0).repeat((B, 1)))

        Xb = Xb + pos

        for block in self.blocks:
            Xb = block(Xb)

        return self.output_head(Xb)
