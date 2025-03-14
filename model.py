import math

import torch
import torch.nn as nn

"""
Only for review
"""


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embeddings = nn.Embeddings(vocab_size, d_model)

    def forward(self, x):
        x = self.embeddings(x) * math.sqrt(self.d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position / div)
        pe[:, 1::2] = torch.cos(position / div)

        pe = pe.unsqeeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return x


class LayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zoros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (self.eps + std) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q, k, v, mask, dropout):
        d_k = q.shape[-1]
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_scores, dim=-1)

        if dropout is not None:  # tipically dropout is 0.2
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, v), attention_scores

    def forward(self, q, k, v, mask):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(q.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], -1, self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            q, k, v, mask, self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_fordward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_fordward_block = feed_fordward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        return self.residual_connections[1](x, self.feed_fordward_block)


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


"""
Only for review
"""


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, n_segment, max_len, embed_dim, dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.seg_embed = nn.Embedding(n_segment, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.pos_inp = torch.tensor(
            [i for i in range(max_len)],
        )

    def forward(self, seq, seg):
        embed_val = (
            self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(self.pos_inp)
        )
        return embed_val


class BERT(nn.Module):
    def __init__(
        self, vocab_size, n_segment, max_len, embed_dim, n_layers, attn_heads, dropout
    ):
        super().__init__()
        self.embedding = BERTEmbedding(
            vocab_size, n_segment, max_len, embed_dim, dropout
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, attn_heads, embed_dim * 4
        )
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = self.encoder_block(out)
        return out
