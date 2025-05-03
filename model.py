import math
import torch
import torch.nn as nn


"""
The first think to do is the coding the input
embeddings, basically we have in input a sequence
of tokens and we produce a sequence of embeddings
with nn.Embeddings(num_embeddings, embedding_dim)
"""
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        """
        Input:
            x: a sequence of tokens
        Output:
            x: a sequence of embeddings
        Parameters:
            vocab_size: the size of the vocabulary
            d_model: the size of the embeddings
        """

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.embeddings(x) * math.sqrt(self.d_model)
        return x


"""
Now we need to add the positional encodings
The positional encodings are added to the input embeddings
to give to the model (transformer) the information
about the position of the words (tokens) inside the sequence
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        
        """
        Input:
            x: a sequence of embeddings
        Output:
            x: a sequence of embeddings + positional encodings
        Parameters:
            d_model: the size of the embeddings
            seq_len: the length of the sequence
        """

        self.d_model = d_model
        self.seq_len = seq_len
        
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)

        pe = torch.zeros(seq_len, d_model)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

"""
We can now add all the transformer parts
"""
class LayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.eps) * self.alpha + self.beta
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q, k, v, mask, dropout):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1])

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2) 
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_scores = torch.softmax(attn_scores, dim=-1)

        if dropout is not None:
            attn_scores = dropout(attn_scores)

        return torch.matmul(attn_scores, v), attn_scores

    def forward(self, x, mask):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(q.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], -1, self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(q, k, v, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(self, attention_block, feed_forward_block, dropout):
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([SkipConnection(dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


"""
Finally we can code the BERT model
"""
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, h, d_ff, seq_len, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.h = h
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        self.input_embeddings = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_len)

        self.encoder = Encoder([EncoderBlock(MultiHeadAttention(d_model, h, dropout), FeedForward(d_model, d_ff, dropout), dropout) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.input_embeddings(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask)
        return x


