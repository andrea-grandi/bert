import torch

from model import BERT, BERTEmbedding

if __name__ == "__main__":
    VOCAB_SIZE = 30000
    N_SEGMENT = 3
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYERS = 12
    ATTN_HEADS = 12
    DROPOUT = 0.2

    sample_seq = torch.randint(
        high=VOCAB_SIZE,
        size=[
            MAX_LEN,
        ],
    )
    sample_seg = torch.randint(
        high=N_SEGMENT,
        size=[
            MAX_LEN,
        ],
    )

    embedding = BERTEmbedding(VOCAB_SIZE, N_SEGMENT, MAX_LEN, EMBED_DIM, DROPOUT)
    embedding_tensor = embedding(sample_seq, sample_seg)
    print(embedding_tensor.shape)

    bert = BERT(
        VOCAB_SIZE, N_SEGMENT, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT
    )
    out = bert(sample_seq, sample_seg)
    print(out.shape)
