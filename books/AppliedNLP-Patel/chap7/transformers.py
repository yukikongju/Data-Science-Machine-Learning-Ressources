import torch

from torch import nn


def generate_masking():
    pass
    

def test_multihead_embedding():
    batch_size, vocab_size, nheads = 5, 100, 2
    N, E, seq_length, target_length = 5, 16, 10, 20

    attention = nn.MultiheadAttention(embed_dim=E, num_heads=nheads)
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=E, 
            padding_idx=0)
    
