from torch import nn as nn
from components.data.common import PAD_ID


def get_embed_matrix(vocab_size, embedding_dim):
    return nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
