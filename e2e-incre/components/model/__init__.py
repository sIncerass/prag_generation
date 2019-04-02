import torch
from torch import nn as nn
from components.model.modules import get_embed_matrix


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.use_cuda = torch.cuda.is_available()


class Seq2SeqModel(BaseModel):
    def set_src_vocab_size(self, vocab_size):
        self._src_vocab_size = vocab_size

    def set_tgt_vocab_size(self, vocab_size):
        self._tgt_vocab_size = vocab_size

    def set_max_src_len(self, l):
        self._max_src_len = l

    def set_max_tgt_len(self, l):
        self._max_tgt_len = l

    @property
    def src_vocab_size(self):
        return self._src_vocab_size

    @property
    def tgt_vocab_size(self):
        return self._tgt_vocab_size

    @property
    def max_src_len(self):
        return self._max_src_len

    @property
    def max_tgt_len(self):
        return self._max_tgt_len


class E2ESeq2SeqModel(Seq2SeqModel):
    def setup(self, data):
        self.set_flags()
        self.set_data_dependent_params(data)
        self.set_embeddings()
        self.set_encoder()
        self.set_decoder()

    def set_data_dependent_params(self, data):
        vocabsize = len(data.vocab)
        self.set_src_vocab_size(vocabsize)
        self.set_tgt_vocab_size(vocabsize)
        self.set_max_src_len(data.max_src_len)
        self.set_max_tgt_len(data.max_tgt_len)

    def set_embeddings(self):
        self.embedding_dim = self.config["embedding_dim"]
        self.embedding_mat = get_embed_matrix(self.src_vocab_size, self.embedding_dim)

        embedding_drop_prob = self.config.get('embedding_dropout', 0.0)
        self.embedding_dropout_layer = nn.Dropout(embedding_drop_prob)

    def embedding_lookup(self, ids, *args, **kwargs):
        return self.embedding_mat(ids)

    def set_flags(self):
        self.teacher_forcing_ratio = self.config.get("teacher_forcing_ratio", 1.0)

    def set_encoder(self):
        raise NotImplementedError()

    def set_decoder(self):
        raise NotImplementedError()


def get_GRU_unit(gru_config):
    return nn.GRU(input_size=gru_config["input_size"],
                  hidden_size=gru_config["hidden_size"],
                  dropout=gru_config["dropout"],
                  bidirectional=gru_config.get("bidirectional", False))
