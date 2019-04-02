# Based on: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from components.data.common import cuda_if_gpu


class EncoderRNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EncoderRNN, self).__init__()

    def forward(self, input_seqs_embedded, hidden, *args, **kwargs):
        outputs, hidden = self.rnn(input_seqs_embedded, hidden)

        # outputs:  # SL x B x (enc_dim * num_directions)
        # hidden: # (num_layers * num_directions) x B x enc_dim

        return outputs, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.rnn.num_layers * self.num_directions,
                                      batch_size,
                                      self.rnn.hidden_size))

        return cuda_if_gpu(hidden)

    def squeeze_output(self, encoder_outputs, encoder_hidden):
        """

        :param encoder_outputs: SL x B x (enc_dim * num_directions)
        :param hidden: (num_layers * num_directions) x B x enc_dim
        :return:
        """

        # Current implementation is summing the states
        encoder_outputs_sum = encoder_outputs[:, :, :self.rnn.hidden_size] + \
                              encoder_outputs[:, :, self.rnn.hidden_size:]

        encoder_hidden_sum = torch.sum(encoder_hidden, 0).unsqueeze(0)

        return (encoder_outputs_sum, encoder_hidden_sum)

    @property
    def num_directions(self):
        nd = 2 if self.rnn.bidirectional else 1
        return nd
