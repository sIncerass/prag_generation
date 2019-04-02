import torch
from torch import nn as nn
import torch.nn.functional as F


class DecoderRNNAttnBase(nn.Module):
    """
    To be implemented for each Decoder:
    self.rnn
    self.attn_module
    self.combine_context_run_rnn
    self.compute_output
    """

    def forward(self, prev_y_batch, prev_h_batch, encoder_outputs_batch):
        """
        A forward step of the Decoder.
        The step operates on one step-slice of the target sequence.

        :param prev_y_batch: embedded previous predictions: B x E
        :param prev_h_batch: current decoder state: 1 x B x dec_dim
        :param encoder_outputs_batch: SL x B x MLP_H
        :return:
        """
        # Calculate attention from current RNN state and all encoder outputs;
        attn_weights = self.attn_module(prev_h_batch, encoder_outputs_batch)  # B x SL

        # Apply attention weights to encoder outputs to get weighted average
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs_batch.transpose(0, 1))  # B x 1 x MLP_H

        # Combine embedded input word and attended context, run through RNN
        dec_rnn_output, dec_hidden = self.combine_context_run_rnn_step(prev_y_batch, prev_h_batch,
                                                                       context)  # 1xBxH, 1xBxH
        dec_output = self.compute_output(dec_rnn_output)  # logits (Softmax(), not LogSoftmax())

        # Return final output, hidden state, and attention weights (for visualization)
        return dec_output, dec_hidden, attn_weights

    @property
    def num_directions(self):
        return 2 if self.rnn.bidirectional else 1

    def combine_context_run_rnn_step(self, *args, **kwargs):
        raise NotImplementedError()

    def compute_output(self, *args, **kwargs):
        raise NotImplementedError()
