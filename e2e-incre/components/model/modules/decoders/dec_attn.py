import torch
import torch.nn as nn

from components.model.modules.decoders import DecoderRNNAttnBase
from components.model.modules.attention.attn_bahd import AttnBahd
from components.model import get_GRU_unit


class DecoderRNNAttnBahd(DecoderRNNAttnBase):
    def __init__(self, rnn_config, output_size, prev_y_dim, enc_dim, enc_num_directions):
        super(DecoderRNNAttnBahd, self).__init__()

        self.rnn = get_GRU_unit(rnn_config)

        # Setting attention
        dec_dim = rnn_config["hidden_size"]
        self.attn_module = AttnBahd(enc_dim, dec_dim, enc_num_directions)
        self.W_combine = nn.Linear(prev_y_dim + enc_dim * enc_num_directions, dec_dim)
        self.W_out = nn.Linear(dec_dim, output_size)
        self.log_softmax = nn.LogSoftmax()  # works with NLL loss

    def combine_context_run_rnn_step(self, prev_y_batch, prev_h_batch, context):
        """

        :param prev_y_batch: B x prev_y_dim
        :param prev_h_batch: 1 x B x dec_dim
        :param context: # B x 1 x MLP_H
        :return:
        """
        y_ctx = torch.cat((prev_y_batch, context.squeeze(1)), 1)  # B x (prev_y_dim+(enc_dim * num_enc_directions))
        rnn_input = self.W_combine(y_ctx)  # B x H
        output, decoder_hidden = self.rnn(rnn_input.unsqueeze(0), prev_h_batch)  # 1 x B x H, 1 x B x H

        return output, decoder_hidden

    def compute_output(self, rnn_output):
        """

        :param rnn_output: 1 x B x H
        :return:
        """
        unnormalized_logits = self.W_out(rnn_output[0])  # B x TV
        logits = self.log_softmax(unnormalized_logits)  # B x TV

        return logits
