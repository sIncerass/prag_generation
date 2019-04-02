from torch import nn as nn


class AttnBahd(nn.Module):
    def __init__(self, enc_dim, dec_dim, num_directions, attn_dim=None):
        """
        Attention mechanism
        :param enc_dim: Dimension of hidden states of the encoder h_j
        :param dec_dim: Dimension of the hidden states of the decoder s_{i-1}
        :param attn_dim: Dimension of the internal dimension (default: same as decoder).
        """
        super(AttnBahd, self).__init__()

        self.num_directions = num_directions
        self.h_dim = enc_dim
        self.s_dim = dec_dim
        self.a_dim = self.s_dim if attn_dim is None else attn_dim
        self.build()

    def build(self):
        self.U = nn.Linear(self.h_dim * self.num_directions, self.a_dim)
        self.W = nn.Linear(self.s_dim, self.a_dim)
        self.v = nn.Linear(self.a_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def precmp_U(self, enc_outputs):
        """
        Precompute U matrix for computational efficiency.
        The result is a # SL x B x self.attn_dim matrix.

        :param enc_outputs: # SL x B x self.attn_dim matrix
        :return: input projected by the weight matrix of the attention module.
        """

        src_seq_len, batch_size, enc_dim = enc_outputs.size()
        enc_outputs_reshaped = enc_outputs.view(-1, self.h_dim)
        proj = self.U(enc_outputs_reshaped)
        proj_reshaped = proj.view(src_seq_len, batch_size, self.a_dim)

        return proj_reshaped

    def forward(self, prev_h_batch, enc_outputs):
        """

        :param prev_h_batch: 1 x B x dec_dim
        :param enc_outputs: SL x B x (num_directions * enc_dim)

        :return: attn weights: # B x SL

        """

        src_seq_len, batch_size, enc_dim = enc_outputs.size()

        # Compute U*h over length and batch (batch, source_l, attn_dim)
        uh = self.U(
            enc_outputs.view(-1, self.h_dim)).view(src_seq_len, batch_size, self.a_dim)  # SL x B x self.attn_dim

        # Compute W*s over the batch (batch, attn_dim)
        wq = self.W(prev_h_batch.view(-1, self.s_dim)).unsqueeze(0)  # 1 x B x self.a_dim

        # tanh( Wh*hj + Ws s_{i-1} )     (batch, source_l, dim)
        wq3d = wq.expand_as(uh)
        wquh = self.tanh(wq3d + uh)

        # v^T*wquh over length and batch
        attn_unnorm_scores = self.v(wquh.view(-1, self.a_dim)).view(batch_size, src_seq_len)

        attn_weights = self.softmax(attn_unnorm_scores)  # B x SL

        return attn_weights
