import torch
import torch.nn as nn


class EncoderMLP(nn.Module):
    def __init__(self, config):
        super(EncoderMLP, self).__init__()

        self.config = config
        self.input_size = self.config['input_size']
        self.hidden_size = self.config['hidden_size']
        self.W = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, input_embedded):
        seq_len, batch_size, emb_dim = input_embedded.size()

        # f(Wx), where f(x) is a non-linear function, e.g. ReLU
        outputs = self.relu(self.W(input_embedded.view(-1, emb_dim)))

        # Reshape outputs:  # SL x B x H
        outputs = outputs.view(seq_len, batch_size, -1)
        dec_hidden = torch.sum(outputs, 0)
        return outputs, dec_hidden.unsqueeze(0)

    @property
    def num_directions(self):
        return 1
