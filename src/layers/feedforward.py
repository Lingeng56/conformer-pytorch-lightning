import torch.nn as nn


class FeedForwardModule(nn.Module):

    def __init__(self, input_dim, dropout, expansion_factor):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim * expansion_factor)
        self.swish = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim * expansion_factor, input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = self.layer_norm(inputs)
        outputs = self.fc1(outputs)
        outputs = self.swish(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.fc2(outputs)
        outputs = self.dropout2(outputs)
        outputs = inputs + 0.5 * outputs
        return outputs


