import torch.nn as nn


class FeedForwardModule(nn.Module):

    def __init__(self, input_dim, dropout, expansion_factor):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-5)
        self.fc1 = nn.Linear(input_dim, int(input_dim * expansion_factor))
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(int(input_dim * expansion_factor), input_dim)

    def forward(self, inputs):
        outputs = self.layer_norm(inputs)
        outputs = self.fc1(outputs)
        outputs = self.swish(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return outputs


