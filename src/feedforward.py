import torch.nn as nn


class PositionwiseFeedForwardModule(nn.Module):

    def __init__(self, input_dim, dropout, hidden_dim, activation='swish'):
        super(PositionwiseFeedForwardModule, self).__init__()
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        if activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, inputs):
        outputs = self.w_1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.w_2(outputs)
        return outputs


