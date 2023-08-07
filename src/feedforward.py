import torch.nn as nn


class FeedForwardModule(nn.Module):

    def __init__(self, input_dim, dropout, hidden_dim):
        super(FeedForwardModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.swish(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return outputs


