import torch
import torch.nn as nn


class Predictor(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_size,
                 output_size,
                 hidden_size,
                 embed_dropout,
                 num_layers,
                 bias=True,
                 dropout=0.1):
        super(Predictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            batch_first=True,
                            dropout=dropout)
        self.projection = nn.Linear(hidden_size, output_size)



    def init_state(
            self,
            inputs
    ):

        return [
            torch.zeros(1 * self.num_layers,
                        inputs.size(0),
                        self.hidden_size,
                        device=inputs.device),
            torch.zeros(1 * self.num_layers,
                        inputs.size(0),
                        self.hidden_size,
                        device=inputs.device)
        ]



    def forward(self,
                inputs,
                states=None):
        embed = self.embed(inputs)
        embed = self.dropout(embed)

        if states is None:
            states = self.init_state(inputs)


        outputs, states = self.lstm(embed, states)
        outputs = self.projection(outputs)

        return outputs
