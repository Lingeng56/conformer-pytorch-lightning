import torch
import torch.nn as nn


def ApplyPadding(input, padding, pad_value) -> torch.Tensor:
    """
    Args:
        input:   [bs, max_time_step, dim]
        padding: [bs, max_time_step]
    """
    return padding * pad_value + input * (1 - padding)


class RNNPredictor(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_size,
                 output_size,
                 hidden_size,
                 embed_dropout,
                 num_layers,
                 bias=True,
                 dropout=0.1):
        super(RNNPredictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(embed_dropout)
        self.rnn = nn.LSTM(input_size=embed_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bias=bias,
                           batch_first=True,
                           dropout=dropout)
        self.projection = nn.Linear(hidden_size, output_size)
        self.embed_size = embed_size


    def init_state(
            self,
            inputs,
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
            states = (states[0].to(embed.dtype), states[1].to(embed.dtype))


        outputs, states = self.rnn(embed, states)
        outputs = self.projection(outputs)

        return outputs


    def forward_step(self, inputs, padding, cache):
        state_m, state_c = cache
        embed = self.embed(inputs)
        embed = self.dropout(embed)
        out, (m, c) = self.rnn(embed, (state_m, state_c))

        out = self.projection(out)
        m = ApplyPadding(m, padding.unsqueeze(0), state_m)
        c = ApplyPadding(c, padding.unsqueeze(0), state_c)

        return out, (m, c)

