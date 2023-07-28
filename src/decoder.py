import torch.nn as nn
import torch


class View(nn.Module):

    def __init__(self, shape, contiguous=True):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous


    def forward(self, inputs):
        return inputs.view(self.shape).contiguous() if self.contiguous else inputs.view(self.shape)


class LSTMAttentionDecoder(nn.Module):

    def __init__(self, hidden_state_dim, decoder_layer_nums, num_heads, dropout, vocab_size, max_len):
        super(LSTMAttentionDecoder, self).__init__()
        self.max_len = max_len
        self.hidden_state_dim = hidden_state_dim
        self.decoder_layer_nums = decoder_layer_nums
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(
            input_size=self.hidden_state_dim,
            hidden_size=self.hidden_state_dim,
            num_layers=self.decoder_layer_nums,
            batch_first=True,
            dropout=dropout,
            bias=True,
            bidirectional=False
                            )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_state_dim << 1, self.hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            nn.Linear(self.hidden_state_dim, self.vocab_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.vocab_size,
                                      self.hidden_state_dim)
        self.attention = nn.MultiheadAttention(self.hidden_state_dim,
                                               num_heads,
                                               batch_first=True)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward_step(self, input_var, hidden_states, encoder_outputs):
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.dropout(embedded)

        if self.training:
            self.lstm.flatten_parameters()

        outputs, hidden_states = self.lstm(embedded, hidden_states)

        context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)

        outputs = torch.cat((outputs, context), dim=2)

        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1))
        step_outputs = self.softmax(step_outputs)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states



    def forward(self, encoder_outputs, targets=None):
        hidden_states, attn = None, None
        targets, batch_size, max_length = self.validate_args(encoder_outputs, targets)
        input_var = targets[:, 0].unsqueeze(1)
        logits = list()

        for di in range(max_length):
            step_outputs, hidden_states = self.forward_step(
                input_var=input_var,
                hidden_states=hidden_states,
                encoder_outputs=encoder_outputs
            )
            logits.append(step_outputs)
            input_var = logits[-1].topk(1)[1]

        logits = torch.stack(logits, dim=1)

        return logits



    def validate_args(self, encoder_outputs, targets=None):
        batch_size = encoder_outputs.size(0)

        if targets is None:
            targets = torch.LongTensor([2] * batch_size).view(batch_size, 1).to(encoder_outputs.device)
            max_length = self.max_len

        else:
            max_length = targets.size(1)

        return targets, batch_size, max_length


