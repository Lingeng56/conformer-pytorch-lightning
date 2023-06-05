import torch
import torch.nn as nn
import math




class RelativePositionalEncoding(nn.Module):

    def __init__(self):
        super(RelativePositionalEncoding, self).__init__()
        pass


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, inputs):
        x = inputs + self.pe[:inputs.size(0)]
        return self.dropout(x)


class RelativeMultiHeadSelfAttentionModule(nn.Module):

    def __init__(self, encoder_dim, num_heads, dropout, max_len):
        super(RelativeMultiHeadSelfAttentionModule, self).__init__()
        pass


class MultiHeadSelfAttentionModule(nn.Module):

    def __init__(self, encoder_dim, num_heads, dropout, max_len):
        super(MultiHeadSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(encoder_dim,
                                                      dropout,
                                                      max_len)
        self.attention = nn.MultiheadAttention(
            encoder_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.w_key = nn.Linear(encoder_dim, encoder_dim)
        self.w_query = nn.Linear(encoder_dim, encoder_dim)
        self.w_value = nn.Linear(encoder_dim, encoder_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        keys = self.w_key(inputs)
        queries = self.w_query(inputs)
        values = self.w_value(inputs)
        attn_output, _ = self.attention(keys, queries, values)
        attn_output = self.dropout(attn_output)
        attn_output = inputs + attn_output
        return attn_output

