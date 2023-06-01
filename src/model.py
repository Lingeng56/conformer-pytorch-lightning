import torch
import torch.nn as nn
import math
import pytorch_lightning as pl


class ConformerEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 kernel_size,
                 encoder_dim,
                 dropout,
                 expansion_factor,
                 num_heads,
                 encoder_layer_nums,
                 max_len,
                 use_relative=False):
        super(ConformerEncoder, self).__init__()
        self.subsampling = ConvolutionSubSampling(in_channels=input_dim, out_channels=encoder_dim)
        self.fc = nn.Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(encoder_dim,
                               kernel_size,
                               dropout,
                               expansion_factor,
                               num_heads,
                               max_len,
                               use_relative)
                for _ in range(encoder_layer_nums)
            ]
        )

        self.criterion = nn.CTCLoss()

    def forward(self, inputs, input_lengths):
        outputs, output_lengths = self.subsampling(inputs, input_lengths)
        outputs = self.fc(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conformer_blocks(outputs)
        return outputs, output_lengths


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


class ConformerBlock(nn.Module):

    def __init__(self, encoder_dim, kernel_size, dropout, expansion_factor, num_heads, max_len, use_relative):
        super(ConformerBlock, self).__init__()
        self.feed_forward_one = FeedForwardModule(encoder_dim,
                                                  dropout,
                                                  expansion_factor)

        if use_relative:
            self.attention = RelativeMultiHeadSelfAttentionModule(encoder_dim,
                                                                  num_heads,
                                                                  dropout,
                                                                  max_len)
        else:
            self.attention = MultiHeadSelfAttentionModule(encoder_dim,
                                                          num_heads,
                                                          dropout,
                                                          max_len)

        self.conv = ConvolutionModule(encoder_dim,
                                      kernel_size)

        self.feed_forward_two = FeedForwardModule(encoder_dim,
                                                  dropout,
                                                  expansion_factor)

        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, inputs):
        outputs = self.feed_forward_one(inputs)
        outputs = self.attention(outputs)
        outputs = self.conv(outputs)
        outputs = self.feed_forward_two(outputs)
        outputs = self.layer_norm(outputs)
        return outputs


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


class ConvolutionModule(nn.Module):

    def __init__(self, input_dim, kernel_size):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.pointwise_conv_one = nn.Linear(input_dim, input_dim * 2)
        self.glu = nn.GLU()
        self.depthwise_conv = nn.Conv1d(in_channels=input_dim,
                                        out_channels=input_dim,
                                        kernel_size=kernel_size // 2,
                                        groups=input_dim)
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.swish = nn.SiLU()
        self.pointwise_conv_two = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout()

    def forward(self, inputs):
        outputs = self.layer_norm(inputs)
        outputs = self.pointwise_conv_one(outputs)
        outputs = self.glu(outputs)
        outputs = self.depthwise_conv(outputs)
        outputs = self.batch_norm(outputs)
        outputs = self.swish(outputs)
        outputs = self.pointwise_conv_two(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs + inputs
        return outputs


class ConvolutionSubSampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvolutionSubSampling, self).__init__()
        self.conv_one = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.relu_one = nn.ReLU()
        self.conv_two = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.relu_two = nn.ReLU()

    def forward(self, inputs, input_lengths):
        outputs = self.conv_one(inputs.unsqueeze(1))
        outputs = self.relu_one(outputs)
        outputs = self.conv_two(outputs)
        outputs = self.relu_two(outputs)
        batch_size, channels, seq_len, feature_dim = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, feature_dim * channels)
        output_lengths = input_lengths >> 2 - 1
        return outputs, output_lengths

