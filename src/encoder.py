import torch.nn as nn
from convolution import ConvolutionSubSampling
from conformer import ConformerBlock
from attention import RelativePositionalEncoding, PositionalEncoding
from torchaudio.models import Conformer


class ConformerEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 kernel_size,
                 encoder_dim,
                 dropout,
                 linear_dim,
                 num_heads,
                 encoder_layer_nums,
                 max_len=5000,
                 use_relative=False):
        super(ConformerEncoder, self).__init__()
        if use_relative:
            self.position_encoding = RelativePositionalEncoding(encoder_dim, dropout, max_len)
        else:
            self.position_encoding = PositionalEncoding(encoder_dim, dropout, max_len)

        self.subsampling = ConvolutionSubSampling(in_channels=1, out_channels=encoder_dim)
        self.fc = nn.Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(encoder_dim,
                               kernel_size,
                               dropout,
                               linear_dim,
                               num_heads,
                               use_relative)
                for _ in range(encoder_layer_nums)
            ]
        )

        self.criterion = nn.CTCLoss()

    def forward(self, inputs, input_lengths):
        outputs, output_lengths = self.subsampling(inputs, input_lengths)
        outputs = self.fc(outputs)
        outputs, pos_embed = self.position_encoding(outputs)
        outputs = self.dropout(outputs)
        for block in self.conformer_blocks:
            outputs = block(outputs, output_lengths, pos_embed)
        # outputs, output_lengths = self.conformer_blocks(outputs, output_lengths)
        return outputs, output_lengths
