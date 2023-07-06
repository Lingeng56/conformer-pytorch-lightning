import torch.nn as nn
from convolution import ConvolutionSubSampling
from conformer import ConformerBlock


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
        self.subsampling = ConvolutionSubSampling(in_channels=1, out_channels=encoder_dim)
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
        for block in self.conformer_blocks:
            outputs = block(outputs)
        return outputs, output_lengths
