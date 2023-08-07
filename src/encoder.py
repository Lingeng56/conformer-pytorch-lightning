import torch.nn as nn
from convolution import ConvolutionSubSampling
from encoder_layer import ConformerEncoderLayer
from attention import RelativePositionalEncoding, PositionalEncoding
from utils import make_pad_mask, make_attn_mask


class ConformerEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 kernel_size,
                 encoder_dim,
                 dropout,
                 attention_dropout,
                 pos_enc_dropout,
                 hidden_dim,
                 num_heads,
                 encoder_num_layers,
                 max_len=5000,
                 use_relative=False):
        super(ConformerEncoder, self).__init__()
        if use_relative:
            self.position_encoding = RelativePositionalEncoding(encoder_dim, pos_enc_dropout, max_len)
        else:
            self.position_encoding = PositionalEncoding(encoder_dim, pos_enc_dropout, max_len)

        self.subsampling = ConvolutionSubSampling(input_dim=input_dim, output_dim=encoder_dim, pos_enc=self.position_encoding)
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerEncoderLayer(encoder_dim,
                                      kernel_size,
                                      dropout,
                                      attention_dropout,
                                      hidden_dim,
                                      num_heads,
                                      use_relative)
                for _ in range(encoder_num_layers)
            ]
        )
        self.encoder_dim = encoder_dim


    def forward(self, inputs, input_lengths):
        max_seq_len = inputs.size(1)
        inputs_pad_mask = ~make_pad_mask(input_lengths, max_seq_len).unsqueeze(1)
        outputs, pos_embed,  inputs_pad_mask = self.subsampling(inputs, inputs_pad_mask)
        inputs_attn_mask = ~make_attn_mask(inputs, inputs_pad_mask)
        for block in self.conformer_blocks:
            outputs, inputs_attn_mask = block(outputs, inputs_attn_mask, pos_embed, inputs_pad_mask)
        return outputs, inputs_pad_mask
