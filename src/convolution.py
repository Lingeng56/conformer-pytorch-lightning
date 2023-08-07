import torch
import torch.nn as nn


class ConvolutionModule(nn.Module):

    def __init__(self, input_dim, kernel_size, bias=True):
        super(ConvolutionModule, self).__init__()
        self.pointwise_conv_one = nn.Conv1d(in_channels=input_dim,
                                            out_channels=input_dim * 2,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=bias,
                                            )
        self.glu = nn.GLU()
        self.depthwise_conv = nn.Conv1d(in_channels=input_dim,
                                        out_channels=input_dim,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=(kernel_size - 1) // 2,
                                        bias=bias,
                                        )
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.swish = nn.SiLU()
        self.pointwise_conv_two = nn.Conv1d(in_channels=input_dim,
                                            out_channels=input_dim * 2,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0
                                            )

    def forward(self, inputs, inputs_pad_mask):
        inputs = inputs.transpose(1, 2)
        if inputs_pad_mask.size(2) > 0:
            inputs = inputs.masked_fill(~inputs_pad_mask, 0.0)
        outputs = self.pointwise_conv_one(inputs)
        outputs = self.glu(outputs)
        outputs = self.depthwise_conv(outputs)
        outputs = self.batch_norm(outputs.permute(0, 2, 1)).permute(0, 2, 1)
        outputs = self.swish(outputs)
        outputs = self.pointwise_conv_two(outputs)
        if inputs_pad_mask.size(2) > 0:
            outputs = outputs.masked_fill(~inputs_pad_mask, 0.0)
        return outputs.transpose(1, 2)


class ConvolutionSubSampling(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 pos_enc):
        super(ConvolutionSubSampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, 3, 2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(output_dim * (((input_dim - 1) // 2 - 1) // 2), output_dim)
        )
        self.positional_encoding = pos_enc

    def forward(self, inputs, inputs_pad_mask, offset=0):
        inputs = inputs.unsqueeze(1)
        outputs = self.conv(inputs)
        batch_size, channel, seq_len, feature_dim = outputs.size()
        outputs = self.out(outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, channel * feature_dim))
        outputs, pos_embed = self.positional_encoding(outputs, offset)
        return outputs, pos_embed, inputs_pad_mask[:, :, 2::2][:, :, 2::2]
