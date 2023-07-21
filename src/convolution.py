import torch
import torch.nn as nn


class ConvolutionModule(nn.Module):

    def __init__(self, input_dim, kernel_size, expansion_factor):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.pointwise_conv_one = nn.Linear(input_dim, input_dim * expansion_factor)
        self.glu = nn.GLU()
        self.depthwise_conv = nn.Conv1d(in_channels=input_dim,
                                        out_channels=input_dim,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=(kernel_size - 1) // 2,
                                        )
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.swish = nn.SiLU()
        self.pointwise_conv_two = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout()

    def forward(self, inputs):
        outputs = self.layer_norm(inputs)
        outputs = self.pointwise_conv_one(outputs)
        outputs = self.glu(outputs)
        outputs = self.depthwise_conv(outputs.permute(0, 2, 1))
        outputs = self.batch_norm(outputs)
        outputs = self.swish(outputs).permute(0, 2, 1)
        outputs = self.pointwise_conv_two(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs + inputs
        return outputs


class MaskConv2d(nn.Module):

    def __init__(self, module):
        super(MaskConv2d, self).__init__()
        self.module = module


    def forward(self, inputs, seq_lengths):
        numerator = seq_lengths + 2 * self.module.padding[1] - self.module.dilation[1] * (
                self.module.kernel_size[1] - 1) - 1
        seq_lengths = numerator.float() / float(self.module.stride[1])
        seq_lengths = (seq_lengths.int() + 1).int()
        outputs = self.module(inputs)
        mask = torch.BoolTensor(outputs.size()).fill_(0).to(outputs.device)
        for idx, length in enumerate(seq_lengths):
            length = length.item()

            if mask[idx].size(2) - length > 0:
                mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

        outputs = outputs.masked_fill(mask, 0)
        return outputs, seq_lengths




class ConvolutionSubSampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvolutionSubSampling, self).__init__()
        self.conv_one = MaskConv2d(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2))
        self.relu_one = nn.ReLU()
        self.conv_two = MaskConv2d(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2))
        self.relu_two = nn.ReLU()


    def forward(self, inputs, input_lengths):
        outputs, output_lengths = self.conv_one(inputs.unsqueeze(1), input_lengths)
        outputs = self.relu_one(outputs)
        outputs, output_lengths = self.conv_two(outputs, output_lengths)
        outputs = self.relu_two(outputs)
        batch_size, channels, seq_len, feature_dim = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, feature_dim * channels)
        return outputs, output_lengths
