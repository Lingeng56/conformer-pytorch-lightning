import torch.nn as nn


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
