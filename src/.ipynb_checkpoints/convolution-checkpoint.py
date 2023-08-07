import torch
import torch.nn as nn


class ConvolutionModule(nn.Module):

    def __init__(self, input_dim, kernel_size, expansion_factor):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.pointwise_conv_one = nn.Conv1d(in_channles=input_dim,
                                            output_channels=input_dim * expansion_factor,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0
                                           )
        self.glu = nn.GLU()
        self.depthwise_conv = nn.Conv1d(in_channels=input_dim * expansion_factor // 2,
                                        out_channels=input_dim,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=(kernel_size - 1) // 2,
                                        )
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.swish = nn.SiLU()
        self.pointwise_conv_two = nn.Conv1d(in_channles=input_dim,
                                            output_channels=input_dim * expansion_factor,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0
                                           )
        self.dropout = nn.Dropout()

    def forward(self, inputs, input_lengths):
        mask = torch.arange(input_lengths.max().item())[None, :] < input_lengths[:, None]
        mask = mask.unsqueeze(-1).expand_as(inputs)
        inputs = inputs * mask.float()
        inputs = self.layer_norm(inputs)
        inputs = inputs.transpose(1, 2)
        outputs = self.pointwise_conv_one(inputs)
        outputs = self.glu(outputs)
        outputs = self.depthwise_conv(outputs)
        outputs = self.batch_norm(outputs.permute(0, 2, 1)).permute(0, 2, 1)
        outputs = self.swish(outputs)
        outputs = self.pointwise_conv_two(outputs)
        outputs = outputs * mask.float()
        return outputs


class ConvolutionSubSampling(nn.Module):

    def __init__(self, 
                 input_dim, 
                 out_channels, 
                 pos_enc):
        super(ConvolutionSubSampling, self).__init__()
        self.conv_one = MaskConv2d(nn.Conv2d(1, out_channels, kernel_size=3, stride=2))
        self.relu_one = nn.ReLU()
        self.conv_two = MaskConv2d(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2))
        self.relu_two = nn.ReLU()
        self.out = nn.Linear(out_channels * (((input_dim - 1) // 2 - 1) //2), out_channels)
        self.positional_encoding = pos_enc

    def forward(self, inputs, input_lengths):
        inputs = inputs.unsqueeze(1)
        outputs = self.conv_one(outputs)
        outputs = self.relu_one(outputs)
        outputs = self.conv_two(outputs)
        outputs = self.relu_two(outputs)
        batch_size, channel, seq_len, feature_dim = outputs.shape
        outputs = self.out(outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim * channel))
        outputs, pos_emb = self.positional_encoding(outputs)
        output_lengths = input_lengths // 4
        return outputs, pos_emb, output_lengths
