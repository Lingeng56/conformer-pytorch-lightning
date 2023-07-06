import torch.nn as nn
from feedforward import FeedForwardModule
from attention import RelativeMultiHeadSelfAttentionModule, MultiHeadSelfAttentionModule
from convolution import ConvolutionModule



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
                                      kernel_size,
                                      expansion_factor)

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



