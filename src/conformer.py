import torch
import torch.nn as nn
from feedforward import FeedForwardModule
from attention import RelativeMultiHeadSelfAttentionModule, MultiHeadSelfAttentionModule
from convolution import ConvolutionModule



class ConformerBlock(nn.Module):

    def __init__(self, encoder_dim, kernel_size, dropout, expansion_factor, num_heads, use_relative):
        super(ConformerBlock, self).__init__()
        self.feed_forward_one = FeedForwardModule(encoder_dim,
                                                  dropout,
                                                  expansion_factor)

        if use_relative:
            self.attention = RelativeMultiHeadSelfAttentionModule(encoder_dim,
                                                                  num_heads,
                                                                  dropout)
        else:
            self.attention = MultiHeadSelfAttentionModule(encoder_dim,
                                                          num_heads,
                                                          dropout)

        self.conv = ConvolutionModule(encoder_dim,
                                      kernel_size,
                                      expansion_factor)

        self.feed_forward_two = FeedForwardModule(encoder_dim,
                                                  dropout,
                                                  expansion_factor)

        self.layer_norm = nn.LayerNorm(encoder_dim, eps=1e-5)

    def forward(self, inputs, input_lengths, pos_embed):
        outputs = self.feed_forward_one(inputs)
        outputs = self.attention(outputs, pos_embed)
        outputs = self.conv(outputs, input_lengths)
        outputs = self.feed_forward_two(outputs)
        outputs = self.layer_norm(outputs)
        return outputs



