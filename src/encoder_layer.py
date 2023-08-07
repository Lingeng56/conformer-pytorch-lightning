import torch
import torch.nn as nn
from feedforward import FeedForwardModule
from attention import RelativeMultiHeadSelfAttentionModule, MultiHeadSelfAttentionModule
from convolution import ConvolutionModule



class ConformerEncoderLayer(nn.Module):

    def __init__(self,
                 encoder_dim,
                 kernel_size,
                 feedforward_dropout,
                 attention_dropout,
                 hidden_dim,
                 num_heads,
                 use_relative):
        super(ConformerEncoderLayer, self).__init__()
        self.feedforward_one = FeedForwardModule(encoder_dim,
                                                 feedforward_dropout,
                                                 hidden_dim)

        if use_relative:
            self.attention = RelativeMultiHeadSelfAttentionModule(encoder_dim,
                                                                  num_heads,
                                                                  attention_dropout)
        else:
            self.attention = MultiHeadSelfAttentionModule(encoder_dim,
                                                          num_heads,
                                                          attention_dropout)

        self.conv = ConvolutionModule(encoder_dim,
                                      kernel_size,
                                      hidden_dim)

        self.feedforward_two = FeedForwardModule(encoder_dim,
                                                 feedforward_dropout,
                                                 hidden_dim)

        self.feedfoward_norm= nn.LayerNorm(encoder_dim, eps=1e-5)
        self.attention_norm = nn.LayerNorm(encoder_dim, eps=1e-5)
        self.conv_norm = nn.LayerNorm(encoder_dim, eps=1e-5)
        self.final_norm = nn.LayerNorm(encoder_dim, eps=1e-5)
        self.dropout = nn.Dropout(feedforward_dropout)


    def forward(self,
                inputs,
                inputs_attn_mask,
                pos_embed,
                inputs_pad_mask=torch.ones((0, 0, 0), dtype=torch.bool)):
        outputs = inputs + 0.5 * self.dropout(self.feedforward_one(inputs))
        outputs = self.feedfoward_norm(outputs)
        outputs = outputs + self.dropout(self.attention(outputs, outputs, outputs, inputs_attn_mask, pos_embed))
        outputs = self.attention_norm(outputs)
        outputs = outputs + self.dropout(self.conv(outputs, inputs_pad_mask))
        outputs = self.conv_norm(outputs)
        outputs = outputs + 0.5 * self.dropout(self.feedforward_two(outputs))
        outputs = self.feedfoward_norm(outputs)
        outputs = self.final_norm(outputs)
        return outputs, inputs_attn_mask



