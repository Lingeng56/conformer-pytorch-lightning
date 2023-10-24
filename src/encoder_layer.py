import torch
import torch.nn as nn
from feedforward import PositionwiseFeedForwardModule
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
        self.feed_forward = PositionwiseFeedForwardModule(encoder_dim,
                                                          feedforward_dropout,
                                                          hidden_dim)

        if use_relative:
            self.self_attn = RelativeMultiHeadSelfAttentionModule(encoder_dim,
                                                                  num_heads,
                                                                  attention_dropout)
        else:
            self.self_attn = MultiHeadSelfAttentionModule(encoder_dim,
                                                          num_heads,
                                                          attention_dropout)

        self.conv_module = ConvolutionModule(encoder_dim,
                                             kernel_size,
                                             hidden_dim)

        self.feed_forward_macaron = PositionwiseFeedForwardModule(encoder_dim,
                                                                  feedforward_dropout,
                                                                  hidden_dim)

        self.norm_ff= nn.LayerNorm(encoder_dim, eps=1e-5)
        self.norm_ff_macaron= nn.LayerNorm(encoder_dim, eps=1e-5)
        self.norm_mha = nn.LayerNorm(encoder_dim, eps=1e-5)
        self.norm_conv = nn.LayerNorm(encoder_dim, eps=1e-5)
        self.norm_final = nn.LayerNorm(encoder_dim, eps=1e-5)
        self.dropout = nn.Dropout(feedforward_dropout)


    def forward(self,
                inputs,
                inputs_attn_mask,
                pos_embed,
                inputs_pad_mask=torch.ones((0, 0, 0), dtype=torch.bool),
                attn_cache=torch.ones((0, 0, 0), dtype=torch.bool),
                cnn_cache=torch.ones((0, 0, 0), dtype=torch.bool),):
        residual = inputs
        outputs = self.norm_ff_macaron(inputs)
        outputs = residual + 0.5 * self.dropout(self.feed_forward_macaron(outputs))
        residual = outputs
        outputs = self.norm_mha(outputs)
        outputs, new_attn_cache = self.self_attn(outputs, outputs, outputs, inputs_attn_mask, pos_embed, attn_cache)
        outputs = residual + self.dropout(outputs)
        residual = outputs
        outputs = self.norm_conv(outputs)
        outputs, new_cnn_cache = self.conv_module(outputs, inputs_pad_mask, cnn_cache)
        outputs = residual + self.dropout(outputs)
        residual = outputs
        outputs = self.norm_ff(outputs)
        outputs = residual + 0.5 * self.dropout(self.feed_forward(outputs))
        outputs = self.norm_final(outputs)
        return outputs, inputs_attn_mask, new_attn_cache, new_cnn_cache





