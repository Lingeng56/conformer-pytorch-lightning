import torch
import torch.nn as nn
from attention import MultiHeadSelfAttentionModule
from feedforward import FeedForwardModule


class TransformerDecoderLayer(nn.Module):


    def __init__(self,
                 decoder_dim,
                 num_heads,
                 hidden_dim,
                 dropout,
                 self_attention_dropout,
                 src_attention_dropout
                 ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttentionModule(
            encoder_dim=decoder_dim,
            num_heads=num_heads,
            dropout=self_attention_dropout
        )
        self.src_attention = MultiHeadSelfAttentionModule(
            encoder_dim=decoder_dim,
            num_heads=num_heads,
            dropout=src_attention_dropout
        )
        self.feedforward = FeedForwardModule(
            input_dim=decoder_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(decoder_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(decoder_dim, eps=1e-5)
        self.norm3 = nn.LayerNorm(decoder_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)


    def forward(self,
                tgt,
                tgt_mask,
                memory,
                memory_mask,
                cache=None):

        if cache is None:
            tgt_q = self.norm1(tgt)
            tgt_q_mask = tgt_mask
            outputs = tgt + self.dropout(self.self_attention(tgt_q, tgt_q, tgt_q, tgt_q_mask))
        else:
            tgt_q = self.norm1(tgt)[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]
            outputs = tgt[:, -1:, :] + self.dropout(self.self_attention(tgt_q, self.norm1(tgt), self.norm1(tgt), tgt_q_mask))

        outputs = outputs + self.dropout(self.src_attention(self.norm2(outputs), memory, memory, memory_mask))
        outputs = outputs + self.dropout(self.feedforward(self.norm3(outputs)))

        if cache is not None:
            outputs = torch.cat([cache, outputs], dim=1)

        return outputs, tgt_mask, memory, memory_mask



