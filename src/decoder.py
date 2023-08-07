import torch.nn as nn
from attention import PositionalEncoding
from decoder_layer import TransformerDecoderLayer
from utils import *


class CTCDecoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 encoder_dim,
                 dropout):
        super(CTCDecoder, self).__init__()
        self.proj = nn.Linear(encoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CTCLoss(reduction='sum')

    def forward(self, encoder_out, encoder_out_lens, padded_labels, label_lengths):
        logits = self.proj(self.dropout(encoder_out))
        probs = logits.transpose(0, 1).log_softmax(2)
        loss = self.criterion(probs, padded_labels, encoder_out_lens, label_lengths)
        loss = loss / padded_labels.size(0)
        return loss


class TransformerDecoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 decoder_dim,
                 num_heads,
                 hidden_dim,
                 num_layers,
                 dropout,
                 positional_dropout,
                 self_attention_dropout,
                 src_attention_dropout
                 ):
        super(TransformerDecoder, self).__init__()
        self.embed = nn.Sequential(nn.Embedding(vocab_size, decoder_dim),
                                   PositionalEncoding(decoder_dim, positional_dropout))
        self.norm = nn.LayerNorm(decoder_dim, eps=1e-5)
        self.proj = nn.Linear(decoder_dim, vocab_size)
        self.decoders = nn.ModuleList([
            TransformerDecoderLayer(decoder_dim,
                                    num_heads,
                                    hidden_dim,
                                    dropout,
                                    self_attention_dropout,
                                    src_attention_dropout)
            for _ in range(num_layers)
        ])

    def forward(self,
                memory,
                memory_mask,
                targets,
                target_lengths,
                r_targets,
                reverse_weight
                ):
        max_len = targets.size(1)
        targets_mask = ~make_pad_mask(target_lengths, max_len).unsqueeze(1).to(targets.device)
        outputs, _ = self.embed(targets)
        m = make_subsequent_mask(targets_mask.size(-1), targets_mask.device).unsuqeeze(0)
        targets_mask = targets_mask & m
        for layer in self.decoders:
            outputs, targets_mask, memory, memory_mask = layer(outputs, targets_mask, memory, memory_mask)

        outputs = self.norm(outputs)
        outputs = self.proj(outputs)
        output_lengths = targets_mask.sum(dim=1)
        return outputs, torch.tensor(0.0), output_lengths


class BiTransformerDecoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 decoder_dim,
                 num_heads,
                 hidden_dim,
                 num_layers,
                 r_num_layers,
                 dropout,
                 pos_enc_dropout,
                 self_attention_dropout,
                 src_attention_dropout):
        super(BiTransformerDecoder, self).__init__()
        self.left_encoder = TransformerDecoder(vocab_size,
                                               decoder_dim,
                                               num_heads,
                                               hidden_dim,
                                               num_layers,
                                               dropout,
                                               pos_enc_dropout,
                                               self_attention_dropout,
                                               src_attention_dropout)
        self.right_encoder = TransformerDecoder(vocab_size,
                                                decoder_dim,
                                                num_heads,
                                                hidden_dim,
                                                r_num_layers,
                                                dropout,
                                                pos_enc_dropout,
                                                self_attention_dropout,
                                                src_attention_dropout)


    def forward(self,
                memory,
                memory_mask,
                targets,
                target_lengths,
                r_targets,
                reverse_weight):
        left_outputs, right_outputs, output_lengths = self.left_encoder(memory, memory_mask, targets, target_lengths)
        if reverse_weight > 0.0:
            right_outputs, _, output_lengths = self.right_encoder(memory, memory_mask, r_targets, target_lengths)

        return left_outputs, right_outputs, output_lengths
