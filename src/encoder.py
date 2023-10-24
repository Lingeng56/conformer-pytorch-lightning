import torch
import torch.nn as nn
from convolution import ConvolutionSubSampling
from encoder_layer import ConformerEncoderLayer
from attention import RelativePositionalEncoding, PositionalEncoding
from utils import make_pad_mask, make_attn_mask


class ConformerEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 kernel_size,
                 encoder_dim,
                 dropout,
                 attention_dropout,
                 pos_enc_dropout,
                 hidden_dim,
                 num_heads,
                 encoder_num_layers,
                 cmvn=None,
                 max_len=5000,
                 use_relative=False,
                 use_dynamic_chunk_size=False,
                 use_dynamic_left_chunk=False,
                 static_chunk_size=-1,
                 ):
        super(ConformerEncoder, self).__init__()
        if use_relative:
            self.position_encoding = RelativePositionalEncoding(encoder_dim, pos_enc_dropout, max_len)
        else:
            self.position_encoding = PositionalEncoding(encoder_dim, pos_enc_dropout, max_len)

        self.embed = ConvolutionSubSampling(input_dim=input_dim, output_dim=encoder_dim, pos_enc=self.position_encoding)
        self.encoders = nn.ModuleList(
            [
                ConformerEncoderLayer(encoder_dim,
                                      kernel_size,
                                      dropout,
                                      attention_dropout,
                                      hidden_dim,
                                      num_heads,
                                      use_relative)
                for _ in range(encoder_num_layers)
            ]
        )
        self.encoder_dim = encoder_dim
        self.after_norm = nn.LayerNorm(encoder_dim, eps=1e-5)
        self.global_cmvn = cmvn
        self.use_dynamic_chunk_size = use_dynamic_chunk_size
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.static_chunk_size = static_chunk_size

    def forward(self,
                inputs,
                input_lengths,
                decoding_chunk_size=0,
                num_decoding_chunk_size=-1):
        if self.global_cmvn is not None:
            inputs = self.global_cmvn(inputs)
        max_seq_len = inputs.size(1)
        inputs_pad_mask = ~make_pad_mask(input_lengths, max_seq_len).unsqueeze(1)
        outputs, pos_embed, inputs_pad_mask = self.embed(inputs, inputs_pad_mask)
        inputs_attn_mask = make_attn_mask(inputs,
                                          inputs_pad_mask,
                                          self.use_dynamic_chunk_size,
                                          self.use_dynamic_left_chunk,
                                          decoding_chunk_size,
                                          self.static_chunk_size,
                                          num_decoding_chunk_size
                                          )
        for block in self.encoders:
            outputs, inputs_attn_mask, _, _ = block(outputs, inputs_attn_mask, pos_embed, inputs_pad_mask)
        outputs = self.after_norm(outputs)
        return outputs, inputs_pad_mask

    def forward_chunk(self,
                      inputs,
                      offset,
                      required_cache_size,
                      attn_cache,
                      cnn_cache,
                      inputs_attn_mask=torch.ones((0, 0, 0))):
        tmp_masks = torch.ones(1,
                               inputs.size(1),
                               device=inputs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            inputs = self.global_cmvn(inputs)
        outputs, pos_embed, _ = self.embed(inputs, tmp_masks, offset)
        num_layers, cache_size = attn_cache.size(0), attn_cache.size(2)
        chunk_size = outputs.size(1)
        attention_key_size = cache_size + chunk_size
        pos_embed = self.embed.position_encoding(
            offset=offset - cache_size, size=attention_key_size
        )
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_attn_cache = []
        r_cnn_cache = []
        for idx, block in enumerate(self.encoders):
            outputs, _, new_attn_cache, new_cnn_cache = block(
                outputs,
                inputs_attn_mask,
                pos_embed,
                attn_cache=attn_cache[idx: idx + 1] if num_layers > 0 else attn_cache,
                cnn_cache=cnn_cache[idx] if cache_size > 0 else cnn_cache
            )
            r_attn_cache.append(new_attn_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))

        outputs = self.after_norm(outputs)
        r_attn_cache = torch.cat(r_attn_cache, dim=0)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)
        return outputs, r_attn_cache, r_cnn_cache

    def forward_chunk_by_chunk(self,
                               inputs,
                               decoding_chunk_size,
                               num_decoding_left_chunks=-1):
        subsampling_rate = 4
        context = 7
        stride = subsampling_rate * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling_rate + context
        num_frames = inputs.size(1)
        attn_cache = torch.zeros((0, 0, 0, 0), device=inputs.device)
        cnn_cache = torch.zeros((0, 0, 0, 0), device=inputs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_inputs = inputs[:, cur:end, :]
            chunk_outputs, attn_cache, cnn_cache = self.forward_chunk(inputs=chunk_inputs,
                                                                      offset=offset,
                                                                      required_cache_size=required_cache_size,
                                                                      attn_cache=attn_cache,
                                                                      cnn_cache=cnn_cache
                                                                      )
            outputs.append(chunk_outputs)
            offset += chunk_outputs.size(1)
        outputs = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, outputs.size(1)))
        return outputs, masks
