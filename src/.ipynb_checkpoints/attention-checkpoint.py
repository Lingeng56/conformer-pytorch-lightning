import torch
import torch.nn as nn
import math


class RelativePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super(RelativePositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, inputs):
        pos_embed = self.pe[:inputs.size(0)].to(inputs.device)
        return self.dropout(inputs), self.dropout(pos_embed)



class RelativeMultiHeadSelfAttentionModule(nn.Module):


    def __init__(self, encoder_dim, num_heads, dropout):
        super(RelativeMultiHeadSelfAttentionModule, self).__init__()
        self.d_k = encoder_dim // num_heads
        self.num_heads = num_heads
        self.pos_projection = nn.Linear(encoder_dim, encoder_dim)
        self.w_key = nn.Linear(encoder_dim, encoder_dim)
        self.w_query = nn.Linear(encoder_dim, encoder_dim)
        self.w_value = nn.Linear(encoder_dim, encoder_dim)
        self.projection = nn.Linear(encoder_dim, encoder_dim)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.d_k))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(encoder_dim, eps=1e-5)

        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)


    def forward(self, inputs, input_lengths, pos_embed):
        batch_size = inputs.size(0)
        mask = torch.arange(input_lengths.max().item())[None, :] < input_lengths[:, None]
        mask = mask.unsqueeze(1).repeat(batch_size, 1, 1)
        inputs = self.layer_norm(inputs)
        q = self.w_query(inputs).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.w_key(inputs).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.w_value(inputs).view(batch_size, -1, self.num_heads, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q.transpose(1, 2)
        p = self.pos_projection(pos_embed).view(batch_size, 1, self.num_heads, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        mask = mask.unsqueeze(1).eq(0)
        scores = scores.masked_fill(mask, -float('inf'))
        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        attn = self.dropout(attn)
        outputs = torch.matmul(attn, v)
        outputs = (outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k))

        outputs = self.projection(outputs)
        return outputs




class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, inputs):
        pos_embed = self.pe[:inputs.size(0)].to(inputs.device)
        x = inputs + pos_embed
        return self.dropout(x), self.dropout(pos_embed)


class MultiHeadSelfAttentionModule(nn.Module):

    def __init__(self, encoder_dim, num_heads, dropout):
        super(MultiHeadSelfAttentionModule, self).__init__()
        self.d_k = encoder_dim // num_heads
        self.num_heads = num_heads
        self.w_key = nn.Linear(encoder_dim, encoder_dim)
        self.w_query = nn.Linear(encoder_dim, encoder_dim)
        self.w_value = nn.Linear(encoder_dim, encoder_dim)
        self.projection = nn.Linear(encoder_dim, encoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(encoder_dim, eps=1e-5)

    def forward(self,
                inputs,
                input_lengths,
                pos_embed):
        batch_size = inputs.size(0)
        mask = torch.arange(input_lengths.max().item())[None, :] < input_lengths[:, None]
        mask = mask.unsqueeze(1).repeat(batch_size, 1, 1)
        inputs = self.layer_norm(inputs)
        q = self.w_query(inputs).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.w_key(inputs).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.w_value(inputs).view(batch_size, -1, self.num_heads, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = mask.unsqueeze(1).eq(0)
        scores = scores.masked_fill(mask, -float('inf'))
        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        attn = self.dropout(attn)
        outputs = torch.matmul(attn, v)
        outputs = (outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k))

        outputs = self.projection(outputs)

        return outputs


