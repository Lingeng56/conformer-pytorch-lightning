import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
  '''
    Generate positional encodings used in the relative multi-head attention module.
    These encodings are the same as the original transformer model: https://arxiv.org/abs/1706.03762

    Parameters:
      max_len (int): Maximum sequence length (time dimension)

    Inputs:
      len (int): Length of encodings to retrieve
    
    Outputs
      Tensor (len, d_model): Positional encodings
  '''
  def __init__(self, d_model, dropout, max_len):
    super(RelativePositionalEncoding, self).__init__()
    self.d_model = d_model
    encodings = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float)
    inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
    encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
    encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('encodings', encodings)
    
  def forward(self, len):
      return self.dropout(self.encodings[:len, :])

class RelativeMultiHeadSelfAttentionModule(nn.Module):
  '''
    Relative Multi-Head Self-Attention Module. 
    Method proposed in Transformer-XL paper: https://arxiv.org/abs/1901.02860

    Parameters:
      d_model (int): Dimension of the model
      num_heads (int): Number of heads to split inputs into
      dropout (float): Dropout probability
      positional_encoder (nn.Module): PositionalEncoder module
    
    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices
    
    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the attention module.
  
  '''
  def __init__(self, d_model, num_heads, dropout, max_len):
    super(RelativeMultiHeadSelfAttentionModule, self).__init__()


    self.d_model = d_model
    self.d_head = d_model // num_heads
    self.num_heads = num_heads

    # Linear projection weights
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    self.W_pos = nn.Linear(d_model, d_model, bias=False)
    self.W_out = nn.Linear(d_model, d_model)

    # Trainable bias parameters
    self.u = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
    self.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
    torch.nn.init.xavier_uniform_(self.u)
    torch.nn.init.xavier_uniform_(self.v)

    # etc
    self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
    self.positional_encoding = RelativePositionalEncoding(d_model,
                                                          dropout,
                                                          max_len)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    batch_size, seq_length, _ = x.size()

    #layer norm and pos embeddings
    x = self.layer_norm(x)
    pos_emb = self.positional_encoding(seq_length)
    pos_emb = pos_emb.repeat(batch_size, 1, 1)

    #Linear projections, split into heads
    q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_head)
    k = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
    v = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
    pos_emb = self.W_pos(pos_emb).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)

    #Compute attention scores with relative position embeddings
    AC = torch.matmul((q + self.u).transpose(1, 2), k)
    BD = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
    BD = self.rel_shift(BD)
    attn = (AC + BD) / math.sqrt(self.d_model)

    #Mask before softmax with large negative number
    if mask is not None:
      mask = mask.unsqueeze(1)
      mask_value = -1e+30 if attn.dtype == torch.float32 else -1e+4
      attn.masked_fill_(mask, mask_value)

    #Softmax
    attn = F.softmax(attn, -1)

    #Construct outputs from values
    output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2) # (batch_size, time, num_heads, d_head)
    output = output.contiguous().view(batch_size, -1, self.d_model) # (batch_size, time, d_model)

    #Output projections and dropout
    output = self.W_out(output)
    return self.dropout(output)


  def rel_shift(self, emb):
    '''
      Pad and shift form relative positional encodings. 
      Taken from Transformer-XL implementation: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py 
    '''
    batch_size, num_heads, seq_length1, seq_length2 = emb.size()
    zeros = emb.new_zeros(batch_size, num_heads, seq_length1, 1)
    padded_emb = torch.cat([zeros, emb], dim=-1)
    padded_emb = padded_emb.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
    shifted_emb = padded_emb[:, :, 1:].view_as(emb)
    return shifted_emb


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
        x = inputs + self.pe[:inputs.size(0)]
        return self.dropout(x)


class MultiHeadSelfAttentionModule(nn.Module):

    def __init__(self, encoder_dim, num_heads, dropout, max_len):
        super(MultiHeadSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(encoder_dim,
                                                      dropout,
                                                      max_len)
        self.attention = nn.MultiheadAttention(
            encoder_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.w_key = nn.Linear(encoder_dim, encoder_dim)
        self.w_query = nn.Linear(encoder_dim, encoder_dim)
        self.w_value = nn.Linear(encoder_dim, encoder_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        keys = self.w_key(inputs)
        queries = self.w_query(inputs)
        values = self.w_value(inputs)
        attn_output, _ = self.attention(keys, queries, values)
        attn_output = self.dropout(attn_output)
        attn_output = inputs + attn_output
        return attn_output

