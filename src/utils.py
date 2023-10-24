import torch
import torch.nn as nn
import json
import math


def load_cmvn(json_cmvn_file):
    """ Load the json format cmvn stats file and calculate cmvn

    Args:
        json_cmvn_file: cmvn stats file in json format

    Returns:
        a numpy array of [means, vars]
    """
    with open(json_cmvn_file) as f:
        cmvn_stats = json.load(f)

    means = cmvn_stats['mean_stat']
    variance = cmvn_stats['var_stat']
    count = cmvn_stats['frame_num']
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    return torch.tensor(means), torch.tensor(variance)


def pad_list(xs,
             pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max([x.size(0) for x in xs])
    pad = torch.zeros(n_batch, max_len, dtype=xs[0].dtype, device=xs[0].device)
    pad = pad.fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def load_vocabs(vocab_path):
    vocabs = dict()
    with open(vocab_path) as f:
        for line in f:
            line = line.strip()
            word, idx = line.split(' ')
            vocabs[word] = int(idx)

    return vocabs, len(vocabs)


def add_blank(targets, blank, ignore_id):
    bs = targets.size(0)
    _blank = torch.tensor([blank],
                          dtype=torch.long,
                          requires_grad=False,
                          device=targets.device)
    _blank = _blank.repeat(bs).unsqueeze(1)  # [bs,1]
    out = torch.cat([_blank, targets], dim=1)  # [bs, Lmax+1]
    return torch.where(out == ignore_id, blank, out)


def make_pad_mask(input_lengths, max_seq_len):
    batch_size = input_lengths.size(0)
    seq_range = torch.arange(0,
                             max_seq_len,
                             dtype=torch.int64,
                             device=input_lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_seq_len)
    seq_length_expand = input_lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def subsequent_chunk_mask(
        size,
        chunk_size,
        num_left_chunks,
        device
):
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True

    return ret



def make_attn_mask(inputs,
                   inputs_pad_mask,
                   use_dynamic_chunk,
                   use_dynamic_left_chunk,
                   decoding_chunk_size,
                   static_chunk_size,
                   num_decoding_left_chunks):
    max_len = inputs.size(1)
    if use_dynamic_chunk:
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1)
                    num_left_chunks = torch.randint(0, max_left_chunks, (1, )).item()
        chunk_masks = subsequent_chunk_mask(max_len,
                                            chunk_size,
                                            num_left_chunks,
                                            inputs.device)

        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = inputs_pad_mask & chunk_masks

    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(max_len,
                                            static_chunk_size,
                                            num_left_chunks,
                                            inputs.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = inputs_pad_mask & chunk_masks

    else:
        chunk_masks = inputs_pad_mask

    return chunk_masks


def make_subsequent_mask(length, device):
    arange = torch.arange(length, device=device)
    mask = arange.expand(length, length)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def add_sos_eos(targets, sos, eos, ignore_id):
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=targets.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=targets.device)
    ys = [y[y != ignore_id] for y in targets]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def reverse_sequence(targets, target_lengths, ignore_id):
    r_ys_pad = nn.utils.rnn.pad_sequence([(torch.flip(y.int()[:i], [0]))
                                          for y, i in zip(targets, target_lengths)], True,
                                         ignore_id)
    return r_ys_pad
