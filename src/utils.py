import torch


def load_vocabs(vocab_path):
    vocabs = dict()
    with open(vocab_path) as f:
        for line in f:
            line = line.strip()
            word, idx = line.split(' ')
            vocabs[word] = int(idx)

    return vocabs, len(vocabs)


def build_joint_text(targets, blank):
    batch_size = targets.size(0)
    _blank = torch.tensor([blank],
                          dtype=torch.long,
                          requires_grad=False,
                          device=targets.device)
    _blank = _blank.repeat(batch_size).unsqueeze(1)
    outputs = torch.cat([_blank, targets], dim=1)
    return outputs
