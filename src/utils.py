import torch


def collate_fn(batch):
    batched_data = dict()
    wavs = []
    wav_lengths = []
    targets = []
    target_lengths = []
    sentences = []
    for wav, target, sentence in batch:
        wav_length = wav.shape[0]
        target_length = target.shape[0]
        wavs.append(wav)
        targets.append(target)
        sentences.append(sentence)
        wav_lengths.append(wav_length)
        target_lengths.append(target_length)

    wavs = torch.hstack(wavs)
    targets = torch.hstack(targets)
    wav_lengths = torch.LongTensor(wav_lengths)
    target_lengths = torch.LongTensor(target_lengths)
    batched_data['inputs'] = wavs
    batched_data['input_lengths'] = wav_lengths
    batched_data['targets'] = targets
    batched_data['target_lengths'] = target_lengths
    batched_data['sentences'] = sentences
    return batched_data
