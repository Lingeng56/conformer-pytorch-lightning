import torch


def collate_fn(batch):
    wavs = []
    wav_lengths = []
    targets = []
    target_lengths = []
    sentences = []
    for wav, target, sentence in batch:
        wav_length = wav.shape[-1]
        target_length = target.shape[0]
        targets.append(target.unsqueeze(0))
        sentences.append(sentence)
        wav_lengths.append(wav_length)
        target_lengths.append(target_length)

    # padding wav features
    wav_max_length = max(wav_lengths)
    for wav, _, _ in batch:
        wav = torch.nn.functional.pad(wav, (0, wav_max_length - wav.shape[-1]))
        wavs.append(wav)


    # convert data into tensor
    wavs = torch.concat(wavs, dim=0).permute(0, 2, 1)
    targets = torch.concat(targets, dim=0)
    wav_lengths = torch.LongTensor(wav_lengths)
    target_lengths = torch.LongTensor(target_lengths)



    return {
        'inputs': wavs,
        'input_lengths': wav_lengths,
        'targets': targets,
        'target_lengths': target_lengths,
        'sentences': sentences
    }
