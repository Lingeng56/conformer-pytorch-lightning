import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch
import random
import re
import math
from torch.nn.utils.rnn import pad_sequence

torchaudio.utils.sox_utils.set_buffer_size(16500)


def parse_raw(data):
    for sample in data:
        key = sample['key']
        wav_file = sample['wav_path']
        transcript = sample['transcript']
        waveform, sample_rate = torchaudio.load(wav_file)
        sample = dict(
            key=key,
            transcript=transcript,
            waveform=waveform,
            sample_rate=sample_rate
        )

        yield sample


def filter_data(data,
                max_length,
                min_length,
                token_max_length,
                token_min_length,
                min_output_input_ratio,
                max_output_input_ratio):
    for sample in data:
        num_frames = sample['waveform'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length or num_frames > max_length:
            continue

        if len(sample['label']) < token_min_length or len(sample['label']) > token_max_length:
            continue

        if len(sample['label']) / num_frames < min_output_input_ratio or len(
                sample['label']) / num_frames > max_output_input_ratio:
            continue
        yield sample


def resample(data, resample_rate=16000):
    for sample in data:
        sample_rate = sample['sample_rate']
        waveform = sample['waveform']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['waveform'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate
            )(waveform)

        yield sample


def speed_perturb(data, speeds=None):
    for sample in data:
        if speeds is None:
            speeds = [0.9, 1.0, 1.1]

        sample_rate = sample['sample_rate']
        waveform = sample['waveform']
        speed = random.choice(speeds)
        if speed != 1.0:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]]
            )
        sample['waveform'] = waveform

        yield sample


def __tokenize_by_bpe_model(sp, transcript):
    tokens = []
    # 这个包含了中文和空格所有的字符，我觉得比较重要
    pattern = re.compile(r'([\u4e00-\u9fff])')
    chars = pattern.split(transcript.upper())
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for char_or_w in mix_chars:
        if pattern.fullmatch(char_or_w) is not None:
            tokens.append(char_or_w)
        else:
            for p in sp.encode_as_pieces(char_or_w):
                tokens.append(p)

    return tokens


def tokenize(data,
             vocabs,
             bpe_model=None,
             non_lang_syms=None,
             split_with_space=False):
    for sample in data:
        if non_lang_syms is not None:
            non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        else:
            non_lang_syms = {}
            non_lang_syms_pattern = None

        if bpe_model is not None:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(bpe_model)
        else:
            sp = None

        transcript = sample['transcript']
        if non_lang_syms_pattern is not None:
            parts = non_lang_syms_pattern.split(transcript.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [transcript]

        label = []
        tokens = []

        for part in parts:
            if part in non_lang_syms:
                tokens.append(part)
            else:
                if bpe_model is not None:
                    tokens.extend(__tokenize_by_bpe_model(sp, part))
                else:
                    if split_with_space:
                        part = part.split(" ")
                    for ch in part:
                        if ch == ' ':
                            ch = '_'
                        tokens.append(ch)

        for ch in tokens:
            if ch in vocabs:
                label.append(vocabs[ch])
            elif '<unk>' in vocabs:
                label.append(vocabs['<unk>'])

        sample['tokens'] = tokens
        sample['label'] = label

        yield sample


def spec_aug(data,
             num_t_mask,
             num_f_mask,
             max_t,
             max_f):
    for sample in data:
        x = sample['feat']
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        for _ in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        for _ in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def compute_fbank(data,
                  num_mel_bins,
                  frame_length,
                  frame_shift,
                  dither):
    for sample in data:
        waveform = sample['waveform']
        sample_rate = sample['sample_rate']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        feat = kaldi.fbank(waveform,
                           num_mel_bins=num_mel_bins,
                           frame_length=frame_length,
                           frame_shift=frame_shift,
                           dither=dither,
                           energy_floor=0.0,
                           sample_frequency=sample_rate)
        sample = dict(key=sample['key'], label=sample['label'], feat=feat, transcript=sample['transcript'], tokens=sample['tokens'])
        yield sample


def compute_mfcc(data,
                 num_mel_bins,
                 frame_length,
                 frame_shift,
                 dither,
                 num_ceps,
                 high_freq,
                 low_freq):
    for sample in data:
        waveform = sample['waveform']
        sample_rate = sample['sample_rate']
        waveform *= (1 << 15)
        feat = kaldi.mfcc(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          num_ceps=num_ceps,
                          high_freq=high_freq,
                          low_freq=low_freq,
                          sample_frequency=sample_rate)
        sample = dict(key=sample['key'], label=sample['label'], feat=feat, tokens=sample['tokens'], transcript=sample['transcript'])
        yield sample


def collate_fn(batch_):
    keys, inputs, input_lengths, targets, target_lengths, sentences = batch_[0]
    return keys, inputs, input_lengths, targets, target_lengths, sentences


def shuffle(data, shuffle_size=10000):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []

    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf = sorted(buf, key=lambda item: item['feat'].size(0))
            for x in buf:
                yield x
            buf = []

    buf = sorted(buf, key=lambda item: item['feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size):
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch):
    buf = []
    longest_frames = 0
    for sample in data:
        longest_frames = max(longest_frames, sample['feat'].size(0))
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = sample['feat'].size(0)
        else:
            buf.append(sample)

    if len(buf) > 0:
        yield buf


def batch(data, batch_type, **kwargs):
    assert batch_type in ['static', 'dynamic']
    if batch_type == 'static':
        return static_batch(data, kwargs['batch_size'])
    else:
        return dynamic_batch(data, kwargs['max_frames_in_batch'])


def padding(data):
    for sample in data:
        feats_length = torch.tensor([x['feat'].size(0) for x in sample], dtype=torch.int32)
        order = torch.argsort(feats_length, descending=True)
        feats_length = torch.tensor([sample[i]['feat'].size(0) for i in order], dtype=torch.int32)
        sorted_feats = [sample[i]['feat'] for i in order]
        sorted_keys = [sample[i]['key'] for i in order]
        sorted_labels = [torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order]
        label_lengths = torch.tensor([x.size(0) for x in sorted_labels], dtype=torch.int32)
        transcripts = [sample[i]['transcript'] for i in order]
        padded_feats = pad_sequence(sorted_feats,
                                    batch_first=True,
                                    padding_value=0)
        padded_labels = pad_sequence(sorted_labels,
                                     batch_first=True,
                                     padding_value=0
                                     )
        yield (
            sorted_keys,
            padded_feats,
            feats_length,
            padded_labels,
            label_lengths,
            transcripts
        )
