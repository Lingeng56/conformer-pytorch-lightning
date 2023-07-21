import torch
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset
from torchaudio import transforms


class CustomDataset(Dataset):

    def __init__(self, data_dir, url, tokenizer):
        super(CustomDataset, self).__init__()
        self.dataset = LIBRISPEECH(root=data_dir,
                                   url=url,
                                   download=True)
        self.tokenizer = tokenizer

        # Apply Mel filterbank transformation
        self.fbank_transform = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,  # Window size of 25ms (assuming sample_rate=16000)
            hop_length=160,  # Stride of 10ms (assuming sample_rate=16000)
            n_mels=80
        )

    def __getitem__(self, index):
        wav, _, sentence, _, _, _ = self.dataset[index]
        target = self.tokenizer(sentence)
        fbank = self.fbank_transform(wav)
        fbank = transforms.AmplitudeToDB()(fbank)
        mean_value = torch.mean(fbank)
        std_value = torch.std(fbank)
        fbank = (fbank - mean_value) / std_value
        return fbank, torch.LongTensor(target), sentence

    def __len__(self):
        return len(self.dataset)
