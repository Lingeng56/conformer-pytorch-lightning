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
        self.mfcc_transform = transforms.MFCC(sample_rate=16000,
                                              n_mfcc=40,
                                              melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40, 'center': False}
                                              )

    def __getitem__(self, index):
        wav, _, sentence, _, _, _ = self.dataset[index]
        target = self.tokenizer(sentence)
        mfcc = self.mfcc_transform(wav)
        return mfcc, torch.LongTensor(target), sentence

    def __len__(self):
        return len(self.dataset)
