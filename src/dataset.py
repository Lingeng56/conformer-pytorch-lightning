import torch
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, data_dir, url, tokenizer):
        super(CustomDataset, self).__init__()
        self.dataset = LIBRISPEECH(root=data_dir,
                                   url=url,
                                   download=True)
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        wav, _, sentence, _, _, _ = self.dataset[index]
        target = self.tokenizer(sentence)
        return wav, torch.LongTensor(target), sentence

    def __len__(self):
        return len(self.dataset)
