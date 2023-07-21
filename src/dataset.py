import torch
import json
import torchaudio
import pickle
import os
from torch.utils.data import Dataset
from torchaudio import transforms
from tqdm import tqdm


class CustomDataset(Dataset):

    def __init__(self, data_list_path, tokenizer):
        super(CustomDataset, self).__init__()
        self.data_list_path = data_list_path
        self.tokenizer = tokenizer

        # Apply Mel filterbank transformation
        self.fbank_transform = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,  # Window size of 25ms (assuming sample_rate=16000)
            hop_length=160,  # Stride of 10ms (assuming sample_rate=16000)
            n_mels=80
        )


        self._prepare_data()

    def _prepare_data(self):
        pkl_path = os.path.join(os.path.dirname(self.data_list_path), 'dataset.pkl')
        if os.path.exists(pkl_path):
            self.data_list = pickle.load(open(pkl_path, 'rb'))
        else:
            self.data_list = []
            with open(self.data_list_path) as f:
                print('Preparing Data')
                for line in tqdm(f):
                    line = line.strip()
                    data_dict = json.loads(line)
                    waveform, sampling_rate = torchaudio.load(data_dict['wav_path'])
                    fbank = self.fbank_transform(waveform)
                    self.data_list.append((data_dict['key'], fbank, data_dict['transcript'], sampling_rate))


            pickle.dump(self.data_list, open(pkl_path, 'wb'))

    def __getitem__(self, index):
        key, fbank, transcript, sampling_rate = self.data_list[index]
        target = self.tokenizer(transcript)
        return fbank, torch.LongTensor(target), transcript

    def __len__(self):
        return len(self.data_list)
