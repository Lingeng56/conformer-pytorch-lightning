import json
import os
import pickle
from torch.utils.data import Dataset
from processor import *
from tqdm import tqdm


class Processor(Dataset):

    def __init__(self, data_source, operation, *args, **kwargs):
        self.dataset = operation(data_source, *args, **kwargs)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class DataList(Dataset):

    def __init__(self, data_list_path):
        super(DataList, self).__init__()
        self.data_list = []
        with open(data_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.data_list.append(json.loads(line))

    def __getitem__(self, idx):
        return self.data_list[idx]


    def __len__(self):
        return len(self.data_list)


class CustomDataset(Dataset):

    def __init__(self, data_config, mode):
        self.data_config = data_config
        self.mode = mode
        self.fbank_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,  # Window size of 25ms (assuming sample_rate=16000)
            hop_length=160,  # Stride of 10ms (assuming sample_rate=16000)
            n_mels=80
        )
        self.__prepare_data()
        if data_config['batch']:
            self.__batched_data()


    def __batched_data(self):
        if os.path.exists(self.data_config[f'{self.mode}_batched_pkl_path']):
            self.dataset = pickle.load(open(self.data_config[f'{self.mode}_pkl_path'], 'rb'))
        else:
            if self.data_config['shuffle']:
                self.dataset = shuffle(self.dataset,
                                       self.data_config['shuffle_size'])

            if self.data_config['sort']:
                self.dataset = sort(self.dataset,
                                    self.data_config['sort_size'])

            self.dataset = batch(self.dataset,
                                 self.data_config['batch_type'],
                                 batch_size=self.data_config['batch_size'],
                                 max_frames_in_batch=self.data_config['max_frames_in_batch'])

            self.dataset = padding(self.dataset)
            pickle.dump(self.dataset, open(self.data_config[f'{self.mode}_batched_pkl_path'], 'wb'))
        print(f'Loaded Batched {self.mode} Dataset ({len(self.dataset)} Batches)')


    def __prepare_data(self):
        if os.path.exists(self.data_config[f'{self.mode}_pkl_path']):
            self.dataset = pickle.load(open(self.data_config[f'{self.mode}_pkl_path'], 'rb'))
        else:
            dataset = DataList(self.data_config[f'{self.mode}_data_list_path'])
            self.dataset = []
            for sample in tqdm(dataset, desc='Preprocessing Data...'):
                sample = parse_raw(sample)
                sample = tokenize(sample,
                                  vocabs=self.data_config['vocabs'],
                                  bpe_model=self.data_config['bpe_model'],
                                  non_lang_syms=self.data_config['non_lang_syms'],
                                  split_with_space=self.data_config['split_with_space']
                                  )
                check = filter_data(sample,
                                    max_length=self.data_config['max_length'],
                                    min_length=self.data_config['min_length'],
                                    token_max_length=self.data_config['token_max_length'],
                                    token_min_length=self.data_config['token_min_length'],
                                    min_output_input_ratio=self.data_config['min_output_input_ratio'],
                                    max_output_input_ratio=self.data_config['max_output_input_ratio'])

                if not check:
                    continue

                sample = resample(sample,
                                  resample_rate=self.data_config['resample_rate'])

                if self.data_config['speed_perturb']:
                    sample = speed_perturb(sample,
                                           speeds=self.data_config['speeds'])

                if self.data_config['feat_type'] == 'fbank':
                    sample = compute_fbank(sample, self.fbank_transform)

                elif self.data_config['feat_type'] == 'mfcc':
                    sample = compute_mfcc(sample,
                                          num_mel_bins=self.data_config['num_mel_bins'],
                                          frame_length=self.data_config['frame_length'],
                                          dither=self.data_config['dither'],
                                          num_ceps=self.data_config['num_ceps'],
                                          high_freq=self.data_config['high_freq'],
                                          low_freq=self.data_config['low_freq'])

                if self.data_config['spec_aug']:
                    sample = spec_aug(sample,
                                      num_t_mask=self.data_config['num_t_mask'],
                                      num_f_mask=self.data_config['num_f_mask'],
                                      max_t=self.data_config['max_t'],
                                      max_f=self.data_config['max_f'],
                                      )
                self.dataset.append(sample)
            pickle.dump(self.dataset, open(self.data_config[f'{self.mode}_pkl_path'], 'wb'))
        print(f'Loaded {self.mode} Dataset ({len(self.dataset)} Samples)')

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
