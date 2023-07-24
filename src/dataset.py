import json
from torch.utils.data import IterableDataset
from processor import *


class Processor(IterableDataset):

    def __init__(self, data_source, operation, *args, **kwargs):
        self.operation = operation
        self.data_source = data_source
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.operation(iter(self.data_source), *self.args, **self.kwargs)

    def apply(self, f):
        return Processor(self, f, *self.args, **self.kwargs)


class DataList(IterableDataset):

    def __init__(self, data_list_path):
        super(DataList, self).__init__()
        self.data_list = []
        with open(data_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.data_list.append(json.loads(line))


    def __iter__(self):
        for data_dict in self.data_list:
            return data_dict


class CustomDataset(IterableDataset):

    def __init__(self, data_config):
        self.data_config = data_config


    def __prepare_data(self):
        self.dataset = DataList(self.data_config['data_list_path'])
        self.dataset = Processor(self.dataset,
                                 parse_raw)
        self.dataset = Processor(self.dataset,
                                 resample,
                                 resample_rate=self.data_config['resample_rate'])
        self.dataset = Processor(self.dataset,
                                 tokenize,
                                 vocabs=self.data_config['vocabs'],
                                 bpe_model=self.data_config['bpe_model'],
                                 non_lang_syms=self.data_config['non_lang_syms'],
                                 split_with_space=self.data_config['split_with_space']
                                 )
        self.dataset = Processor(self.dataset,
                                 filter_data,
                                 max_length=self.data_config['max_length'],
                                 min_length=self.data_config['min_length'],
                                 token_max_length=self.data_config['token_max_length'],
                                 token_min_length=self.data_config['token_min_length'],
                                 min_output_input_ratio=self.data_config['min_output_input_ratio'],
                                 max_output_input_ratio=self.data_config['max_output_input_ratio'])

        if self.data_config['speed_perturb']:
            self.dataset = Processor(self.dataset,
                                     speed_perturb,
                                     speeds=self.data_config['speeds'])

        if self.data_config['feat_type'] == 'fbank':
            self.dataset = Processor(self.dataset,
                                     compute_fbank,
                                     num_mel_bins=self.data_config['num_mel_bins'],
                                     frame_length=self.data_config['frame_length'],
                                     frame_shift=self.data_config['frame_shift'],
                                     dither=self.data_config['dither'])
        elif self.data_config['feat_type'] == 'mfcc':
            self.dataset = Processor(self.dataset,
                                     compute_mfcc,
                                     num_mel_bins=self.data_config['num_mel_bins'],
                                     frame_length=self.data_config['frame_length'],
                                     dither=self.data_config['dither'],
                                     num_ceps=self.data_config['num_ceps'],
                                     high_freq=self.data_config['high_freq'],
                                     low_freq=self.data_config['low_freq'])

        if self.data_config['spec_aug']:
            self.dataset = Processor(self.dataset,
                                     spec_aug,
                                     num_t_mask=self.data_config['num_t_mask'],
                                     num_f_mask=self.data_config['num_f_mask'],
                                     max_t=self.data_config['max_t'],
                                     max_f=self.data_config['max_f'],
                                     )

        if self.data_config['shuffle']:
            self.dataset = Processor(self.dataset,
                                     shuffle,
                                     shuffle_size=self.data_config['shuffle_size']
                                     )


        if self.data_config['sort']:
            self.dataset = Processor(self.dataset,
                                     sort,
                                     sort_size=self.data_config['sort_size'])


        self.dataset = Processor(self.dataset,
                                 batch,
                                 batch_type=self.data_config['batch_type'],
                                 batch_size=self.data_config['batch_size'],
                                 max_frames_in_batch=self.data_config['max_frames_in_batch'])

        self.dataset = Processor(self.dataset,
                                 padding)


    def __iter__(self):
        return iter(self.dataset)



