import json
import torch.distributed as dist
import os
import pickle
from torch.utils.data import Dataset, IterableDataset
from processor import *
from tqdm import tqdm


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class Processor(IterableDataset):

    def __init__(self, data_source, operation, *args, **kwargs):
        self.data_source = data_source
        self.operation = operation
        self.args = args
        self.kwargs = kwargs

    def set_epoch(self, epoch):
        self.data_source.set_epoch(epoch)

    def __iter__(self):
        return self.operation(iter(self.data_source), *self.args, **self.kwargs)


class DataList(IterableDataset):
    def __init__(self, data_list_path, shuffle=True, partition=True, extend=True):
        self.sampler = DistributedSampler(shuffle, partition)
        self.data_list = []
        with open(data_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.data_list.append(json.loads(line))

        if extend:
            for _ in range(10):
                self.data_list += self.data_list


    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.data_list)
        for index in indexes:
            # yield dict(src=src)
            data = self.data_list[index]
            data.update(sampler_info)
            yield data




class NormalCustomDataset(Dataset):

    def __init__(self, data_config, mode):
        self.data_config = data_config
        self.mode = mode
        self.__prepare_data()

    def __prepare_data(self):
        self.dataset = DataList(self.data_config[f'{self.mode}_data_list_path'], extend=False)
        self.dataset = Processor(self.dataset, parse_raw)
        self.dataset = Processor(self.dataset,
                                 tokenize,
                                 vocabs=self.data_config['vocabs'],
                                 bpe_model=self.data_config['bpe_model'],
                                 non_lang_syms=self.data_config['non_lang_syms'],
                                 split_with_space=self.data_config['split_with_space']
                                 )
        # self.dataset = Processor(self.dataset,
        #                          filter_data,
        #                          max_length=self.data_config['max_length'],
        #                          min_length=self.data_config['min_length'],
        #                          token_max_length=self.data_config['token_max_length'],
        #                          token_min_length=self.data_config['token_min_length'],
        #                          min_output_input_ratio=self.data_config['min_output_input_ratio'],
        #                          max_output_input_ratio=self.data_config['max_output_input_ratio'])
        self.dataset = Processor(self.dataset,
                                 resample,
                                 resample_rate=self.data_config['resample_rate']
                                 )
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
                                     dither=self.data_config['dither']
                                     )
        elif self.data_config['feat_type'] == 'mfcc':
            self.dataset = Processor(self.dataset,
                                     compute_mfcc,
                                     num_mel_bins=self.data_config['num_mel_bins'],
                                     frame_length=self.data_config['frame_length'],
                                     dither=self.data_config['dither'],
                                     num_ceps=self.data_config['num_ceps'],
                                     high_freq=self.data_config['high_freq'],
                                     low_freq=self.data_config['low_freq'],
                                     frame_shift=self.data_config['frame_shift']
                                     )

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
                                     shuffle_size=self.data_config['shuffle_size'])

        if self.data_config['sort']:
            self.dataset = Processor(self.dataset,
                                     sort,
                                     sort_size=self.data_config['sort_size'])

        self.dataset = Processor(self.dataset,
                                 batch,
                                 batch_type=self.data_config['batch_type'],
                                 batch_size=self.data_config['batch_size'],
                                 max_frames_in_batch=self.data_config['max_frames_in_batch'])

        self.dataset = [item for item in Processor(self.dataset,
                                                   padding
                                                   )]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class IterableCustomDataset(IterableDataset):

    def __init__(self, data_config, mode):
        self.data_config = data_config
        self.mode = mode
        self.__prepare_data()

    def __prepare_data(self):
        self.dataset = DataList(self.data_config[f'{self.mode}_data_list_path'])
        self.dataset = Processor(self.dataset, parse_raw)
        self.dataset = Processor(self.dataset,
                                 tokenize,
                                 vocabs=self.data_config['vocabs'],
                                 bpe_model=self.data_config['bpe_model'],
                                 non_lang_syms=self.data_config['non_lang_syms'],
                                 split_with_space=self.data_config['split_with_space']
                                 )
        # self.dataset = Processor(self.dataset,
        #                          filter_data,
        #                          max_length=self.data_config['max_length'],
        #                          min_length=self.data_config['min_length'],
        #                          token_max_length=self.data_config['token_max_length'],
        #                          token_min_length=self.data_config['token_min_length'],
        #                          min_output_input_ratio=self.data_config['min_output_input_ratio'],
        #                          max_output_input_ratio=self.data_config['max_output_input_ratio'])
        self.dataset = Processor(self.dataset,
                                 resample,
                                 resample_rate=self.data_config['resample_rate']
                                 )
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
                                     dither=self.data_config['dither']
                                     )
        elif self.data_config['feat_type'] == 'mfcc':
            self.dataset = Processor(self.dataset,
                                     compute_mfcc,
                                     num_mel_bins=self.data_config['num_mel_bins'],
                                     frame_length=self.data_config['frame_length'],
                                     dither=self.data_config['dither'],
                                     num_ceps=self.data_config['num_ceps'],
                                     high_freq=self.data_config['high_freq'],
                                     low_freq=self.data_config['low_freq'],
                                     frame_shift=self.data_config['frame_shift']
                                     )

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
                                     shuffle_size=self.data_config['shuffle_size'])

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
                                 padding
                                 )

    def __iter__(self):
        for item in self.dataset:
            yield item


