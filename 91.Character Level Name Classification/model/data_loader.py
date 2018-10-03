import random
import numpy as np
import os
import sys

import torch
from torchtext import data

import utils

class DataLoader(object):

    def __init__(self, data_dir, params):
        #
        # self.BABYNAME = data.Field(sequential=True, pad_first=True,
        #                       tokenize=utils.tokenizer, fix_length=20,
        #                       batch_first=True, init_token="<bos>", eos_token="<eos>")

        device = torch.device(params.device)

        self.BABYNAME = data.Field(sequential=True, pad_first=True,
                                   tokenize=utils.tokenizer,
                                   batch_first=True, init_token="<bos>", eos_token="<eos>")
        self.SEX = data.Field(sequential=False, use_vocab=True)

        self.train_ds, self.val_ds = data.TabularDataset.splits(
            path=data_dir, skip_header=True, train='train/train_dataset.csv',
            validation='val/val_dataset.csv', format='csv',
            fields=[('babyname', self.BABYNAME), ('sex', self.SEX)]
        )

        self.build_vocab()

        self.train_iter, self.val_iter = data.BucketIterator.splits(
            (self.train_ds, self.val_ds), batch_sizes=(params.batch_size, params.batch_size), device=device,
            repeat=False, sort_key=lambda x: len(x.babyname))


    def build_vocab(self):
        self.BABYNAME.build_vocab(self.train_ds, self.val_ds)
        self.SEX.build_vocab(self.train_ds, self.val_ds)
        print("vocab built")
#
# model_dir = '../experiments/model/model_selfattention'
# json_path = os.path.join(model_dir, 'params.json')
# params = utils.Params(json_path)
# data_dir = '../data/full_version'
#
# data_loader = DataLoader(data_dir, params)
#
# print(data_loader.BABYNAME.vocab.freqs)
# print(data_loader.SEX.vocab.freqs)
#
# print(list(data_loader.train_ds.attributes))
#
# sample = next(iter(data_loader.train_iter))
# print(sample.babyname[0,:])
# print(data_loader.BABYNAME.vocab.stoi)
#
# print(data_loader.val_ds)