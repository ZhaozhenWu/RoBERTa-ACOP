# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy


class BucketIterator(object):
    def __init__(self, data, batch_size, other=None, sort_key='text_indices', shuffle=True, sort=False):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batch_size = batch_size
        # choose sentence which size < 100
        self.data = [item for item in data if len(item['text_indices']) < 100]
        self.other = None
        self.batches = self.sort_and_pad(self.data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batches

    def pad_data(self, batch_data, other_data=None):
        batch_text = []
        batch_text_indices = []
        batch_text_indices_trans = []
        batch_trans = []
        batch_aspect_indices = []
        batch_polarity = []
        batch_order = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        max_len_word = max([len(t['trans']) for t in batch_data])
        max_len_trans = max([len(t['text_indices_trans']) for t in batch_data])
        max_len_aspect = max([len(t['aspect_indices']) for t in batch_data])
        for item in batch_data:
            text, text_indices, text_indices_trans, trans, polarity, order, aspect_indices = \
                item['text'], item['text_indices'], item['text_indices_trans'], item['trans'], \
                item['polarity'], item['order'], item['aspect_indices']
            text_padding = [0] * (max_len - len(text_indices))
            text_trans_padding = [0] * (max_len_trans - len(text_indices_trans))
            aspect_padding = [0] * (max_len_aspect - len(aspect_indices))
            batch_text.append(text)
            batch_text_indices.append(text_indices + text_padding)
            batch_text_indices_trans.append(text_indices_trans + text_trans_padding)
            batch_trans.append(trans)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_polarity.append(polarity)
            batch_order.append(order)

        return {
            'text': batch_text,
            'text_indices': torch.tensor(batch_text_indices),
            'text_indices_trans': torch.tensor(batch_text_indices_trans),
            'trans': batch_trans,
            'aspect_indices': torch.tensor(batch_aspect_indices),
            'polarity': torch.tensor(batch_polarity),
            'order': torch.tensor(batch_order)
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
            self.batches = self.sort_and_pad(self.data, self.batch_size)
        for idx in range(self.batch_len):
            yield self.batches[idx]


class BucketIteratorOrder(object):
    def __init__(self, data, batch_size, other=None, sort_key='text_indices', shuffle=True, sort=False):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batch_size = batch_size
        # choose sentence which size < 100
        self.data = [item for item in data if len(item['text_indices']) < 100]
        self.other = None
        self.batches = self.sort_and_pad(self.data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batches

    def pad_data(self, batch_data, other_data=None):
        batch_text_indices = []
        batch_order = []

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, order = item['text_indices'], item['order']
            text_padding = [0] * (max_len - len(text_indices))
            batch_text_indices.append(text_indices + text_padding)
            batch_order.append(order)

        return {
            'text_indices': torch.tensor(batch_text_indices),
            'order': torch.tensor(batch_order)
        }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
            self.batches = self.sort_and_pad(self.data, self.batch_size)
        for idx in range(self.batch_len):
            yield self.batches[idx]
