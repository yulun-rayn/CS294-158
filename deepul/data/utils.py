import random
from collections import OrderedDict

import numpy as np
from numpy.lib.stride_tricks import as_strided

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def batch_tensor_apply(tensor, fn, batch_size=1024):
    loader = DataLoader(tensor, batch_size=batch_size)
    out = []
    for batch in loader:
        out.append(fn(batch))
    return torch.cat(out, dim=0)

def take_per_row_strided(A, indx, num_elem=2):
    m,n = A.shape
    A.shape = (-1)
    s0 = A.strides[0]
    l_indx = indx + n*np.arange(len(indx))
    out = as_strided(A, (len(A)-num_elem+1, num_elem), (s0,s0))[l_indx]
    A.shape = m,n
    return out


class IntDataset:
    def __init__(self, data):
        self.data = torch.LongTensor(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class LabeledDataset:
    def __init__(self, data, labels, dropout=0.):
        self.data = torch.FloatTensor(data)
        self.labels = labels
        self.dropout = dropout

    def __getitem__(self, index):
        return (self.data[index], self.labels[index]
            if self.dropout <= 0 or random.random() > self.dropout else -1)

    def __len__(self):
        return len(self.labels)


class SeqDataset:
    def __init__(self, data, max_len=128, pad_value=-1):
        self.max_len = max_len
        self.pad_value = pad_value

        self.data = [torch.LongTensor(d) for d in data]

    def get_collate(self):
        return lambda x: pad_sequence(x, batch_first=True, padding_value=self.pad_value)

    def __getitem__(self, index):
        return self.data[index][:self.max_len]

    def __len__(self):
        return len(self.data)


class MultimodalDataset:
    def __init__(self, img_data, txt_data, max_len=128, pad_value=-1):
        assert len(img_data) == len(txt_data)
        self.img_set = IntDataset(img_data)
        self.txt_set = SeqDataset(txt_data, max_len, pad_value)

    def get_collate(self):
        def fn(batch):
            img = torch.stack([x for (x, _) in batch])
            txt = pad_sequence([x for (_, x) in batch], batch_first=True, padding_value=self.txt_set.pad_value)
            if random.randrange(2):
                return OrderedDict([("img", img), ("txt", txt)])
            else:
                return OrderedDict([("txt", txt), ("img", img)])

        return fn

    def __getitem__(self, index):
        return (self.img_set[index], self.txt_set[index])

    def __len__(self):
        return len(self.img_set)


class SeqDatasetRandom(SeqDataset):
    def __init__(self, data, max_len=128, pad_value=-1):
        super().__init__(data, max_len, pad_value)

    def __getitem__(self, index):
        data = self.data[index]
        start_pos = random.randrange(max(len(data)-self.max_len, 0)+1)
        return data[start_pos:min(start_pos+self.max_len, len(data))]


class SeqDatasetWeight(SeqDataset):
    def __init__(self, data, max_len=128, pad_value=-1, temperature=0.5):
        super().__init__(data, max_len, pad_value)
        self.weight = []
        for i, d in enumerate(data):
            self.weight.extend([i]*int((max(len(d)-self.max_len, 0)+1)**temperature))

    def __getitem__(self, index):
        data = self.data[self.weight[index]]
        start_pos = random.randrange(max(len(data)-self.max_len, 0)+1)
        return data[start_pos:min(start_pos+self.max_len, len(data))]

    def __len__(self):
        return len(self.weight)


class SeqDatasetLong(SeqDataset):
    def __init__(self, data, max_len=128, pad_value=-1):
        long_data = []
        for d in data:
            mat = np.repeat(np.array(d)[None,:], max(len(d)-max_len, 0)+1, axis=0)
            take = take_per_row_strided(mat, range(max(len(d)-max_len, 0)+1), num_elem=min(len(d), max_len))
            long_data.extend([*take])
        super().__init__(long_data, max_len, pad_value)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
