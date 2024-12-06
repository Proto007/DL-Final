import numpy as np
import os
import librosa
import numpy as np
from torch.utils import data

class MtatDataset(data.Dataset):
    def __init__(self, split, input_length=None, augmentations=True):
        np.random.seed(0)
        self.split = split
        self.input_length = input_length
        self.augmentations = augmentations
        if split == 'TRAIN':
            self.fl = np.load('./split/train.npy')
        elif split == 'VALID':
            self.fl = np.load('./split/valid.npy')
        elif split == 'TEST':
            self.fl = np.load('./split/test.npy')
        else:
            print('Split should be one of [TRAIN, VALID, TEST]')
        self.binary = np.load('./split/binary.npy')

    def __getitem__(self, index):
        ix, fn = self.fl[index].split('\t')
        npy_path = os.path.join('data', 'npy', fn.split('/')[1][:-3]) + 'npy'
        npy = np.load(npy_path, mmap_mode='r')
        random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
        npy = np.array(npy[random_idx:random_idx+self.input_length])
        tag_binary = self.binary[int(ix)]
        # add data augmentation to datapoints in training
        if self.split == 'TRAIN' and self.augmentations:
            npy = librosa.effects.time_stretch(npy,rate=np.random.uniform(0.8,1.2))
            # TODO: add other augmentations, add augmentations only for some data points
        # ensure the length of npy is consistent with input_length
        if len(npy) > self.input_length:
            npy = npy[:self.input_length]
        elif len(npy) < self.input_length:
            padding = np.zeros((self.input_length - len(npy),), dtype=npy.dtype)
            npy = np.concatenate((npy, padding))
        return npy.astype('float32'), tag_binary.astype('float32')

    def __len__(self):
        return len(self.fl)

