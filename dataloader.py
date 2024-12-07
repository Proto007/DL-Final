import numpy as np
import os
import librosa
import numpy as np
from torch.utils import data

class MtatDataset(data.Dataset):
    def __init__(self, split, input_length,aug=False,aug_prob=0.5,aug_types=(True,True,True),
                 noise_factor=0.005,timeshift_rate=None,pitchshift_rate=None):
        self.split = split
        self.input_length = input_length
        self.augmentations = aug
        self.aug_prob = aug_prob
        self.aug_types = aug_types
        self.noise_factor = noise_factor
        self.timeshift_rate = timeshift_rate
        self.pitchshift_rate = pitchshift_rate
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
        if self.augmentations:
            apply_noise = self.aug_types[0] and np.random.rand() < self.aug_prob
            apply_stretch = self.aug_types[1] and np.random.rand() < self.aug_prob
            apply_pitchshift = self.aug_types[2] and np.random.rand() < self.aug_prob
            if apply_noise:
                noise = np.random.randn(len(npy))
                npy = npy + self.noise_factor * noise
            if apply_stretch:
                self.timeshift_rate = np.random.uniform(0.8,1.2) if self.timeshift_rate==None else self.timeshift_rate
                npy = librosa.effects.time_stretch(npy,rate=self.timeshift_rate)
            if apply_pitchshift:
                self.pitchshift_rate = np.random.randint(-5,5) if self.pitchshift_rate==None else self.pitchshift_rate
                npy = librosa.effects.pitch_shift(npy,sr=16000,n_steps=np.random.randint(-5,5))
        # ensure the length of npy is consistent with input_length
        if len(npy) > self.input_length:
            npy = npy[:self.input_length]
        elif len(npy) < self.input_length:
            padding = np.zeros((self.input_length - len(npy),), dtype=npy.dtype)
            npy = np.concatenate((npy, padding))
        return npy.astype('float32'), tag_binary.astype('float32')

    def __len__(self):
        return len(self.fl)

