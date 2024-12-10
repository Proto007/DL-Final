'''
Deformation codes are borrowed from MUDA
McFee et al., A software framework for musical data augmentation, 2015
https://github.com/bmcfee/muda
'''
import os
import numpy as np
import tqdm
from sklearn import metrics
import librosa

import torch
import torch.nn as nn
from torch.autograd import Variable

import model as Model

class PredictNoisy(object):
    def __init__(self, model_type, model_load_path, batch_size=16, mod='time_stretch', rate=0):
        self.model_type = model_type
        self.model_load_path = model_load_path
        self.batch_size = batch_size
        self.is_cuda = torch.cuda.is_available()
        self.build_model()
        self.get_dataset()
        self.mod = mod
        self.rate = rate

    def get_model(self):
        if self.model_type == 'fcn':
            self.input_length = 29 * 16000
            return Model.FCN()
        elif self.model_type == 'musicnn':
            self.input_length = 3 * 16000
            return Model.Musicnn()
        elif self.model_type == 'crnn':
            self.input_length = 29 * 16000
            return Model.CRNN()
        elif self.model_type == 'sample':
            self.input_length = 59049
            return Model.SampleCNN()
        elif self.model_type == 'se':
            self.input_length = 59049
            return Model.SampleCNNSE()
        elif self.model_type == 'short':
            self.input_length = 59049
            return Model.ShortChunkCNN()
        elif self.model_type == 'short_res':
            self.input_length = 59049
            return Model.ShortChunkCNN_Res()
        elif self.model_type == 'hcnn':
            self.input_length = 5 * 16000
            return Model.HarmonicCNN()
        elif self.model_type == 'vit':
            self.input_length = 15 * 16000
            return Model.ViT()
        else:
            print('model_type has to be one of [fcn, musicnn, crnn, sample, se, short, short_res, vit]')

    def build_model(self):
        self.model = self.get_model()
        # cuda
        if self.is_cuda:
            self.model.cuda()
        # load model
        self.load(self.model_load_path)

    def get_dataset(self):
        self.test_list = np.load('./split/test.npy')
        self.train_list = np.load('./split/train.npy')
        self.validation_list = np.load('./split/valid.npy')
        self.binary = np.load('./split/binary.npy')

    def load(self, filename):
        S = torch.load(filename,weights_only=True)
        self.model.load_state_dict(S)

    def to_var(self, x):
        if self.is_cuda:
            x = x.cuda()
        return Variable(x)

    def get_tensor(self, fn):
        # load audio
        npy_path = os.path.join('data', 'npy', fn.split('/')[1][:-3]) + 'npy'
        raw = np.load(npy_path, mmap_mode='r')
        raw = self.modify(raw, self.rate, self.mod)
        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x

    def modify(self, x, mod_rate, mod_type):
        if mod_type == 'time_stretch':
            return self.time_stretch(x, mod_rate)
        elif mod_type == 'pitch_shift':
            return self.pitch_shift(x, mod_rate)
        elif mod_type == 'white_noise':
            return self.white_noise(x, mod_rate)
        else:
            print('choose from [time_stretch, pitch_shift, dynamic_range, white_noise]')

    def time_stretch(self, x, rate):
        '''
        [2 ** (-.5), 2 ** (.5)]
        '''
        res = librosa.effects.time_stretch(x,rate=rate)
        if len(res) > self.input_length:
            res = res[:self.input_length]
        elif len(res) < self.input_length:
            padding = np.zeros((self.input_length - len(res),), dtype=res.dtype)
            res = np.concatenate((res, padding))
        return res

    def pitch_shift(self, x, rate):
        '''
        [-1, 1]
        '''
        return librosa.effects.pitch_shift(x,sr=16000,n_steps=rate)

    def white_noise(self, x, rate):
        '''
        [0.1, 0.4]
        '''
        n_frames = len(x)
        noise_white = np.random.RandomState().randn(n_frames)
        noise_fft = np.fft.rfft(noise_white)
        colored_filter = np.linspace(1, n_frames / 2 + 1, n_frames // 2 + 1) ** 0
        noise_filtered = noise_fft * colored_filter
        noise = librosa.util.normalize(np.fft.irfft(noise_filtered)) * (x.max())
        if len(noise) < len(x):
            x = x[:len(noise)]
        return (1 - rate) * x + (noise * rate)

    def get_auc(self, est_array, gt_array):
        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        return roc_aucs, pr_aucs

    def test(self):
        roc_auc, pr_auc, loss = self.get_test_score()
        print('loss: %.4f' % loss)
        print('roc_auc: %.4f' % roc_auc)
        print('pr_auc: %.4f' % pr_auc)

    def get_test_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCELoss()
        for line in tqdm.tqdm(self.test_list):
            ix, fn = line.split('\t')
            # load and split
            x = self.get_tensor(fn)
            # ground truth
            ground_truth = self.binary[int(ix)]
            # forward
            x = self.to_var(x)
            y = torch.tensor(np.repeat(ground_truth.astype('float32')[np.newaxis, :], self.batch_size, axis=0))
            if self.is_cuda:
                y = y.cuda() 
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()
            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(ground_truth)

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        return roc_auc, pr_auc, loss
    
    def get_train_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCELoss()
        for line in tqdm.tqdm(self.train_list):
            ix, fn = line.split('\t')
            # load and split
            x = self.get_tensor(fn)
            # ground truth
            ground_truth = self.binary[int(ix)]
            # forward
            x = self.to_var(x)
            y = torch.tensor(np.repeat(ground_truth.astype('float32')[np.newaxis, :], self.batch_size, axis=0))
            if self.is_cuda:
                y = y.cuda() 
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()
            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(ground_truth)

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        return roc_auc, pr_auc, loss

    def get_validation_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCELoss()
        for line in tqdm.tqdm(self.validation_list):
            ix, fn = line.split('\t')
            # load and split
            x = self.get_tensor(fn)
            # ground truth
            ground_truth = self.binary[int(ix)]
            # forward
            x = self.to_var(x)
            y = torch.tensor(np.repeat(ground_truth.astype('float32')[np.newaxis, :], self.batch_size, axis=0))
            if self.is_cuda:
                y = y.cuda() 
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()
            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(ground_truth)

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        return roc_auc, pr_auc, loss

