# -*- coding: utf-8 -*-
import cv2
import os
import re
import numpy as np
import pandas as pd
import scipy.io.wavfile
from torch.utils.data import Dataset
from math import ceil

from path import root_path


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

"""
    #####################################
    Adversarial training dataset
    #####################################
"""


class Adv_Dataset(Dataset):
    def __init__(self, dataset, image_path, landmark_path, transform, rescale):
        super(Adv_Dataset, self).__init__()
        self.dataset = dataset
        self.image_path = image_path
        self.landmark_path = landmark_path
        self.transform = transform
        self.rescale = rescale

    def read_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.rescale, self.rescale), interpolation=cv2.INTER_CUBIC)
        if self.transform:
            img = self.transform(img)
        return np.asarray(img)

    def read_landmark(self, landmark_path):
        landmark = pd.read_csv(landmark_path)
        landmark_x = []
        landmark_y = []
        for i in range(68):
            landmark_x.append(landmark['x{}'.format(i)][0])
            landmark_y.append(landmark['y{}'.format(i)][0])
        landmark_x = np.asarray(landmark_x) / 256.
        landmark_y = np.asarray(landmark_y) / 256.
        landmark = np.append(landmark_x, landmark_y)[:, np.newaxis, np.newaxis]
        return landmark

    def __getitem__(self, idx):
        img_path = './dataset/{}/{}'.format(self.dataset, self.image_path[idx])
        img = self.read_img(img_path)
        landmark_path = './dataset/{}/{}'.format(self.dataset, self.landmark_path[idx])
        landmark = self.read_landmark(landmark_path)
        return img, landmark

    def __len__(self):
        return len(self.image_path)


class Stage_II_Dataset(Adv_Dataset):
    def __init__(self, dataset, data_path, score, transform, rescale, t_len):
        super(Adv_Dataset, self).__init__()
        self.dataset = dataset
        self.image_path = data_path
        self.score = score
        self.transform = transform
        self.rescale = rescale
        self.t_len = t_len

    def read_score(self, idx):
        return self.score[idx] / 63.

    def __getitem__(self, idx):
        rand_flag = np.random.random()
        img_path = '{}/dataset/{}/{}'.format(root_path, self.dataset, self.image_path[idx])
        landmark_path = '{}/dataset/avec14/landmark_aligned_merge/{}.csv'.format(root_path, self.image_path[idx])

        rand_range = len(landmark_path) - self.t_len - 1
        if rand_range < 0:
            rand_range = 1
        rand_flag = int(rand_flag * rand_range)

        landmark = pd.read_csv(landmark_path)
        landmark_batch = np.zeros((self.t_len, 136, 1, 1))
        data_batch_len = len(landmark['x0'][rand_flag:rand_flag + self.t_len])
        for i in range(68):
            landmark_batch[0:data_batch_len, 2 * i, 0, 0] = landmark['x{}'.format(i)][rand_flag:rand_flag + self.t_len]
            landmark_batch[0:data_batch_len, 2 * i + 1, 0, 0] = landmark['y{}'.format(i)][
                                                                rand_flag:rand_flag + self.t_len]
        landmark_batch = landmark_batch / 256.

        img_batch = np.zeros((self.t_len, 3, 64, 64))

        image_name = os.listdir(img_path)
        image_name.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))

        for i in range(self.t_len):
            try:
                img_batch[i, :, :, :] = self.read_img(img_path + '/' + image_name[i+rand_flag])
            except IndexError:
                print('deprecate invalid frames {}'.format(image_name[i+rand_flag]))
                print(landmark_batch)

        score = self.read_score(idx)
        return img_batch, landmark_batch, score




"""
    #####################################
    Audio dataset
    #####################################
"""


class AVEC17_Audio_Dataset(Dataset):
    def __init__(self, audio_list, label_list, wlen, t_len):
        super(AVEC17_Audio_Dataset, self).__init__()
        self.audio_list = audio_list
        self.label_list = label_list
        self.wlen = wlen
        self.t_len = t_len

    def read_score(self, idx):
        return self.label_list[idx]

    def read_train_audio(self, audio_path):
        [fs, signal] = scipy.io.wavfile.read(audio_path)
        if len(signal.shape) > 1:
            signal = (signal[:, 0] + signal[:, 1]) / 2
        signal = signal * 1.0 / (max(abs(signal)))
        snt_len = signal.shape[0]
        rand_range = snt_len - self.wlen * self.t_len - 1
        if rand_range < 0:
            rand_range = 1
        rand_flag = np.random.randint(rand_range)
        signal_tensor = np.zeros((self.t_len, self.wlen))

        for i in range(self.t_len):
            start = rand_flag + i * self.wlen
            end = start + self.wlen
            if end > len(signal[start:end]):
                signal_tensor[i, 0:len(signal[start:end])] = signal[start:end]
            else:
                signal_tensor[i, :] = signal[start:end]
        return signal_tensor

    def read_audio(self, idx):
        audio_path = './dataset/avec17/audio/{}_P_AUDIO.wav'.format(self.audio_list[idx])
        return self.read_train_audio(audio_path)

    def __getitem__(self, idx):
        audio = self.read_audio(idx)
        label = self.read_score(idx)
        return audio, label

    def __len__(self):
        return len(self.label_list)


class AVEC17_Audio_Val_Dataset(AVEC17_Audio_Dataset):
    def __init__(self, audio_list, label_list, wlen, t_len):
        super(AVEC17_Audio_Dataset, self).__init__()
        self.audio_list = audio_list
        self.label_list = label_list
        self.wlen = wlen
        self.t_len = t_len

    def read_val_audio(self, audio_path):
        [fs, signal] = scipy.io.wavfile.read(audio_path)
        if len(signal.shape) > 1:
            signal = (signal[:, 0] + signal[:, 1]) / 2
        signal = signal * 1.0 / (max(abs(signal)))
        snt_len = signal.shape[0]

        section = int(snt_len / self.wlen)
        slice_idx = [i * self.wlen for i in range(section)]

        val_audio = []
        audio_seg = []
        for batch_idx, sub_idx in enumerate(slice_idx):
            audio_seg.append(signal[sub_idx:sub_idx + self.wlen])
            if (batch_idx + 1) % self.t_len == 0:
                audio_seg = np.asarray(audio_seg)
                val_audio.append(audio_seg)
                audio_seg = []
        if audio_seg:
            audio_seg = np.asarray(audio_seg)  # (338, wlen)
            audio_seg = np.pad(audio_seg, ((0, self.t_len - audio_seg.shape[0]), (0, 0)),
                               'constant', constant_values=(0, 0))  # (batchsize, wlen)
            val_audio.append(audio_seg)

        return val_audio

    def read_audio(self, idx):
        audio_path = './dataset/avec17/audio/{}_P_AUDIO.wav'.format(self.audio_list[idx])
        return self.read_val_audio(audio_path)

    def __getitem__(self, idx):
        audio = self.read_audio(idx)
        label = self.read_score(idx)
        return audio, label

    def __len__(self):
        return len(self.label_list)


class AVEC14_Audio_Dataset(AVEC17_Audio_Dataset):
    def __init__(self, audio_list, label_list, wlen, t_len):
        super(AVEC17_Audio_Dataset, self).__init__()
        self.audio_list = audio_list
        self.label_list = label_list
        self.wlen = wlen
        self.t_len = t_len

    def read_audio(self, idx):
        data_idx = self.audio_list[idx].split('/')
        audio_path = '{}/dataset/avec14/{}/audio/{}/{}.wav'.format(root_path, data_idx[0], data_idx[1], data_idx[2])
        return self.read_train_audio(audio_path)

    def __getitem__(self, idx):
        audio = self.read_audio(idx)
        label = self.read_score(idx)
        return audio, label

    def __len__(self):
        return len(self.label_list)


class AVEC14_Audio_Val_Dataset(AVEC17_Audio_Val_Dataset):
    def __init__(self, audio_list, label_list, wlen, t_len):
        super(AVEC17_Audio_Dataset, self).__init__()
        self.audio_list = audio_list
        self.label_list = label_list
        self.wlen = wlen
        self.t_len = t_len

    def read_audio(self, idx):
        data_idx = self.audio_list[idx].split('/')
        audio_path = '{}/dataset/avec14/{}/audio/{}/{}.wav'.format(root_path, data_idx[0], data_idx[1], data_idx[2])
        return self.read_val_audio(audio_path)

    def __getitem__(self, idx):
        audio = self.read_audio(idx)
        label = self.read_score(idx)
        return audio, label

    def __len__(self):
        return len(self.label_list)


"""
    #####################################
    Landmark dataset
    #####################################
    
    read landmarks as batches
"""


class AVEC14_Landmark_Pack_Dataset(Dataset):
    def __init__(self, data_list, label_list, t_len):
        super(AVEC14_Landmark_Pack_Dataset, self).__init__()
        self.data_list = data_list
        self.label_list = label_list
        self.t_len = t_len

    def read_score(self, idx):
        return self.label_list[idx]

    def read_train_data(self, landmark_path, rand_flag):
        landmark = pd.read_csv(landmark_path)
        landmark_batch = np.zeros((self.t_len, 136, 1, 1))
        rand_range = len(landmark) - self.t_len - 1
        if rand_range < 0:
            rand_range = 1

        rand_flag = int(rand_flag * rand_range)

        data_batch_len = len(landmark['x0'][rand_flag:rand_flag + self.t_len])
        for i in range(68):
            landmark_batch[0:data_batch_len, 2*i, 0, 0] = landmark['x{}'.format(i)][rand_flag:rand_flag + self.t_len]
            landmark_batch[0:data_batch_len, 2*i + 1, 0, 0] = landmark['y{}'.format(i)][rand_flag:rand_flag + self.t_len]
        landmark_batch = landmark_batch / 256.

        return landmark_batch

    def __getitem__(self, idx):
        rand_flag = np.random.random()
        landmark_folder = 'dataset/avec14/landmark_aligned_merge/{}.csv'.format(self.data_list[idx])
        landmark_batch = self.read_train_data(landmark_folder, rand_flag)  # (batchsize, 136, 1, 1)
        score = self.label_list[idx]
        return landmark_batch, score

    def __len__(self):
        return len(self.label_list)


class AVEC14_Landmark_Val_Pack_Dataset(AVEC14_Landmark_Pack_Dataset):
    def __init__(self, data_list, label_list, t_len):
        super(AVEC14_Landmark_Val_Pack_Dataset, self).__init__(data_list, label_list, t_len)

    def read_val_data(self, landmark_path, rand_flag_list):
        val_landmark = []
        for rand_flag in rand_flag_list:
            val_landmark.append(self.read_train_data(landmark_path, rand_flag))
        return val_landmark

    def __getitem__(self, idx):
        rand_flag = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        landmark_folder = 'dataset/avec14/landmark_aligned_merge/{}.csv'.format(self.data_list[idx])
        landmark_batch = self.read_val_data(landmark_folder, rand_flag)  # (LEN, batchsize, 136, 1, 1)
        score = self.label_list[idx]
        return landmark_batch, score

    def __len__(self):
        return len(self.label_list)


class AVEC17_Landmark_Pack_Dataset(AVEC14_Landmark_Pack_Dataset):
    def __init__(self, data_list, label_list, t_len):
        super(AVEC17_Landmark_Pack_Dataset, self).__init__(data_list, label_list, t_len)

    def __getitem__(self, idx):
        rand_flag = np.random.random()
        landmark_folder = 'dataset/avec17/landmark_aligned/{}_landmark_aligned.csv'.format(self.data_list[idx])
        landmark_batch = self.read_train_data(landmark_folder, rand_flag)  # (batchsize, 136, 1, 1)
        score = self.label_list[idx]
        return landmark_batch, score

    def __len__(self):
        return len(self.label_list)


class AVEC17_Landmark_Val_Pack_Dataset(AVEC14_Landmark_Val_Pack_Dataset):
    def __init__(self, data_list, label_list, t_len):
        super(AVEC17_Landmark_Val_Pack_Dataset, self).__init__(data_list, label_list, t_len)

    def __getitem__(self, idx):
        rand_flag = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        landmark_folder = 'dataset/avec17/landmark_aligned/{}_landmark_aligned.csv'.format(self.data_list[idx])
        landmark_batch = self.read_val_data(landmark_folder, rand_flag)  # (LEN, batchsize, 136, 1, 1)
        score = self.label_list[idx]
        return landmark_batch, score

    def __len__(self):
        return len(self.label_list)


"""
    read landmark as single frame
"""


class AVEC17_Landmark_Dataset(Dataset):
    def __init__(self, landmark_path, score):
        super(AVEC17_Landmark_Dataset, self).__init__()
        self.landmark_path = landmark_path
        self.score = score

    def read_landmark(self, landmark_path):
        landmark = pd.read_csv(landmark_path)
        landmark_x = []
        landmark_y = []
        rand = np.random.randint(len(landmark))
        # rand = 0
        for i in range(68):
            landmark_x.append(landmark['x{}'.format(i)][rand])
            landmark_y.append(landmark['y{}'.format(i)][rand])
        landmark_x = np.asarray(landmark_x) / 256.
        landmark_y = np.asarray(landmark_y) / 256.
        landmark = np.append(landmark_x, landmark_y)[:, np.newaxis, np.newaxis]
        return landmark

    def __getitem__(self, idx):
        landmark_path = 'dataset/avec17/landmark_aligned/{}_landmark_aligned.csv'.format(self.landmark_path[idx])
        landmark = self.read_landmark(landmark_path)
        score = self.score[idx] / 23.
        return landmark, score

    def __len__(self):
        return len(self.landmark_path)


class AVEC17_Landmark_Val_Dataset(Dataset):
    def __init__(self, landmark_path, score, t_len):
        super(AVEC17_Landmark_Val_Dataset, self).__init__()
        self.landmark_path = landmark_path
        self.score = score
        self.t_len = t_len

    def read_landmark(self, landmark_path):

        landmark = pd.read_csv(landmark_path)
        snt_len = len(landmark)

        section = ceil(snt_len / self.t_len)
        slice_idx = [int(i * (self.t_len - ((section * self.t_len - snt_len) / int(section - 1))))
                     for i in range(section)]

        val_landmark_pack = []

        for batch_idx, sub_idx in enumerate(slice_idx):

            landmark_seg_tensor = np.zeros((self.t_len, 136, 1, 1))

            for inner_seg_id in range(self.t_len):

                landmark_x = []
                landmark_y = []
                for i in range(68):
                    landmark_x.append(landmark['x{}'.format(i)][sub_idx + inner_seg_id])
                    landmark_y.append(landmark['y{}'.format(i)][sub_idx + inner_seg_id])

                landmark_x.extend(landmark_y)
                landmark_seg_tensor[inner_seg_id, :, 0, 0] = landmark_x

            val_landmark_pack.append(landmark_seg_tensor / 256.)

        return val_landmark_pack

    def __getitem__(self, idx):
        landmark_path = 'dataset/avec17/landmark_aligned/{}_landmark_aligned.csv'.format(self.landmark_path[idx])
        landmark = self.read_landmark(landmark_path)
        score = self.score[idx] / 23.
        return landmark, score

    def __len__(self):
        return len(self.landmark_path)


"""
    #####################################
    multimodal dataset
    #####################################
"""

class AVEC14_MM_Pack_Dataset(AVEC14_Landmark_Pack_Dataset):
    def __init__(self, data_list, label_list, wlen, t_len):
        super(AVEC14_MM_Pack_Dataset, self).__init__(data_list, label_list, t_len)
        self.wlen = wlen

    def read_score(self, idx):
        return self.label_list[idx]

    def read_train_audio(self, audio_path, rand_flag):
        [fs, signal] = scipy.io.wavfile.read(audio_path)
        if len(signal.shape) > 1:
            signal = (signal[:, 0] + signal[:, 1]) / 2
        signal = signal * 1.0 / (max(abs(signal)))
        snt_len = signal.shape[0]
        rand_range = snt_len - self.wlen * self.t_len - 1
        if rand_range < 0:
            rand_range = 1
        rand_flag = int(rand_flag * rand_range)
        signal_tensor = np.zeros((self.t_len, self.wlen))

        for i in range(self.t_len):
            start = rand_flag + i * self.wlen
            end = start + self.wlen
            if end > len(signal[start:end]):
                signal_tensor[i, 0:len(signal[start:end])] = signal[start:end]
            else:
                signal_tensor[i, :] = signal[start:end]
        return signal_tensor

    def read_audio(self, idx, rand_flag):
        data_idx = self.data_list[idx].split('/')
        audio_path = '{}/dataset/avec14/{}/audio/{}/{}.wav'.format(root_path, data_idx[0], data_idx[1], data_idx[2])
        return self.read_train_audio(audio_path, rand_flag)

    def __getitem__(self, idx):
        rand_flag = np.random.random()
        landmark_folder = 'dataset/avec14/landmark_aligned_merge/{}.csv'.format(self.data_list[idx])
        landmark_batch = self.read_train_data(landmark_folder, rand_flag)  # (batchsize, 136, 1, 1)
        score = self.label_list[idx]
        audio = self.read_audio(idx, rand_flag)
        return landmark_batch, audio, score

    def __len__(self):
        return len(self.label_list)


class AVEC14_MM_Val_Pack_Dataset(AVEC14_MM_Pack_Dataset):
    def __init__(self, data_list, label_list, wlen, t_len):
        super(AVEC14_MM_Val_Pack_Dataset, self).__init__(data_list, label_list, wlen, t_len)

    def read_audio(self, idx, rand_flag):
        data_idx = self.data_list[idx].split('/')
        audio_path = '{}/dataset/avec14/{}/audio/{}/{}.wav'.format(root_path, data_idx[0], data_idx[1], data_idx[2])
        audio = []
        for i in rand_flag:
            audio.append(self.read_train_audio(audio_path, i))
        return audio

    def read_val_data(self, landmark_path, rand_flag_list):
        val_landmark = []
        for rand_flag in rand_flag_list:
            val_landmark.append(self.read_train_data(landmark_path, rand_flag))
        return val_landmark

    def __getitem__(self, idx):
        rand_flag_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        landmark_folder = 'dataset/avec14/landmark_aligned_merge/{}.csv'.format(self.data_list[idx])
        landmark_batch = self.read_val_data(landmark_folder, rand_flag_list)  # (LEN, batchsize, 136, 1, 1)
        score = self.label_list[idx]
        audio = self.read_audio(idx, rand_flag_list)
        return landmark_batch, audio, score

    def __len__(self):
        return len(self.label_list)


class AVEC17_MM_Pack_Dataset(AVEC14_MM_Pack_Dataset):
    def __init__(self, data_list, label_list, wlen, t_len):
        super(AVEC17_MM_Pack_Dataset, self).__init__(data_list, label_list, wlen, t_len)

    def read_audio(self, idx, rand_flag):
        audio_path = './dataset/avec17/audio/{}_P_AUDIO.wav'.format(self.data_list[idx])
        return self.read_train_audio(audio_path, rand_flag)

    def __getitem__(self, idx):
        rand_flag = np.random.random()
        landmark_folder = 'dataset/avec17/landmark_aligned/{}_landmark_aligned.csv'.format(self.data_list[idx])
        landmark_batch = self.read_train_data(landmark_folder, rand_flag)  # (batchsize, 136, 1, 1)
        score = self.label_list[idx]
        audio = self.read_audio(idx, rand_flag)
        return landmark_batch, audio, score

    def __len__(self):
        return len(self.label_list)


class AVEC17_MM_Val_Pack_Dataset(AVEC14_MM_Val_Pack_Dataset):
    def __init__(self, data_list, label_list, wlen, t_len):
        super(AVEC17_MM_Val_Pack_Dataset, self).__init__(data_list, label_list, wlen, t_len)

    def read_audio(self, idx, rand_flag):
        audio_path = './dataset/avec17/audio/{}_P_AUDIO.wav'.format(self.data_list[idx])
        audio = []
        for i in rand_flag:
            audio.append(self.read_train_audio(audio_path, i))
        return audio

    def __getitem__(self, idx):
        rand_flag = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        landmark_folder = 'dataset/avec17/landmark_aligned/{}_landmark_aligned.csv'.format(self.data_list[idx])
        landmark_batch = self.read_val_data(landmark_folder, rand_flag)  # (LEN, batchsize, 136, 1, 1)
        score = self.label_list[idx]
        audio = self.read_audio(idx, rand_flag)
        return landmark_batch, audio, score

    def __len__(self):
        return len(self.label_list)
