import pyedflib
import torch.nn as nn
import torch
import numpy as np
from epilepsy2bids.eeg import Eeg
from scipy.signal import butter, lfilter, iirnotch, resample

import os

class SeizureDataset(nn.Module):
    def __init__(self, data, detection_label, window_size=15360, fs=256):
        super(SeizureDataset, self).__init__()
        self.data = data
        self.detection_label = detection_label
        self.window_size = window_size
        self.fs = fs
        self.recording_duration = int(data.shape[1] / fs)

        # params for preprocessing
        self.lowcut = 0.5
        self.highcut = 120
        notch_1_b, notch_1_a = iirnotch(1, Q=30, fs=fs)
        notch_60_b, notch_60_a = iirnotch(60, Q=30, fs=fs)
        self.notch_1_b = notch_1_b
        self.notch_1_a = notch_1_a
        self.notch_60_b = notch_60_b
        self.notch_60_a = notch_60_a

        # Example: set 75% overlap
        stride = self.window_size // 4

        # Create window indices
        self.window_idx = np.arange(
            0, 
            self.recording_duration * self.fs - self.window_size, 
            stride
        ).astype(int)

    def __len__(self):
        return len(self.window_idx)

    def butter_bandpass_filter(self, data, order=3):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        # y = filtfilt(b, a, data)
        return y

    def preprocess_clip(self, eeg_clip):
        bandpass_filtered_signal = self.butter_bandpass_filter(eeg_clip, order=3)
        filtered_1_signal = lfilter(self.notch_1_b, self.notch_1_a, bandpass_filtered_signal)
        filtered_60_signal = lfilter(self.notch_60_b, self.notch_60_a, filtered_1_signal)  
        eeg_clip = filtered_60_signal
        return eeg_clip

    def __getitem__(self, idx):
        eeg_clip = self.data[:, self.window_idx[idx]:self.window_idx[idx]+self.window_size]
        x = self.preprocess_clip(eeg_clip)
        y_detect = self.detection_label[self.window_idx[idx]:self.window_idx[idx]+self.window_size]
        # print('x', x.shape)
        # print('y_detect', y_detect.shape)
        return {'X': torch.FloatTensor(x), 
                'y_detect': y_detect, 
        }


def get_data(edf_path, label_path):
    channel_seq = ['fp1', 'f3', 'c3', 'p3', 'o1', 'f7', 't3', 't5', 'fz', 'cz', 'pz', 'fp2', 'f4', 
                   'c4', 'p4', 'o2', 'f8', 't4', 't6']
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    sample_rate = f.getSampleFrequencies()[0]
    signal_labels = f.getSignalLabels()

    # print('n', n)
    # print('sample_rate', sample_rate)
    # print('signal_labels', signal_labels)
    # get channel index based on channel_seq
    channel_idx = []
    for channel in channel_seq:
        for j in range(n):
            if signal_labels[j].lower().find(channel) != -1:
                channel_idx.append(j)
                break
        else:
            assert False, 'channel {} not found'.format(channel)
    # print('channel_idx', len(channel_idx))

    # get data
    data = np.zeros((19, f.readSignal(0).shape[0]))
    for idx, i in enumerate(channel_idx):
        temp = f.readSignal(i)
        data[idx, :] = temp
    f.close()

    # normalize data
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    # resample data
    if sample_rate != 256:
        new_n_samples = int(data.shape[1] / sample_rate * float(256))
        data = resample(data, new_n_samples, axis=1)

    # read csv_bi file
    with open(label_path, 'r') as f:
        lines = f.readlines()
    detect_label = np.zeros((data.shape[1]))
    start = True
    for line in lines:
        if '#' in line:
            continue
        if start:
            start = False
            continue
        _, start_time, end_time, cls, _ = line.strip().split(',')
        if cls == 'bckg':
            continue
        start_time = int(float(start_time) * 256)
        end_time = int(float(end_time) * 256)
        # print('start_time', start_time) 
        # print('end_time', end_time)
        detect_label[start_time:end_time] = 1
    
    detect_label = torch.tensor(detect_label).float()
    return data, detect_label


def get_data_18(edf_path, label_path):
    eeg = Eeg.loadEdf(edfFile=edf_path)
    eeg.reReferenceToBipolar()
    data = eeg.data

    # normalize data
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    # resample data
    sample_rate = eeg.fs
    if sample_rate != 256:
        new_n_samples = int(data.shape[1] / sample_rate * float(256))
        data = resample(data, new_n_samples, axis=1)

    # read csv_bi file
    with open(label_path, 'r') as f:
        lines = f.readlines()
    detect_label = np.zeros((data.shape[1]))
    start = True
    for line in lines:
        if '#' in line:
            continue
        if start:
            start = False
            continue
        _, start_time, end_time, cls, _ = line.strip().split(',')
        if cls == 'bckg':
            continue
        start_time = int(float(start_time) * 256)
        end_time = int(float(end_time) * 256)
        # print('start_time', start_time) 
        # print('end_time', end_time)
        detect_label[start_time:end_time] = 1
    
    detect_label = torch.tensor(detect_label).float()
    return data, detect_label

def get_dataloader(data, detect_label, batch_size=128, fs=256, window_size=256*60):
    # print('data', data.shape)
    # print('detect_label', detect_label.shape)
    dataset = SeizureDataset(data, detect_label, window_size=window_size, fs=fs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def get_file(path):
    file_list = []
    label_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.edf') and not file.startswith('_') and not file.startswith('.'):
                name = file.split('.')[0]
                file_list.append(os.path.join(root, file))
                label_list.append(os.path.join(root, name + '.csv_bi'))
    return file_list, label_list