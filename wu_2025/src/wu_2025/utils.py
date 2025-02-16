import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, lfilter, iirnotch, resample
from scipy.ndimage import binary_opening, binary_closing
import math

import os
from wu_2025.architecture import SeizureTransformer

def load_models(device):
    model = SeizureTransformer()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint = torch.load(os.path.join(dir_path, 'model.pth'), map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    return model


class SeizureDataset(nn.Module):
    def __init__(self, data, fs=256, window_size=15360, overlap_ratio=0.0):
        super(SeizureDataset, self).__init__()
        self.data = data
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio

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

    def __len__(self):
        if self.data.shape[1] < self.window_size:
            return 1
        return 1 + math.ceil((self.data.shape[1] - self.window_size) / ((1-self.overlap_ratio) * self.window_size))

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
        start_idx = int(idx * self.window_size * (1 - self.overlap_ratio))
        if start_idx + self.window_size + 1 > self.data.shape[1]:
            eeg_clip = self.data[:,start_idx:]
            # print('eeg_clip', eeg_clip.shape)
            pad = np.zeros((self.data.shape[0], self.window_size - eeg_clip.shape[1]))
            # print('pad', pad.shape)
            eeg_clip = np.concatenate((eeg_clip, pad), axis=1)
        else:
            eeg_clip = self.data[:, start_idx:start_idx + self.window_size]
        x = self.preprocess_clip(eeg_clip)
        return torch.FloatTensor(x)
    

def get_dataloader(data, fs, batch_size=256, window_size=15360):
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    if fs != 256:
        new_n_samples = int(data.shape[1] * float(256) / fs)
        # print('new_n_sample', new_n_samples)
        data = resample(data, new_n_samples, axis=1)
    dataset = SeizureDataset(data, fs=256, window_size=window_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def predict(model, dataloader, device, seq_len):
    total_output = None  # shape (batch_size, seq_len)
    with torch.no_grad():
        model.eval()
        for data in dataloader:
            data = data.float().to(device)
            output = model(data)
            
            if total_output is None:
                total_output = output
            else:
                total_output = torch.cat((total_output, output), 0)
    y_predict = total_output.detach().cpu().numpy()
    y_predict = y_predict.flatten()[:seq_len]

    binary_output = (y_predict > 0.8).astype(int)
    # 2) Morphological opening to remove very short 1-bursts
    binary_output = morphological_filter_1d(binary_output, operation="opening", kernel_size=5)

    # 3) Morphological closing to fill small gaps
    binary_output = morphological_filter_1d(binary_output, operation="closing", kernel_size=5)

    # 4) Remove events shorter than e.g. 2 seconds
    binary_output = remove_short_events(binary_output, min_length=2.0, fs=256)

    return binary_output



def morphological_filter_1d(binary_signal, operation="closing", kernel_size=5):
    """
    Applies a 1D morphological operation (closing or opening) to a binary time series.
    
    Parameters
    ----------
    binary_signal : 1D numpy array of shape (n_samples,)
        The binary signal (values 0 or 1).
    operation : str, optional
        Which operation to apply: "closing" or "opening".
    kernel_size : int, optional
        Length of the 1D structuring element used for dilation/erosion.
        Larger values have a stronger smoothing/merging effect.
    
    Returns
    -------
    filtered_signal : 1D numpy array of shape (n_samples,)
        The binary signal after the morphological operation.
    """
    # Construct a 1D structuring element (all ones of length 'kernel_size')
    # For time-series, we treat this as a 1D "window."
    structure = np.ones(kernel_size, dtype=bool)
    
    # Apply the selected morphological operation
    if operation == "closing":
        # Closes small holes (0’s) inside regions of 1’s
        filtered = binary_closing(binary_signal, structure=structure)
    elif operation == "opening":
        # Removes small spurious 1’s (noise) that do not fit the structuring element
        filtered = binary_opening(binary_signal, structure=structure)
    else:
        raise ValueError("operation must be 'closing' or 'opening'")
    
    return filtered.astype(int)


def remove_short_events(binary_output, min_length, fs):
    """
    binary_output: 1D numpy array of shape (n_samples,)
    min_length: minimum length in seconds
    fs: sampling frequency (samples per second)
    """
    min_samples = int(min_length * fs)
    out = binary_output.copy()
    
    # Identify “on” regions
    is_seizure = False
    start_idx = 0
    
    for i in range(len(binary_output)):
        if not is_seizure and out[i] == 1:
            # transition from 0 to 1
            is_seizure = True
            start_idx = i
        elif is_seizure and (out[i] == 0 or i == len(binary_output)-1):
            # transition from 1 to 0, or end of array
            end_idx = i if out[i] == 0 else i+1
            length = end_idx - start_idx
            
            # If the region is too short, set it to 0
            if length < min_samples:
                out[start_idx:end_idx] = 0
            is_seizure = False
    return out