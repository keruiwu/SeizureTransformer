from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing import create_windows_from_events
import numpy as np
from braindecode.preprocessing.preprocess import preprocess, Preprocessor


from numpy import multiply
from braindecode.preprocessing.windowers import create_windows_from_events
from sklearn.preprocessing import scale as standard_scale

import numpy as np
import torch
import os

n_jobs = 4

# -- convert the data to microvolts and apply a lowpass filter. Since the Sleep Physionet data is already sampled at 100 Hz
high_cut_hz = 30
# Factor to convert from V to uV
factor = 1e6


preprocessors = [
    Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
    Preprocessor('filter', l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs)
]

# Transform the data


# --  extract 30-s windows to be used in both the pretext and downstream tasks. 

window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq

mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

# -- reprocess the windows by applying channel-wise z-score normalization

dataset_fold = "./sleep_edf_full/"

train_dataset_fold = dataset_fold + "TrainFold/"
valid_dataset_fold = dataset_fold + "ValidFold/"
test_dataset_fold = dataset_fold + "TestFold/"


# subjects = np.unique(windows_dataset.description['subject'])
np.random.seed(7)
for sub in range(83): # This dataset contains subjects 0 to 82 with missing subjects [39, 68, 69, 78, 79].
    if sub in [39, 68, 69, 78, 79]: continue
    r = np.random.rand()
    if r<0.6: 
        save_path = train_dataset_fold
    elif r<0.8:
        save_path = valid_dataset_fold
    else:
        save_path = test_dataset_fold
        
    dataset = SleepPhysionet(subject_ids=[sub], 
        crop_wake_mins=30)#recording_ids=[1], 

    preprocess(dataset, preprocessors)
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True, mapping=mapping)

    preprocess(windows_dataset, [Preprocessor(standard_scale, channel_wise=True)])
    for i, x in enumerate(windows_dataset):
        path = save_path+str(x[1])+'/'
        os.makedirs(path, exist_ok=True)
        path+= f"s{sub}_{x[2][1]}_{x[2][2]}.pt"
        torch.save(torch.tensor(x[0]), path)
        
