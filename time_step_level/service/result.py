import json
from pathlib import Path

from epilepsy2bids.annotations import Annotations
import numpy as np
from timescoring import scoring
from timescoring.annotations import Annotation

from scipy.signal import butter, lfilter, iirnotch, resample
import torch
import torch.nn as nn
import math


class Result(scoring._Scoring):
    """Helper class built on top of scoring._Scoring that implements the sum
    operator between two scoring objects. The sum corresponds to the
    concatenation of both objects.
    Args:
        scoring (scoring._Scoring): initialized as None (all zeros) or from a
                                    scoring._Scoring object.
    """

    def __init__(self, score: scoring._Scoring = None):
        if score is None:
            self.fs = 0
            self.duration = 0
            self.numSamples = 0
            self.tp = 0
            self.fp = 0
            self.refTrue = 0
        else:
            self.fs = score.ref.fs
            self.duration = len(score.ref.mask) / score.ref.fs
            self.numSamples = score.numSamples
            self.tp = score.tp
            self.fp = score.fp
            self.refTrue = score.refTrue

    def __add__(self, other_result: scoring._Scoring):
        new_result = Result()
        new_result.fs = other_result.fs
        new_result.duration = self.duration + other_result.duration
        new_result.numSamples = self.numSamples + other_result.numSamples
        new_result.tp = self.tp + other_result.tp
        new_result.fp = self.fp + other_result.fp
        new_result.refTrue = self.refTrue + other_result.refTrue

        return new_result

    def __iadd__(self, other_result: scoring._Scoring):
        self.fs = other_result.fs
        self.duration += other_result.duration
        self.numSamples += other_result.numSamples
        self.tp += other_result.tp
        self.fp += other_result.fp
        self.refTrue += other_result.refTrue

        return self


def evaluate_dataset(
    reference: Path, hypothesis: Path, outFile: Path, avg_per_subject=True
) -> dict:
    """
    Compares two sets of seizure annotations accross a full dataset.

    Parameters:
    reference (Path): The path to the folder containing the reference TSV files.
    hypothesis (Path): The path to the folder containing the hypothesis TSV files.
    outFile (Path): The path to the output JSON file where the results are saved.
    avg_per_subject (bool): Whether to compute average scores per subject or
                            average across the full dataset.

    Returns:
    dict. return the evaluation result. The dictionary contains the following
          keys: {'sample_results': {'sensitivity', 'precision', 'f1', 'fpRate',
                    'sensitivity_std', 'precision_std', 'f1_std', 'fpRate_std'},
                 'event_results':{...}
                 }
    """

    FS = 1

    sample_results = dict()
    event_results = dict()
    for subject in Path(reference).glob("sub-*"):
        sample_results[subject.name] = Result()
        event_results[subject.name] = Result()

        for ref_tsv in subject.glob("**/*.tsv"):
            # Load reference
            ref = Annotations.loadTsv(ref_tsv)
            ref = Annotation(ref.getMask(FS), FS)

            # Load hypothesis
            hyp_tsv = Path(hypothesis) / ref_tsv.relative_to(reference)
            if hyp_tsv.exists():
                hyp = Annotations.loadTsv(hyp_tsv)
                hyp = Annotation(hyp.getMask(FS), FS)
            else:
                hyp = Annotation(np.zeros_like(ref.mask), ref.fs)

            # Compute evaluation
            sample_score = scoring.SampleScoring(ref, hyp)
            event_score = scoring.EventScoring(ref, hyp)

            # Store results
            sample_results[subject.name] += Result(sample_score)
            event_results[subject.name] += Result(event_score)

        # Compute scores
        sample_results[subject.name].computeScores()
        event_results[subject.name].computeScores()

    aggregated_sample_results = dict()
    aggregated_event_results = dict()
    if avg_per_subject:
        for result_builder, aggregated_result in zip(
            (sample_results, event_results),
            (aggregated_sample_results, aggregated_event_results),
        ):
            for metric in ["sensitivity", "precision", "f1", "fpRate"]:
                aggregated_result[metric] = np.nanmean(
                    [getattr(x, metric) for x in result_builder.values()]
                )
                aggregated_result[f"{metric}_std"] = np.nanstd(
                    [getattr(x, metric) for x in result_builder.values()]
                )
    else:
        for result_builder, aggregated_result in zip(
            (sample_results, event_results),
            (aggregated_sample_results, aggregated_event_results),
        ):
            result_builder["cumulated"] = Result()
            for result in result_builder.values():
                result_builder["cumulated"] += result
            result_builder["cumulated"].computeScores()
            for metric in ["sensitivity", "precision", "f1", "fpRate"]:
                aggregated_result[metric] = getattr(result_builder["cumulated"], metric)

    output = {
        "sample_results": aggregated_sample_results,
        "event_results": aggregated_event_results,
    }
    with open(outFile, "w") as file:
        json.dump(output, file, indent=2, sort_keys=False)

    return output


class SeizureTestingDataset(nn.Module):
    def __init__(self, data, fs=256, overlap_ratio=0.0, window_size=6000):
        super(SeizureTestingDataset, self).__init__()
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
        # return self.data.shape[1] // (self.window_size) if self.data.shape[1] % self.window_size == 0 else self.data.shape[1] // (self.window_size) + 1

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
    

class SeizureTestingDataset_window(nn.Module):
    def __init__(self, data, label, fs=256, overlap_ratio=0.0, window_size=6000):
        super(SeizureTestingDataset_window, self).__init__()
        self.data = data
        self.label = label
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
        # return self.data.shape[1] // (self.window_size) if self.data.shape[1] % self.window_size == 0 else self.data.shape[1] // (self.window_size) + 1

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

            label_clip = self.label[start_idx:]
            pad = np.zeros((self.window_size - label_clip.shape[0]))
            label_clip = np.concatenate((label_clip, pad), axis=0)
        else:
            eeg_clip = self.data[:, start_idx:start_idx + self.window_size]
            label_clip = self.label[start_idx:start_idx + self.window_size]
        x = self.preprocess_clip(eeg_clip)
        return torch.FloatTensor(x), torch.FloatTensor(label_clip)
    

def get_testingdataloader(data, batch_size=128, fs=256, window_size=15360):
    # print('data', data.shape)
    # print('detect_label', detect_label.shape)
    dataset = SeizureTestingDataset(data, window_size=window_size, fs=fs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def get_testingdataloader_window(data, label, batch_size=128, fs=256, window_size=15360):
    # print('data', data.shape)
    # print('detect_label', detect_label.shape)
    dataset = SeizureTestingDataset_window(data, label, window_size=window_size, fs=fs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader