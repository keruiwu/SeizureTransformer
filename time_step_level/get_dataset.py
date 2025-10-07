from service.handle_data import get_file, get_data, get_dataloader, get_data_18
from tqdm import tqdm
import argparse
import os

import numpy as np
from sklearn.utils import shuffle
from scipy.signal import resample
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg

def parse_args():
    parser = argparse.ArgumentParser(description='Get ETHZ dataset')
    parser.add_argument('--window_size', type=int, default=15360, help='Window size')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha value for training data')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta value for training data')
    args = parser.parse_args()
    return args


def get_siena(window_size, args):
    print('=========Siena=========')
    full_window_li = []
    full_label_li = []
    no_window_li = []
    no_label_li = []
    valid_window_li = []
    valid_label_li = []

    file_list, label_list = get_file('./data/BIDS_Siena')

    progress = tqdm(file_list)
    for edf_path in progress:
        eeg = Eeg.loadEdfAutoDetectMontage(edf_path)
        eeg.reReferenceToBipolar()
        data = eeg.data
        # normalize data
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        # resample data
        sample_rate = eeg.fs
        if sample_rate != 256:
            new_n_samples = int(data.shape[1] / sample_rate * float(256))
            data = resample(data, new_n_samples, axis=1)
        if data.shape[1] < window_size:
            continue
        name = edf_path.split('/')[-1].split('.')[0]
        label_name = name.split('_')
        label_name = '_'.join(label_name[:-1]) + '_events.tsv'
        label_path = os.path.join('/'.join(edf_path.split('/')[:-1]), label_name)
        ref = Annotations.loadTsv(label_path)
        detect_label = ref.getMask(fs=256)

        dataloader = get_dataloader(data, detect_label, window_size=window_size)
        cnt = 0
        for batch in dataloader:
            print('cnt:', cnt)
            cnt += 1

            X = batch['X']
            y_detect = batch['y_detect']

            X = X.detach().numpy()
            y_detect = y_detect.detach().numpy()
            # print('x:', X.shape)
            # print('y_detect:', y_detect.shape)

            for mini_batch in range(X.shape[0]):
                seizure_sample_cnt = np.sum(y_detect[mini_batch, :])
                temp = X[mini_batch]
                # print('temp', temp.shape)
                if seizure_sample_cnt == 0:
                    no_window_li.append(temp)
                    no_label_li.append(y_detect[mini_batch, :])
                elif seizure_sample_cnt == window_size:
                    full_window_li.append(temp)
                    full_label_li.append(y_detect[mini_batch, :])
                else:
                    valid_window_li.append(temp)
                    valid_label_li.append(y_detect[mini_batch, :])
            progress.set_description(f'ps: {len(valid_window_li)}, fs: {len(full_window_li)}, ns: {len(no_window_li)}')
    print('full_window:', len(full_window_li))
    print('no_window:', len(no_window_li))
    print('valid_window:', len(valid_window_li))

    full_window_li, full_label_li = shuffle(full_window_li, full_label_li)
    no_window_li, no_label_li = shuffle(no_window_li, no_label_li)
    valid_window_li, valid_label_li = shuffle(valid_window_li, valid_label_li)

    no_window_li = no_window_li[:int(len(valid_window_li) * 3)]
    no_label_li = no_label_li[:int(len(valid_label_li) * 3)]

    np.save(f'./data/dataset/siena_18_full_window_{window_size}.npy', np.array(full_window_li))
    np.save(f'./data/dataset/siena_18_full_label_{window_size}.npy', np.array(full_label_li))
    np.save(f'./data/dataset/siena_18_no_window_{window_size}.npy', np.array(no_window_li))
    np.save(f'./data/dataset/siena_18_no_label_{window_size}.npy', np.array(no_label_li))
    np.save(f'./data/dataset/siena_18_valid_window_{window_size}.npy', np.array(valid_window_li))
    np.save(f'./data/dataset/siena_18_valid_label_{window_size}.npy', np.array(valid_label_li))

    return full_window_li, full_label_li, no_window_li, no_label_li, valid_window_li, valid_label_li


def get_mit(window_size, args):
    """
    MIT Seizure Corpus
    """
    print('=========MIT=========')
    full_window_li = []
    full_label_li = []
    no_window_li = []
    no_label_li = []
    valid_window_li = []
    valid_label_li = []

    file_list, label_list = get_file('./data/BIDS_CHB-MIT')

    progress = tqdm(file_list)
    for edf_path in progress:
        eeg = Eeg.loadEdfAutoDetectMontage(edf_path)
        data = eeg.data
        # normalize data
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        # resample data
        sample_rate = eeg.fs
        if sample_rate != 256:
            new_n_samples = int(data.shape[1] / sample_rate * float(256))
            data = resample(data, new_n_samples, axis=1)
        if data.shape[1] < window_size:
            continue
        name = edf_path.split('/')[-1].split('.')[0]
        label_name = name.split('_')
        label_name = '_'.join(label_name[:-1]) + '_events.tsv'
        label_path = os.path.join('/'.join(edf_path.split('/')[:-1]), label_name)
        ref = Annotations.loadTsv(label_path)
        detect_label = ref.getMask(fs=256)

        dataloader = get_dataloader(data, detect_label, window_size=window_size)
        for batch in dataloader:
            X = batch['X']
            y_detect = batch['y_detect']

            X = X.detach().numpy()
            y_detect = y_detect.detach().numpy()
            # print('x:', X.shape)
            # print('y_detect:', y_detect.shape)

            for mini_batch in range(X.shape[0]):
                seizure_sample_cnt = np.sum(y_detect[mini_batch, :])
                temp = X[mini_batch]
                # print('temp', temp.shape)
                if seizure_sample_cnt == 0:
                    no_window_li.append(temp)
                    no_label_li.append(y_detect[mini_batch, :])
                elif seizure_sample_cnt == window_size:
                    full_window_li.append(temp)
                    full_label_li.append(y_detect[mini_batch, :])
                else:
                    valid_window_li.append(temp)
                    valid_label_li.append(y_detect[mini_batch, :])
            progress.set_description(f'ps: {len(valid_window_li)}, fs: {len(full_window_li)}, ns: {len(no_window_li)}')
    print('full_window:', len(full_window_li))
    print('no_window:', len(no_window_li))
    print('valid_window:', len(valid_window_li))

    full_window_li, full_label_li = shuffle(full_window_li, full_label_li)
    no_window_li, no_label_li = shuffle(no_window_li, no_label_li)

    no_window_li = no_window_li[:int(len(valid_window_li) * 3)]
    no_label_li = no_label_li[:int(len(valid_label_li) * 3)]

    valid_window_li, valid_label_li = shuffle(valid_window_li, valid_label_li)

    np.save(f'./data/dataset/mit_18_full_window_{window_size}.npy', np.array(full_window_li))
    np.save(f'./data/dataset/mit_18_full_label_{window_size}.npy', np.array(full_label_li))
    np.save(f'./data/dataset/mit_18_no_window_{window_size}.npy', np.array(no_window_li))
    np.save(f'./data/dataset/mit_18_no_label_{window_size}.npy', np.array(no_label_li))
    np.save(f'./data/dataset/mit_18_valid_window_{window_size}.npy', np.array(valid_window_li))
    np.save(f'./data/dataset/mit_18_valid_label_{window_size}.npy', np.array(valid_label_li))
    
    return full_window_li, full_label_li, no_window_li, no_label_li, valid_window_li, valid_label_li


def get_ethz(window_size, args):
    """
    ETHZ Seizure Corpus
    """
    print('=========ETHZ train set=========')
    full_window_li = []
    full_label_li = []
    no_window_li = []
    no_label_li = []
    valid_window_li = []
    valid_label_li = []

    file_list, label_list = get_file('./data/v2.0.3/edf/train')
    progress = tqdm(range(len(file_list)))
    for i in progress:
        data_file, label_file = file_list[i], label_list[i]
        try:
            data, detect_label = get_data_18(data_file, label_file)
        except:
            continue
        if data.shape[1] < window_size:
            continue
        dataloader = get_dataloader(data, detect_label, window_size=window_size)
        for batch in dataloader:
            X = batch['X']
            y_detect = batch['y_detect']

            X = X.detach().numpy()
            y_detect = y_detect.detach().numpy()
            # print('x:', X.shape)
            # print('y_detect:', y_detect.shape)

            for mini_batch in range(X.shape[0]):
                seizure_sample_cnt = np.sum(y_detect[mini_batch, :])
                temp = X[mini_batch]
                # print('temp', temp.shape)
                if seizure_sample_cnt == 0:
                    no_window_li.append(temp)
                    no_label_li.append(y_detect[mini_batch, :])
                elif seizure_sample_cnt == window_size:
                    full_window_li.append(temp)
                    full_label_li.append(y_detect[mini_batch, :])
                else:
                    valid_window_li.append(temp)
                    valid_label_li.append(y_detect[mini_batch, :])
            progress.set_description(f'ps: {len(valid_window_li)}, fs: {len(full_window_li)}, ns: {len(no_window_li)}')
    print('full_window:', len(full_window_li))
    print('no_window:', len(no_window_li))
    print('valid_window:', len(valid_window_li))

    full_window_li, full_label_li = shuffle(full_window_li, full_label_li)
    no_window_li, no_label_li = shuffle(no_window_li, no_label_li)
    
    no_window_li = no_window_li[:int(len(valid_window_li) * 3)]
    no_label_li = no_label_li[:int(len(valid_label_li) * 3)]

    valid_window_li, valid_label_li = shuffle(valid_window_li, valid_label_li)

    np.save(f'./data/dataset/ethz_18_full_window_{window_size}.npy', np.array(full_window_li))
    np.save(f'./data/dataset/ethz_18_full_label_{window_size}.npy', np.array(full_label_li))
    np.save(f'./data/dataset/ethz_18_no_window_{window_size}.npy', np.array(no_window_li))
    np.save(f'./data/dataset/ethz_18_no_label_{window_size}.npy', np.array(no_label_li))
    np.save(f'./data/dataset/ethz_18_valid_window_{window_size}.npy', np.array(valid_window_li))
    np.save(f'./data/dataset/ethz_18_valid_label_{window_size}.npy', np.array(valid_label_li))

    return full_window_li, full_label_li, no_window_li, no_label_li, valid_window_li, valid_label_li


def get_train(
        full_window_li,
        full_label_li,
        no_window_li,
        no_label_li,
        valid_window_li,
        valid_label_li,
        alpha,
        beta,
        window_size,
):
    print('valid_window:', len(valid_window_li))
    print('full_window:', len(full_window_li))
    print('no_window:', len(no_window_li))
    full_window_li = full_window_li[:int(len(valid_window_li) * alpha)]
    full_label_li = full_label_li[:int(len(valid_label_li) * alpha)]
    no_window_li = no_window_li[:int(len(valid_window_li) * beta)]
    no_label_li = no_label_li[:int(len(valid_label_li) * beta)]

    print('-' * 5, 'after filter', '-' * 5)
    print('full_window:', len(full_window_li))
    print('no_window:', len(no_window_li))

    valid_window_li = valid_window_li + full_window_li + no_window_li
    valid_label_li = valid_label_li + full_label_li + no_label_li
    valid_window_li, valid_label_li = shuffle(valid_window_li, valid_label_li)
    valid_window_li, valid_label_li = np.array(valid_window_li), np.array(valid_label_li)

    print()
    print('valid_window:', len(valid_window_li))
    print('valid_label:', len(valid_label_li))
    np.save(f'./data/dataset/full_train_data_{alpha}_{beta}_{window_size}.npy', valid_window_li)
    np.save(f'./data/dataset/full_train_label_{alpha}_{beta}_{window_size}.npy', valid_label_li)


if __name__ == '__main__':
    args = parse_args()
    window_size = args.window_size
    alpha = args.alpha
    beta = args.beta

    full_window_li, full_label_li, no_window_li, no_label_li, valid_window_li, valid_label_li = get_siena(window_size, args)

    # temp_full_window_li, temp_full_label_li, temp_no_window_li, temp_no_label_li, temp_valid_window_li, temp_valid_label_li = get_mit(window_size, args)
    # full_window_li = full_window_li + temp_full_window_li
    # full_label_li = full_label_li + temp_full_label_li
    # no_window_li = no_window_li + temp_no_window_li
    # no_label_li = no_label_li + temp_no_label_li
    # valid_window_li = valid_window_li + temp_valid_window_li
    # valid_label_li = valid_label_li + temp_valid_label_li

    temp_full_window_li, temp_full_label_li, temp_no_window_li, temp_no_label_li, temp_valid_window_li, temp_valid_label_li = get_ethz(window_size, args)
    full_window_li = full_window_li + temp_full_window_li
    full_label_li = full_label_li + temp_full_label_li
    no_window_li = no_window_li + temp_no_window_li
    no_label_li = no_label_li + temp_no_label_li
    valid_window_li = valid_window_li + temp_valid_window_li
    valid_label_li = valid_label_li + temp_valid_label_li

    full_window_li, full_label_li = shuffle(full_window_li, full_label_li)
    no_window_li, no_label_li = shuffle(no_window_li, no_label_li)
    valid_window_li, valid_label_li = shuffle(valid_window_li, valid_label_li)
    

    full_window_li = full_window_li[:int(len(valid_window_li) * alpha)]
    full_label_li = full_label_li[:int(len(valid_label_li) * alpha)]
    no_window_li = no_window_li[:int(len(valid_window_li) * beta)]
    no_label_li = no_label_li[:int(len(valid_label_li) * beta)]

    print('-' * 5, 'after filter', '-' * 5)
    print('full_window:', len(full_window_li))
    print('no_window:', len(no_window_li))

    valid_window_li = valid_window_li + full_window_li + no_window_li
    valid_label_li = valid_label_li + full_label_li + no_label_li
    valid_window_li, valid_label_li = shuffle(valid_window_li, valid_label_li)
    valid_window_li, valid_label_li = np.array(valid_window_li), np.array(valid_label_li)
    print()
    print('valid_window:', len(valid_window_li))
    print('valid_label:', len(valid_label_li))
    np.save(f'./data/dataset/full_train_data_{alpha}_{beta}_{window_size}.npy', valid_window_li)
    np.save(f'./data/dataset/full_train_label_{alpha}_{beta}_{window_size}.npy', valid_label_li)