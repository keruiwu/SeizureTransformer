from model import SeizureTransformer
import numpy as np
import torch
import torch.nn as nn

import argparse
from tqdm import tqdm
import os

from timescoring import scoring
from timescoring.annotations import Annotation
from service.handle_data import *
from service.post_process import *
from service.result import Result, get_testingdataloader
from train_sd import ParseArgs


def eval_TestSet(args, device):
        model.eval()

        avg_sample_score = Result()
        avg_event_score = Result()

        file_list, label_list = get_file('./data/v2.0.3/edf/eval')
        progress = tqdm(range(len(file_list)))
        for i in progress:
            data_file, label_file = file_list[i], label_list[i]
            try:
                data, ref = get_data_18(data_file, label_file)
            except:
                continue
            if data.shape[1] < args.window_size:
                continue

            # print('data', data.shape)
            # print('ref', ref.shape)
            
            dataloader = get_testingdataloader(data, window_size=args.window_size, batch_size=64)

            total_output = None  # shape (batch_size, seq_len)
            for data in dataloader:
                data = data.float().to(device)
                # print('data', data.shape)
                output = model(data)
                # print('output', output[0].shape)
                
                if total_output is None:
                    total_output = output.detach().cpu()
                else:
                    total_output = torch.cat((total_output, output.detach().cpu()), 0)

            y_predict = total_output.numpy()
            y_predict = y_predict.flatten()[:ref.shape[0]]
            # print('y_predict', y_predict.shape)

            binary_output = (y_predict > args.threshold).astype(int)
            binary_output = morphological_filter_1d(binary_output, operation="opening", kernel_size=5)
            binary_output = morphological_filter_1d(binary_output, operation="closing", kernel_size=5)
            binary_output = remove_short_events(binary_output, min_length=2.0, fs=256)
            hyp = Annotation(binary_output, 256)
            ref = Annotation(ref.detach().numpy(), 256)

            sample_score = scoring.SampleScoring(ref, hyp)
            event_score = scoring.EventScoring(ref, hyp)

            avg_sample_score += Result(sample_score)
            avg_event_score += Result(event_score)

        avg_sample_score.computeScores()
        avg_event_score.computeScores()

        print('=' * 10, 'Sample', '=' * 10)
        print('f1', avg_sample_score.f1)
        print('sensitivity', avg_sample_score.sensitivity)
        print('precision', avg_sample_score.precision)
        print('fpRate', avg_sample_score.fpRate)
        print('=' * 10, 'Event', '=' * 10)
        print('f1', avg_event_score.f1)
        print('sensitivity', avg_event_score.sensitivity)
        print('precision', avg_event_score.precision)
        print('fpRate', avg_event_score.fpRate)


if __name__ == '__main__':
    args = ParseArgs()
    restore_path = f'./ckp/full_18_beta{args.beta}_alpha{args.alpha}_window{args.window_size}_dim{args.dim_feedforward}_layer{args.num_layers}_num_head{args.num_heads}_f1.pth'
    model = SeizureTransformer(
        in_channels=args.num_channel,
        in_samples=args.window_size,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model.load_state_dict(torch.load(restore_path))
    model = model.to(f'cuda:{args.device_ids[0]}')

    eval_TestSet(args, f'cuda:{args.device_ids[0]}')