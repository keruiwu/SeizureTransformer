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


def ParseArgs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--window_size', type=int, default=15360,
                        help='input sequence length of the model')
    parser.add_argument('--num_channel', type=int, default=18, 
                        help='number of channels')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha value for training data')
    parser.add_argument('--beta', type=float, default=2.0, help='Beta value for training data')
    parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for binary classification')
    
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='dim_feedforward')
    parser.add_argument('--num_layers', type=int, default=8, help='num_layers')
    parser.add_argument('--num_heads', type=int, default=4, help='num_heads')
    parser.add_argument('--epochs', type=int, default=100,
                        help='train epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--weight_decay', type=float, default=2e-5)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    args = parser.parse_args()
    args.task_name = 'classification'
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    return args


class TrainingDataset(nn.Module):
    def __init__(self, data, label):
        super(TrainingDataset, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.label[idx])

def get_trainingloader(data, label, batch_size=128):
    dataset = TrainingDataset(data, label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


def main():
    def eval_TestSet(args, device):
        model.eval()

        avg_sample_score = Result()
        avg_event_score = Result()

        file_list, label_list = get_file('./data/v2.0.3/edf/dev')
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

        return avg_event_score.f1
    args = ParseArgs()
    data = np.load(f'./data/dataset/full_train_data_{args.alpha}_{args.beta}_{args.window_size}.npy')
    label = np.load(f'./data/dataset/full_train_label_{args.alpha}_{args.beta}_{args.window_size}.npy')
    
    print('data', data.shape)
    print('label', label.shape)

    dataloader = get_trainingloader(data, label, batch_size=86)
    model = SeizureTransformer(
        in_channels=args.num_channel,
        in_samples=args.window_size,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    
    # model = nn.DataParallel(model, device_ids=args.device_ids)
    model = model.to(f'cuda:{args.device_ids[0]}')

    loss_fn = nn.functional.binary_cross_entropy
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=11, gamma=0.5)
    best_f1 = 0
    for epoch in range(args.epochs):
        print('='* 10, 'epoch', epoch, '=' * 10)
        print('-'*5, 'train', '-' * 5)
        model.train()
        progress = tqdm(dataloader, total=len(dataloader))

        for X, y_detect in progress:
            # print('X', X.shape)
            # print('y_detect', y_detect.shape)
            X = X.to(f'cuda:{args.device_ids[0]}')
            y_detect = y_detect.to(f'cuda:{args.device_ids[0]}')
            
            output = model(X)
            # print('detect', output[0].shape)
            detect_loss = loss_fn(output, y_detect)
            # print('detect_loss:', detect_loss.detach().cpu().item())
            progress.set_description(f'detect_loss: {detect_loss.detach().item():.4f}')
            
            optimizer.zero_grad()
            detect_loss.backward()
            optimizer.step()
        # scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Learning rate after epoch {epoch}: {current_lr:.6f}")
        print('-'*5, 'eval', '-' * 5)
        event_f1 = eval_TestSet(args, device=f'cuda:{args.device_ids[0]}')
        if event_f1 > best_f1:
            best_f1 = event_f1
            print('store best model with f1: ', best_f1)
            # store
            model_path = f'./ckp/'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            # model_name = model_path + f'0331_seizuretransformer_fully_res_elu_seq15360_head4_f512_layer8_ff2048_{args.lr}_{args.weight_decay}_{args.epochs}.pth'
            model_name = model_path + f'full_18_beta{args.beta}_alpha{args.alpha}_window{args.window_size}_dim{args.dim_feedforward}_layer{args.num_layers}_num_head{args.num_heads}_f1.pth'

            if os.path.exists(model_name):
                os.remove(model_name)
            torch.save(model.state_dict(), model_name)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()