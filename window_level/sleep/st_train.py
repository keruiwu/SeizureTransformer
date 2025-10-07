import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl
import math

import torchvision
import numpy as np
import random
import os 
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

from utils import temporal_interpolation
from utils_eval import get_metrics
from Modules.Transformers.pos_embed import create_1d_absolute_sin_cos_embedding
# from Modules.models.EEGPT_mcae import EEGTransformer
from st_model import SeizureTransformer_window

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint

use_channels_names = ['F3', 'F4', 'C3', 'C4', 'P3','P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ' ]

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self):
        super().__init__()    
        self.chans_num = len(use_channels_names)
        self.num_class = 5
        # init model
        self.target_encoder = SeizureTransformer_window(
            in_channels=2,
            in_samples=7680,
            num_classes=5,    
            dim_feedforward=2048,
        )
        self.target_encoder.requires_grad_(True)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
        
    def forward(self, x):
        x = temporal_interpolation(x, 256*30)
        x = self.target_encoder(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        # print('x', x.shape)
        # print('label', label.shape)
        
        logit = self.forward(x)
        # print('logit', logit.shape)
        # print('label', label.shape)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
        
        print('=' * 20)
        for key, value in results.items():
            print(f'{key}: {value}')
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        print('=' * 20)
        return super().on_validation_epoch_end()
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        
        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.target_encoder.parameters(),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        
# load configs
# -- LOSO 
# Train Data Num : 5Class: 36914 13604 34722 7922 15398
fold = 0
for fold in range(5):
    train_dataset = torchvision.datasets.DatasetFolder(root="../datasets/sleep_edf_full/TrainFold", loader=lambda x: torch.load(x),  extensions=['.pt'])
    valid_dataset = torchvision.datasets.DatasetFolder(root="../datasets/sleep_edf_full/TestFold", loader=lambda x: torch.load(x), extensions=[f'.pt'])

    # -- begin Training ------------------------------


    global max_epochs
    global steps_per_epoch
    global max_lr

    batch_size=256

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    max_epochs = 200
    steps_per_epoch = math.ceil(len(train_loader))
    max_lr = 4e-4

    # init model
    model = LitEEGPTCausal()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]

    trainer = pl.Trainer(accelerator='cuda',
                        devices=1,
                        precision=32,
                        max_epochs=max_epochs, 
                        callbacks=callbacks,
                        logger=[pl_loggers.TensorBoardLogger('./logs/', name="ST_SLEEPEDF_tb", version=f"fold{fold+1}"), 
                                pl_loggers.CSVLogger('./logs/', name="ST_SLEEPEDF_csv")])

    trainer.fit(model, train_loader, valid_loader)