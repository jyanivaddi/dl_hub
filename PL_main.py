from __future__ import print_function
import os
import sys
from typing import Any
sys.path.append("../dl_hub")
sys.path.append("../ERA_V1/session_12")
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.augmentations as AA
from albumentations.pytorch import ToTensorV2
from dataloaders.pl_custom_cifar10_datamodule import CustomCifar10DataModule
from models.pl_custom_resnet import LitResnet
from utils.helper_utils import find_best_lr
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar


def _define_optimizer(params, model):
    optimizer_type = params['optimizer_type']
    optimizer_params = params['optimizer_params']
    if optimizer_type.upper() == 'ADAM':
        if 'weight_decay' in optimizer_params:
            weight_decay = optimizer_params['weight_decay']
        else:
            weight_decay =  1e-4
        optimizer = optim.Adam(model.parameters(), 
                               lr = optimizer_params['lr'], 
                               weight_decay=weight_decay)
    elif optimizer_type.upper() == 'SGD':
        if 'weight_decay' in optimizer_params:
            weight_decay = optimizer_params['weight_decay']
        else:
            weight_decay =  5e-4
        if 'momentum' in optimizer_params:
            momentum = optimizer_params['momentum']
        else:
            momentum = 0.9
        optimizer = optim.SGD(model.parameters(),
                              lr=optimizer_params['lr'],
                              momentum=momentum,
                              weight_decay=weight_decay)
    return optimizer


def _define_scheduler_dict(params, optimizer):
    scheduler_type = params['scheduler_type']
    scheduler_params = params['scheduler_params']
    if scheduler_type.upper() == 'ONECYCLELR':
        scheduler = OneCycleLR(
            optimizer, 
            max_lr = scheduler_params['max_lr'], 
            steps_per_epoch = scheduler_params['num_steps_per_epoch'],
            epochs = params['num_epochs'], 
            pct_start = scheduler_params['pct_start'],
            div_factor = scheduler_params['div_factor'], 
            three_phase = scheduler_params['three_phase'], 
            final_div_factor = scheduler_params['final_div_factor'], 
            anneal_strategy = scheduler_params['anneal_strategy'], 
            verbose=False)
        interval = 'step'
    elif scheduler_type.upper() == 'REDUCELRONPLATEAU':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=scheduler_params['mode'], 
            factor=scheduler_params['factor'], 
            patience=scheduler_params['patience'], 
            threshold=scheduler_params['threshold'],
            verbose=True)
        interval = 'epoch'
    else:
        return {}
    scheduler_dict = {"scheduler": scheduler,
                      "interval": interval}
    return scheduler_dict


def define_pl_model_resnet(params):
    """
    Define a Pytorch Lightning for Resnet model
    """
    base_lr = params['optimizer_params']['lr']
    base_channels = params['base_channels']
    num_classes = params['num_classes']
    drop_out_probability = params['drop_out_probability']
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    resnet_model = LitResnet(loss_func, lr=base_lr,base_channels=base_channels, num_classes=num_classes,
                    drop_out_probability=drop_out_probability, epochs=num_epochs)
    optimizer = _define_optimizer(params, resnet_model)
    scheduler_dict = _define_scheduler_dict(params, optimizer)
    resnet_model.set_optimizer(optimizer)
    resnet_model.set_scheduler_dict(scheduler_dict)
    return resnet_model


def train_and_eval_pl_model(params, resnet_model, data_module):
    """
    Define a trainer for the lightning model. Then run training and test on it
    """
    save_dir = params['save_dir']
    steps_per_epoch = len(data_module.train_dataset) // params['batch_size']
    trainer = Trainer(
        max_epochs=resnet_model.epochs,
        max_steps = steps_per_epoch*resnet_model.epochs,
        accelerator='auto',
        devices = 1 if torch.cuda.is_available() else None,
        logger = CSVLogger(save_dir=save_dir),
        log_every_n_steps = 1,
        callbacks=[LearningRateMonitor(logging_interval='step'),
                TQDMProgressBar(refresh_rate=10)],
        num_sanity_val_steps=0)

    trainer.fit(resnet_model, datamodule=data_module)
    trainer.test(resnet_model, datamodule=data_module)
    return trainer


def build_data_module(params):
    """
    Build a custom data module for CIFAR 10 using user defined transforms
    """
    data_dir = params['data_dir']
    train_transforms = params['train_transforms']
    val_transforms = params['val_transforms']
    test_transforms = params['test_transforms']
    val_split = params['val_split']
    batch_size = params['batch_size']
    cifar10_dm = CustomCifar10DataModule(
        data_dir,
        train_transforms,
        val_transforms,
        test_transforms,
        batch_size,
        val_split = val_split
    )
    cifar10_dm.prepare_data()
    cifar10_dm.setup()
    return cifar10_dm


def get_max_lr(params, cifar10_dm):
    base_lr = params['optimizer_params']['lr']
    base_channels = params['base_channels']
    num_classes = params['num_classes']
    drop_out_probability = params['drop_out_probability']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    num_workers = params['num_workers']
    loss_func = params['loss_func']
    device = params['device']
    dummy_model = LitResnet(loss_func, lr=base_lr,base_channels=base_channels, num_classes=num_classes,
                    drop_out_probability=drop_out_probability, epochs=num_epochs)
    optimizer =_define_optimizer(params, dummy_model)
    tl = DataLoader(cifar10_dm.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    _, best_lr = find_best_lr(dummy_model, tl, optimizer, loss_func, device)
    return best_lr
