import os
#set the environment variable to use the GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import yaml
import json
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from dataset.cxr_datamodule import CxrDataModule
from model.cxr_model import CxrModel

torch.set_float32_matmul_precision('high')

def train():
    def load_yaml(file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    file_path = 'config_custom.yaml'
    config_dict = load_yaml(file_path)

    datamodule_cfg = config_dict['data']['datamodule_cfg']
    dataloader_init_args = config_dict['data']['dataloader_init_args']
    data_module = CxrDataModule(datamodule_cfg, dataloader_init_args)

    # def __init__(self, lr, classes, loss_init_args, timm_init_args):
    lr = float(config_dict['model']['lr'])
    classes = config_dict['model']['classes']
    loss_init_args = config_dict['model']['loss_init_args']
    timm_init_args = config_dict['model']['timm_init_args']
    model = CxrModel(lr, classes, loss_init_args, timm_init_args)

    max_epochs = config_dict['trainer']['max_epochs']
    logger = TensorBoardLogger("lightning_logs", name="cheXmodel")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='output/checkpoints',
        filename='chexmodel-{epoch:02d}-{val_loss:.5f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, data_module)

if __name__ == '__main__':
    train()

