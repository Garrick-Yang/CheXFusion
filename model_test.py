import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import yaml
import lightning.pytorch as pl
from dataset.cxr_datamodule import CxrDataModule
from model.cxr_model import CxrModel
from callbacks.fusion_submit_callback import FusionSubmissonWriter
from callbacks.submit_callback import SubmissonWriter

torch.set_float32_matmul_precision('high')

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def test():
    file_path = 'config_custom.yaml'
    config_dict = load_yaml(file_path)

    datamodule_cfg = config_dict['data']['datamodule_cfg']
    dataloader_init_args = config_dict['data']['dataloader_init_args']
    data_module = CxrDataModule(datamodule_cfg, dataloader_init_args)

    # Initialize the model
    lr = float(config_dict['model']['lr'])
    classes = config_dict['model']['classes']
    loss_init_args = config_dict['model']['loss_init_args']
    timm_init_args = config_dict['model']['timm_init_args']
    model = CxrModel(lr, classes, loss_init_args, timm_init_args)
    ckpt_path = config_dict['ckpt_path']

    writer_init_args = config_dict['writer']['init_args']
    writer = FusionSubmissonWriter(**writer_init_args)

    # trainer = pl.Trainer(callbacks=[writer])
    trainer = pl.Trainer(
        callbacks=[writer]
    )
    trainer.predict(model, data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    test()