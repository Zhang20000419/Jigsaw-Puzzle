# 解决国内下载模型失败的问题
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from tools.cfg import py2cfg
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import numpy as np
import shutil


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--config', type=str, default='config/config_normal.py')
    return parser.parse_args()


def prepare(config, args):
    if not os.path.exists(config.checkpoint_save_path):
        os.makedirs(config.checkpoint_save_path)
    if not os.path.exists(config.config_save_path):
        os.makedirs(config.config_save_path)
    file_path = os.path.join(config.config_save_path, config.config_save_name + '.py')
    shutil.copy(args.config, file_path)


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.model = timm.create_model('resnet18', pretrained=True, num_classes=24)
        self.net = self.config.net

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.config.loss(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.config.loss(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def on_validation_epoch_end(self) -> None:
        # print('Epoch:{:02},mean val loss:{}'.format(self.current_epoch, self.trainer.callback_metrics['val_loss']))
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.config.num_labels, self.config.num_labels)
        # 计算准确率
        accuracy = (logits.argmax(dim=2) == y).sum().item() / (self.config.batch_size * self.config.num_labels)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {'test_accuracy': accuracy}

    def on_test_epoch_end(self) -> None:
        print('Epoch:{:02},mean test accuracy:{}'.format(self.current_epoch,
                                                         self.trainer.callback_metrics['test_accuracy']))

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        return optimizer


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config)
    model = LitModel(config)
    prepare(config, args)
    checkpoint_callback = ModelCheckpoint(monitor=config.monitor, dirpath=config.checkpoint_save_path,
                                          filename=config.checkpoint_filename, save_top_k=config.save_top_k,
                                          mode=config.mode, save_last=config.save_last)
    trainer = pl.Trainer(max_epochs=config.max_epochs, devices=config.gpus, callbacks=[checkpoint_callback],
                         check_val_every_n_epoch=config.check_val_every_n_epoch, strategy=config.strategy,
                         logger=config.logger)
    trainer.fit(model, train_dataloaders=config.dataloader_train, val_dataloaders=config.dataloader_valid)
    trainer.test(model, dataloaders=config.dataloader_test)


if __name__ == '__main__':
    main()

