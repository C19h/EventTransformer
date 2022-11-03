# author:c19h
# datetime:2022/10/31 20:10
import torch
from torch import nn
from argparse import ArgumentParser

import torchvision
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics import Accuracy


# ================================================================================
# 一，准备数据
# ================================================================================

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./minist/",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        transform = T.Compose([T.ToTensor()])
        self.ds_test = MNIST(self.data_dir, download=True, train=False, transform=transform)
        self.ds_predict = MNIST(self.data_dir, download=True, train=False, transform=transform)
        ds_full = MNIST(self.data_dir, download=True, train=True, transform=transform)
        self.ds_train, self.ds_val = random_split(ds_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.ds_predict, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    @staticmethod
    def add_dataset_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--pin_memory', type=bool, default=True)
        return parser


# ================================================================================
# 二，定义模型
# ================================================================================

net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout2d(p=0.1),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)


class Model(pl.LightningModule):

    def __init__(self, net, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        x = self.net(x)
        return x

    # 定义loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        return {"loss": loss, "preds": preds.detach(), "y": y.detach()}

    # 定义各种metrics
    def training_step_end(self, outputs):
        train_acc = self.train_acc(outputs['preds'], outputs['y']).item()
        self.log("train_acc", train_acc, prog_bar=True)
        return {"loss": outputs["loss"].mean()}

    # 定义optimizer,以及可选的lr_scheduler
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        return {"loss": loss, "preds": preds.detach(), "y": y.detach()}

    def validation_step_end(self, outputs):
        val_acc = self.val_acc(outputs['preds'], outputs['y']).item()
        self.log("val_loss", outputs["loss"].mean(), on_epoch=True, on_step=False)
        self.log("val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        return {"loss": loss, "preds": preds.detach(), "y": y.detach()}

    def test_step_end(self, outputs):
        test_acc = self.test_acc(outputs['preds'], outputs['y']).item()
        self.log("test_acc", test_acc, on_epoch=True, on_step=False)
        self.log("test_loss", outputs["loss"].mean(), on_epoch=True, on_step=False)

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser


# ================================================================================
# 三，训练模型
# ================================================================================

def main(hparams):
    pl.seed_everything(1234)

    data_mnist = MNISTDataModule(batch_size=hparams.batch_size,
                                 num_workers=hparams.num_workers)

    model = Model(net, learning_rate=hparams.learning_rate)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min'
    )
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=3,
                                                mode='min')

    trainer = pl.Trainer.from_argparse_args(
        hparams,
        max_epochs=10,

        callbacks=[ckpt_callback, early_stopping]
    )

    if hparams.auto_scale_batch_size is not None:
        # 搜索不发生OOM的最大batch_size
        max_batch_size = trainer.tuner.scale_batch_size(model, data_mnist,
                                                        mode=hparams.auto_scale_batch_size)
        data_mnist.batch_size = max_batch_size

        # 等价于
        # trainer.tune(model,data_mnist)

    # gpus=0, #单CPU模式
    # gpus=1, #单GPU模式
    # num_processes=4,strategy="ddp_find_unused_parameters_false", #多CPU(进程)模式
    # gpus=4,strategy="dp", #多GPU(dp速度提升效果一般)
    # gpus=4,strategy=“ddp_find_unused_parameters_false" #多GPU(ddp速度提升效果好)

    trainer.fit(model, data_mnist)
    result = trainer.test(model, data_mnist, ckpt_path='best')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = MNISTDataModule.add_dataset_args(parser)
    parser = Model.add_model_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    main(hparams)