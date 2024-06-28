#!/usr/bin/env python
# coding: utf-8

# (lightning_mnist_example)=
# 
# # 训练 Pytorch Lightning 图像分类器
# 
# 此示例介绍如何使用 Ray Train 训练 Pytorch Lightning 模块 {class}`TorchTrainer <ray.train.torch.TorchTrainer>`。它演示了如何使用分布式数据并行在 MNIST 数据集上训练基本神经网络。
# 

# In[1]:


get_ipython().system('pip install "torchmetrics>=0.9" "pytorch_lightning>=1.6"')


# In[2]:


import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from filelock import FileLock
from torch.utils.data import DataLoader, random_split, Subset
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import trainer
from pytorch_lightning.loggers.csv_logs import CSVLogger


# ## 准备数据集和模块
# 
# Pytorch Lightning Trainer 接受 `torch.utils.data.DataLoader` 或 `pl.LightningDataModule` 作为数据输入。您可以继续使用它们，而无需对 Ray Train 进行任何更改。

# In[4]:


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=100):
        super().__init__()
        self.data_dir = os.getcwd()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            mnist = MNIST(
                self.data_dir, train=True, download=True, transform=self.transform
            )

            # Split data into train and val sets
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        with FileLock(f"{self.data_dir}.lock"):
            self.mnist_test = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)


# 接下来，定义一个简单的多层感知作为的子类 `pl.LightningModule`。

# In[5]:


class MNISTClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, feature_dim=128):
        torch.manual_seed(421)
        super(MNISTClassifier, self).__init__()
        self.save_hyperparameters()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 10),
            nn.ReLU(),
        )
        self.lr = lr
        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.eval_loss = []
        self.eval_accuracy = []
        self.test_accuracy = []
        pl.seed_everything(888)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.linear_relu_stack(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, acc = self._shared_eval(val_batch)
        self.log("val_accuracy", acc)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(acc)
        return {"val_loss": loss, "val_accuracy": acc}

    def test_step(self, test_batch, batch_idx):
        loss, acc = self._shared_eval(test_batch)
        self.test_accuracy.append(acc)
        self.log("test_accuracy", acc, sync_dist=True, on_epoch=True)
        return {"test_loss": loss, "test_accuracy": acc}

    def _shared_eval(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        return loss, acc

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# 您不需要修改 PyTorch Lightning 模型或数据模块的定义。

# ## 定义训练函数
# 
# 此代码为每个 worker 定义了一个 {ref}`训练函数 <train-overview-training-function>` 。将训练函数与原始 PyTorch Lightning 代码进行比较，请注意三个主要区别：
# 
# - 分布式策略: 使用 {class}`RayDDPStrategy <ray.train.lightning.RayDDPStrategy>`。
# - 集群环境: 使用 {class}`RayLightningEnvironment <ray.train.lightning.RayLightningEnvironment>`。
# - 并发设备: ``TorchTrainer`` 配置始终设置为 `devices="auto"` 来使用所有可用设备。
# 
# 参考 {ref}`PyTorch Lightning 入门 <train-pytorch-lightning>` 获取更多信息。
# 
# 
# 对于检查点报告，Ray Train 提供了一个最小 {class}`RayTrainReportCallback <ray.train.lightning.RayTrainReportCallback>` 类，用于在每个训练时期结束时报告指标和检查点。对于更复杂的检查点逻辑，请实现自定义回调。请参阅 {ref}`保存和加载检查点 <train-checkpointing>`。

# In[6]:


use_gpu = True # Set to False if you want to run without GPUs
num_workers = 4


# In[7]:


import pytorch_lightning as pl
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

def train_func_per_worker():
    model = MNISTClassifier(lr=1e-3, feature_dim=128)
    datamodule = MNISTDataModule(batch_size=128)

    trainer = pl.Trainer(
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        max_epochs=10,
        accelerator="gpu" if use_gpu else "cpu",
        log_every_n_steps=100,
        logger=CSVLogger("logs"),
    )
    
    trainer = prepare_trainer(trainer)
    
    # Train model
    trainer.fit(model, datamodule=datamodule)

    # Evaluation on the test dataset
    trainer.test(model, datamodule=datamodule)


# 现在把所有内容放在一起：

# In[8]:


scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

run_config = RunConfig(
    name="ptl-mnist-example",
    storage_path="/tmp/ray_results",
    checkpoint_config=CheckpointConfig(
        num_to_keep=3,
        checkpoint_score_attribute="val_accuracy",
        checkpoint_score_order="max",
    ),
)

trainer = TorchTrainer(
    train_func_per_worker,
    scaling_config=scaling_config,
    run_config=run_config,
)


# 现在开始你的训练：

# In[9]:


result = trainer.fit()


# ## 检查训练结果和检查点

# In[10]:


result


# In[11]:


print("Validation Accuracy: ", result.metrics["val_accuracy"])
print("Trial Directory: ", result.path)
print(sorted(os.listdir(result.path)))


# Ray Train 在试验目录中保存了三个检查点（`checkpoint_000007`, `checkpoint_000008`, `checkpoint_000009`）。以下代码从拟合结果中检索最新的检查点并将其重新加载到模型中。
# 
# 如果丢失了内存中的结果对象，您可以从检查点文件恢复模型。检查点路径为： `/tmp/ray_results/ptl-mnist-example/TorchTrainer_eb925_00000_0_2023-08-07_23-15-06/checkpoint_000009/checkpoint.ckpt`。

# In[12]:


checkpoint = result.checkpoint

with checkpoint.as_directory() as ckpt_dir:
    best_model = MNISTClassifier.load_from_checkpoint(f"{ckpt_dir}/checkpoint.ckpt")

best_model


# ## 也可以看看
# 
# * {ref}`PyTorch Lightning 入门 <train-pytorch-lightning>` ，有关使用 Ray Train 和 PyTorch Lightning 的教程
# 
# * 更多用例的 {ref}`Ray Train 示例 <train-examples>`
