#!/usr/bin/env python
# coding: utf-8

# # Using PyTorch Lightning with Tune
# 
# (tune-pytorch-lightning-ref)=
# 
# PyTorch Lightning is a framework which brings structure into training PyTorch models. It aims to avoid boilerplate code, so you don't have to write the same training loops all over again when building a new model.
# 
# ```{image} /images/pytorch_lightning_full.png
# :align: center
# ```
# 
# The main abstraction of PyTorch Lightning is the `LightningModule` class, which should be extended by your application. There is [a great post on how to transfer your models from vanilla PyTorch to Lightning](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09).
# 
# The class structure of PyTorch Lightning makes it very easy to define and tune model parameters. This tutorial will show you how to use Tune with Ray Train's {class}`TorchTrainer <ray.train.torch.TorchTrainer>` to find the best set of parameters for your application on the example of training a MNIST classifier. Notably, the `LightningModule` does not have to be altered at all for this - so you can use it plug and play for your existing models, assuming their parameters are configurable!
# 
# :::{note}
# To run this example, you will need to install the following:
# 
# ```bash
# $ pip install "ray[tune]" torch torchvision pytorch_lightning
# ```
# :::
# 
# ```{contents}
# :backlinks: none
# :local: true
# ```
# 
# ## PyTorch Lightning classifier for MNIST
# 
# Let's first start with the basic PyTorch Lightning implementation of an MNIST classifier. This classifier does not include any tuning code at this point.
# 
# First, we run some imports:

# In[ ]:


import os
import torch
import tempfile
import pytorch_lightning as pl
import torch.nn.functional as F
from filelock import FileLock
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


# In[2]:


# If you want to run full test, please set SMOKE_TEST to False
SMOKE_TEST = True


# Our example builds on the MNIST example from the [blog post](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09) we mentioned before. We adapted the original model and dataset definitions into `MNISTClassifier` and `MNISTDataModule`. 

# In[3]:


class MNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super(MNISTClassifier, self).__init__()
        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)
        self.eval_loss = []
        self.eval_accuracy = []

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.data_dir = tempfile.mkdtemp()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            mnist = MNIST(
                self.data_dir, train=True, download=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])

            self.mnist_test = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)


# In[4]:


default_config = {
    "layer_1_size": 128,
    "layer_2_size": 256,
    "lr": 1e-3,
}


# Define a training function that creates model, datamodule, and lightning trainer with Ray Train utilities.

# In[ ]:


from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)


def train_func(config):
    dm = MNISTDataModule(batch_size=config["batch_size"])
    model = MNISTClassifier(config)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)


# ## Tuning the model parameters
# 
# The parameters above should give you a good accuracy of over 90% already. However, we might improve on this simply by changing some of the hyperparameters. For instance, maybe we get an even higher accuracy if we used a smaller learning rate and larger middle layer size.
# 
# Instead of manually loop through all the parameter combinitions, let's use Tune to systematically try out parameter combinations and find the best performing set.
# 
# First, we need some additional imports:

# In[21]:


from ray import tune
from ray.tune.schedulers import ASHAScheduler


# ### Configuring the search space
# 
# Now we configure the parameter search space. We would like to choose between different layer dimensions, learning rate, and batch sizes. The learning rate should be sampled uniformly between `0.0001` and `0.1`. The `tune.loguniform()` function is syntactic sugar to make sampling between these different orders of magnitude easier, specifically we are able to also sample small values. Similarly for `tune.choice()`, which samples from all the provided options.

# In[22]:


search_space = {
    "layer_1_size": tune.choice([32, 64, 128]),
    "layer_2_size": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64]),
}


# ### Selecting a scheduler
# 
# In this example, we use an [Asynchronous Hyperband](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/)
# scheduler. This scheduler decides at each iteration which trials are likely to perform
# badly, and stops these trials. This way we don't waste any resources on bad hyperparameter
# configurations.

# In[24]:


# The maximum training epochs
num_epochs = 5

# Number of sampls from parameter space
num_samples = 10


# If you have more resources available, you can modify the above parameters accordingly. e.g. more epochs, more parameter samples.

# In[9]:


if SMOKE_TEST:
    num_epochs = 3
    num_samples = 3


# In[25]:


scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)


# ### Training with GPUs
# 
# We can specify the number of resources, including GPUs, that Tune should request for each trial.
# 
# `TorchTrainer` takes care of environment setup for Distributed Data Parallel training, the model and data will automatically get distributed across GPUs. You only need to set the number of GPUs per worker in `ScalingConfig` and also set `accelerator="auto"` in your training function.

# In[26]:


from ray.train import RunConfig, ScalingConfig, CheckpointConfig

scaling_config = ScalingConfig(
    num_workers=3, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="ptl/val_accuracy",
        checkpoint_score_order="max",
    ),
)


# In[13]:


if SMOKE_TEST:
    scaling_config = ScalingConfig(
        num_workers=3, use_gpu=False, resources_per_worker={"CPU": 1}
    )


# In[27]:


from ray.train.torch import TorchTrainer

# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)


# ### Putting it together
# 
# Lastly, we need to create a `Tuner()` object and start Ray Tune with `tuner.fit()`. The full code looks like this:

# In[28]:


def tune_mnist_asha(num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


results = tune_mnist_asha(num_samples=num_samples)


# In[29]:


results.get_best_result(metric="ptl/val_accuracy", mode="max")


# In the example above, Tune runs 10 trials with different hyperparameter configurations.
# 
# As you can see in the `training_iteration` column, trials with a high loss (and low accuracy) have been terminated early. The best performing trial used
# `batch_size=64`, `layer_1_size=128`, `layer_2_size=256`, and `lr=0.0037`.

# ## More PyTorch Lightning Examples
# 
# - {ref}`[Basic] Train a PyTorch Lightning Image Classifier with Ray Train <lightning_mnist_example>`.
# - {ref}`[Intermediate] Fine-tune a BERT Text Classifier with PyTorch Lightning and Ray Train <lightning_advanced_example>`
# - {ref}`[Advanced] Fine-tune dolly-v2-7b with PyTorch Lightning and FSDP <dolly_lightning_fsdp_finetuning>`
# - {doc}`/tune/examples/includes/mlflow_ptl_example`: Example for using [MLflow](https://github.com/mlflow/mlflow/)
#   and [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) with Ray Tune.
# 
