#!/usr/bin/env python
# coding: utf-8

# # Using PyTorch Lightning with Tune
# 
# (tune-vanilla-pytorch-lightning-ref)=
# 
# PyTorch Lightning is a framework which brings structure into training PyTorch models. It
# aims to avoid boilerplate code, so you don't have to write the same training
# loops all over again when building a new model.
# 
# ```{image} /images/pytorch_lightning_full.png
# :align: center
# ```
# 
# The main abstraction of PyTorch Lightning is the `LightningModule` class, which
# should be extended by your application. There is [a great post on how to transfer your models from vanilla PyTorch to Lightning](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09).
# 
# The class structure of PyTorch Lightning makes it very easy to define and tune model
# parameters. This tutorial will show you how to use Tune to find the best set of
# parameters for your application on the example of training a MNIST classifier. Notably,
# the `LightningModule` does not have to be altered at all for this - so you can
# use it plug and play for your existing models, assuming their parameters are configurable!
# 
# :::{note}
# To run this example, you will need to install the following:
# 
# ```bash
# $ pip install "ray[tune]" torch torchvision pytorch-lightning
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
# Let's first start with the basic PyTorch Lightning implementation of an MNIST classifier.
# This classifier does not include any tuning code at this point.
# 
# Our example builds on the MNIST example from the [blog post we talked about
# earlier](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09).
# 
# First, we run some imports:

# In[1]:


import math

import torch
import pytorch_lightning as pl
from filelock import FileLock
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import os


# And then there is the Lightning model adapted from the blog post.
# Note that we left out the test set validation and made the model parameters
# configurable through a `config` dict that is passed on initialization.
# Also, we specify a `data_dir` where the MNIST data will be stored. Note that
# we use a `FileLock` for downloading data so that the dataset is only downloaded
# once per node.
# Lastly, we added a new metric, the validation accuracy, to the logs.

# In[2]:


class LightningMNISTClassifier(pl.LightningModule):
    """
    This has been adapted from
    https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    """

    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)

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

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

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
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    @staticmethod
    def download_data(data_dir):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        with FileLock(os.path.expanduser("~/.data.lock")):
            return MNIST(data_dir, train=True, download=True, transform=transform)

    def prepare_data(self):
        mnist_train = self.download_data(self.data_dir)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=int(self.batch_size))

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=int(self.batch_size))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_mnist(config):
    model = LightningMNISTClassifier(config)
    trainer = pl.Trainer(max_epochs=10, enable_progress_bar=False)

    trainer.fit(model)


# And that's it! You can now run `train_mnist(config)` to train the classifier, e.g.
# like so:

# In[3]:


def train_mnist_no_tune():
    config = {"layer_1_size": 128, "layer_2_size": 256, "lr": 1e-3, "batch_size": 64}
    train_mnist(config)


# ## Tuning the model parameters
# 
# The parameters above should give you a good accuracy of over 90% already. However,
# we might improve on this simply by changing some of the hyperparameters. For instance,
# maybe we get an even higher accuracy if we used a larger batch size.
# 
# Instead of guessing the parameter values, let's use Tune to systematically try out
# parameter combinations and find the best performing set.
# 
# First, we need some additional imports:

# In[4]:


from pytorch_lightning.loggers import TensorBoardLogger
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)


# ### Talking to Tune with a PyTorch Lightning callback
# 
# PyTorch Lightning introduced [Callbacks](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html)
# that can be used to plug custom functions into the training loop. This way the original
# `LightningModule` does not have to be altered at all. Also, we could use the same
# callback for multiple modules.
# 
# Ray Tune comes with ready-to-use PyTorch Lightning callbacks. To report metrics
# back to Tune after each validation epoch, we will use the `TuneReportCallback`:

# In[5]:


TuneReportCallback(
    {"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"}, on="validation_end"
)


# This callback will take the `val_loss` and `val_accuracy` values
# from the PyTorch Lightning trainer and report them to Tune as the `loss`
# and `mean_accuracy`, respectively.
# 
# ### Adding the Tune training function
# 
# Then we specify our training function. Note that we added the `data_dir` as a
# parameter here to avoid
# that each training run downloads the full MNIST dataset. Instead, we want to access
# a shared data location.
# 
# We are also able to specify the number of epochs to train each model, and the number
# of GPUs we want to use for training. We also create a TensorBoard logger that writes
# logfiles directly into Tune's root trial directory - if we didn't do that PyTorch
# Lightning would create subdirectories, and each trial would thus be shown twice in
# TensorBoard, one time for Tune's logs, and another time for PyTorch Lightning's logs.

# In[6]:


def train_mnist_tune(config, num_epochs=10, num_gpus=0, data_dir="~/data"):
    data_dir = os.path.expanduser(data_dir)
    model = LightningMNISTClassifier(config, data_dir)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
        enable_progress_bar=False,
        callbacks=[
            TuneReportCallback(
                {"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"},
                on="validation_end",
            )
        ],
    )
    trainer.fit(model)


# ### Configuring the search space
# 
# Now we configure the parameter search space. We would like to choose between three
# different layer and batch sizes. The learning rate should be sampled uniformly between
# `0.0001` and `0.1`. The `tune.loguniform()` function is syntactic sugar to make
# sampling between these different orders of magnitude easier, specifically
# we are able to also sample small values.

# In[7]:


config = {
    "layer_1_size": tune.choice([32, 64, 128]),
    "layer_2_size": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
}


# ### Selecting a scheduler
# 
# In this example, we use an [Asynchronous Hyperband](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/)
# scheduler. This scheduler decides at each iteration which trials are likely to perform
# badly, and stops these trials. This way we don't waste any resources on bad hyperparameter
# configurations.

# In[8]:


num_epochs = 10

scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)


# ### Changing the CLI output
# 
# We instantiate a `CLIReporter` to specify which metrics we would like to see in our
# output tables in the command line. This is optional, but can be used to make sure our
# output tables only include information we would like to see.

# In[9]:


reporter = CLIReporter(
    parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
    metric_columns=["loss", "mean_accuracy", "training_iteration"],
)


# ### Passing constants to the train function
# 
# The `data_dir`, `num_epochs` and `num_gpus` we pass to the training function
# are constants. To avoid including them as non-configurable parameters in the `config`
# specification, we can use `tune.with_parameters` to wrap around the training function.

# In[10]:


gpus_per_trial = 0
data_dir = "~/data"

train_fn_with_parameters = tune.with_parameters(
    train_mnist_tune, num_epochs=num_epochs, num_gpus=gpus_per_trial, data_dir=data_dir
)


# ### Training with GPUs
# 
# We can specify how many resources Tune should request for each trial.
# This also includes GPUs.
# 
# PyTorch Lightning takes care of moving the training to the GPUs. We
# already made sure that our code is compatible with that, so there's
# nothing more to do here other than to specify the number of GPUs
# we would like to use:

# In[11]:


resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}


# You can also specify {ref}`fractional GPUs for Tune <tune-parallelism>`,
# allowing multiple trials to share GPUs and thus increase concurrency under resource constraints.
# While the `gpus_per_trial` passed into
# Tune is a decimal value, the `gpus` passed into the `pl.Trainer` should still be an integer.
# Please note that if using fractional GPUs, it is the user's responsibility to
# make sure multiple trials can share GPUs and there is enough memory to do so.
# Ray does not automatically handle this for you.
# 
# If you want to use multiple GPUs per trial, you should check out {ref}`Getting Start with Lightning and Ray TorchTrainer <train-pytorch-lightning>`.
# 
# ### Putting it together
# 
# Lastly, we need to create a `Tuner()` object and start Ray Tune with `tuner.fit()`.
# 
# The full code looks like this:

# In[12]:


def tune_mnist_asha(num_samples=10, num_epochs=10, gpus_per_trial=0, data_dir="~/data"):
    config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_mnist_tune,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
        data_dir=data_dir,
    )
    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="tune_mnist_asha",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


# In the example above, Tune runs 10 trials with different hyperparameter configurations.
# An example output could look like so:
# 
# ```{code-block} bash
# :emphasize-lines: 12
# 
#   +------------------------------+------------+-------+----------------+----------------+-------------+--------------+----------+-----------------+----------------------+
#   | Trial name                   | status     | loc   |   layer_1_size |   layer_2_size |          lr |   batch_size |     loss |   mean_accuracy |   training_iteration |
#   |------------------------------+------------+-------+----------------+----------------+-------------+--------------+----------+-----------------+----------------------|
#   | train_mnist_tune_63ecc_00000 | TERMINATED |       |            128 |             64 | 0.00121197  |          128 | 0.120173 |       0.972461  |                   10 |
#   | train_mnist_tune_63ecc_00001 | TERMINATED |       |             64 |            128 | 0.0301395   |          128 | 0.454836 |       0.868164  |                    4 |
#   | train_mnist_tune_63ecc_00002 | TERMINATED |       |             64 |            128 | 0.0432097   |          128 | 0.718396 |       0.718359  |                    1 |
#   | train_mnist_tune_63ecc_00003 | TERMINATED |       |             32 |            128 | 0.000294669 |           32 | 0.111475 |       0.965764  |                   10 |
#   | train_mnist_tune_63ecc_00004 | TERMINATED |       |             32 |            256 | 0.000386664 |           64 | 0.133538 |       0.960839  |                    8 |
#   | train_mnist_tune_63ecc_00005 | TERMINATED |       |            128 |            128 | 0.0837395   |           32 | 2.32628  |       0.0991242 |                    1 |
#   | train_mnist_tune_63ecc_00006 | TERMINATED |       |             64 |            128 | 0.000158761 |          128 | 0.134595 |       0.959766  |                   10 |
#   | train_mnist_tune_63ecc_00007 | TERMINATED |       |             64 |             64 | 0.000672126 |           64 | 0.118182 |       0.972903  |                   10 |
#   | train_mnist_tune_63ecc_00008 | TERMINATED |       |            128 |             64 | 0.000502428 |           32 | 0.11082  |       0.975518  |                   10 |
#   | train_mnist_tune_63ecc_00009 | TERMINATED |       |             64 |            256 | 0.00112894  |           32 | 0.13472  |       0.971935  |                    8 |
#   +------------------------------+------------+-------+----------------+----------------+-------------+--------------+----------+-----------------+----------------------+
# ```
# 
# As you can see in the `training_iteration` column, trials with a high loss
# (and low accuracy) have been terminated early. The best performing trial used
# `layer_1_size=128`, `layer_2_size=64`, `lr=0.000502428` and
# `batch_size=32`.
# 
# ## Using Population Based Training to find the best parameters
# 
# The `ASHAScheduler` terminates those trials early that show bad performance.
# Sometimes, this stops trials that would get better after more training steps,
# and which might eventually even show better performance than other configurations.
# 
# Another popular method for hyperparameter tuning, called
# [Population Based Training](https://deepmind.com/blog/article/population-based-training-neural-networks),
# instead perturbs hyperparameters during the training run. Tune implements PBT, and
# we only need to make some slight adjustments to our code.
# 
# ### Adding checkpoints to the PyTorch Lightning module
# 
# First, we need to introduce
# another callback to save model checkpoints. Since Tune requires a call to
# `train.report()` after creating a new checkpoint to register it, we will use
# a combined reporting and checkpointing callback:

# In[13]:


TuneReportCheckpointCallback(
    metrics={"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"},
    filename="checkpoint",
    on="validation_end",
)


# The `checkpoint` value is the name of the checkpoint file within the
# checkpoint directory.
# 
# We also include checkpoint loading in our training function:

# In[14]:


def train_mnist_tune_checkpoint(config, num_epochs=10, num_gpus=0, data_dir="~/data"):
    data_dir = os.path.expanduser(data_dir)
    kwargs = {
        "max_epochs": num_epochs,
        # If fractional GPUs passed in, convert to int.
        "gpus": math.ceil(num_gpus),
        "logger": TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
        "enable_progress_bar": False,
        "callbacks": [
            TuneReportCheckpointCallback(
                metrics={"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"},
                filename="checkpoint",
                on="validation_end",
            )
        ],
    }

    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            kwargs["resume_from_checkpoint"] = os.path.join(checkpoint_dir, "checkpoint")

    model = LightningMNISTClassifier(config=config, data_dir=data_dir)
    trainer = pl.Trainer(**kwargs)

    trainer.fit(model)


# ### Configuring and running Population Based Training
# 
# We need to call Tune slightly differently:

# In[17]:


def tune_mnist_pbt(num_samples=10, num_epochs=10, gpus_per_trial=0, data_dir="~/data"):
    config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": 1e-3,
        "batch_size": 64,
    }

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": [32, 64, 128],
        },
    )

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"],
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_mnist_tune_checkpoint,
                num_epochs=num_epochs,
                num_gpus=gpus_per_trial,
                data_dir=data_dir,
            ),
            resources={"cpu": 1, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="tune_mnist_asha",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)


# Instead of passing tune parameters to the `config` dict, we start
# with fixed values, though we are also able to sample some of them, like the
# layer sizes. Additionally, we have to tell PBT how to perturb the hyperparameters.
# Note that the layer sizes are not tuned right here. This is because we cannot simply
# change layer sizes during a training run - which is what would happen in PBT.
# 
# To test running both of our main scripts (`tune_mnist_asha` and `tune_mnist_pbt`), all you have to do is specify
# a `data_dir` folder and run the scripts with reasonable parameters:

# In[18]:


data_dir = "~/data/"

tune_mnist_asha(num_samples=1, num_epochs=6, gpus_per_trial=0, data_dir=data_dir)
tune_mnist_pbt(num_samples=1, num_epochs=6, gpus_per_trial=0, data_dir=data_dir)


# If you have more resources available (e.g. a GPU), you can modify the above parameters accordingly.
# 
# An example output of a run could look like this:
# 
# ```bash
# +-----------------------------------------+------------+-------+----------------+----------------+-----------+--------------+-----------+-----------------+----------------------+
# | Trial name                              | status     | loc   |   layer_1_size |   layer_2_size |        lr |   batch_size |      loss |   mean_accuracy |   training_iteration |
# |-----------------------------------------+------------+-------+----------------+----------------+-----------+--------------+-----------+-----------------+----------------------|
# | train_mnist_tune_checkpoint_85489_00000 | TERMINATED |       |            128 |            128 | 0.001     |           64 | 0.108734  |        0.973101 |                   10 |
# | train_mnist_tune_checkpoint_85489_00001 | TERMINATED |       |            128 |            128 | 0.001     |           64 | 0.093577  |        0.978639 |                   10 |
# | train_mnist_tune_checkpoint_85489_00002 | TERMINATED |       |            128 |            256 | 0.0008    |           32 | 0.0922348 |        0.979299 |                   10 |
# | train_mnist_tune_checkpoint_85489_00003 | TERMINATED |       |             64 |            256 | 0.001     |           64 | 0.124648  |        0.973892 |                   10 |
# | train_mnist_tune_checkpoint_85489_00004 | TERMINATED |       |            128 |             64 | 0.001     |           64 | 0.101717  |        0.975079 |                   10 |
# | train_mnist_tune_checkpoint_85489_00005 | TERMINATED |       |             64 |             64 | 0.001     |           64 | 0.121467  |        0.969146 |                   10 |
# | train_mnist_tune_checkpoint_85489_00006 | TERMINATED |       |            128 |            256 | 0.00064   |           32 | 0.053446  |        0.987062 |                   10 |
# | train_mnist_tune_checkpoint_85489_00007 | TERMINATED |       |            128 |            256 | 0.001     |           64 | 0.129804  |        0.973497 |                   10 |
# | train_mnist_tune_checkpoint_85489_00008 | TERMINATED |       |             64 |            256 | 0.0285125 |          128 | 0.363236  |        0.913867 |                   10 |
# | train_mnist_tune_checkpoint_85489_00009 | TERMINATED |       |             32 |            256 | 0.001     |           64 | 0.150946  |        0.964201 |                   10 |
# +-----------------------------------------+------------+-------+----------------+----------------+-----------+--------------+-----------+-----------------+----------------------+
# ```
# 
# As you can see, each sample ran the full number of 10 iterations.
# All trials ended with quite good parameter combinations and showed relatively good performances.
# In some runs, the parameters have been perturbed. And the best configuration even reached a
# mean validation accuracy of `0.987062`!
# 
# In summary, PyTorch Lightning Modules are easy to extend to use with Tune. It just took
# us importing one or two callbacks and a small wrapper function to get great performing
# parameter configurations.
# 
# ## More PyTorch Lightning Examples
# 
# - {doc}`/tune/examples/includes/mnist_ptl_mini`:
#   A minimal example of using [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
#   to train a MNIST model. This example utilizes the Ray Tune-provided
#   {ref}`PyTorch Lightning callbacks <tune-integration-pytorch-lightning>`.
#   See also {ref}`this tutorial for a full walkthrough <tune-pytorch-lightning-ref>`.
# - {ref}`A walkthrough tutorial for using Ray Tune with Pytorch-Lightning <tune-pytorch-lightning-ref>`.
# - {doc}`/tune/examples/includes/mlflow_ptl_example`: Example for using [MLflow](https://github.com/mlflow/mlflow/)
#   and [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) with Ray Tune.