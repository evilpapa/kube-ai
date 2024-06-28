#!/usr/bin/env python
# coding: utf-8

# # How to use Tune with PyTorch
# 
# (tune-pytorch-cifar-ref)=
# 
# In this walkthrough, we will show you how to integrate Tune into your PyTorch
# training workflow. We will follow [this tutorial from the PyTorch documentation](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
# for training a CIFAR10 image classifier.
# 
# ```{image} /images/pytorch_logo.png
# :align: center
# ```
# 
# Hyperparameter tuning can make the difference between an average model and a highly
# accurate one. Often simple things like choosing a different learning rate or changing
# a network layer size can have a dramatic impact on your model performance. Fortunately,
# Tune makes exploring these optimal parameter combinations easy - and works nicely
# together with PyTorch.
# 
# As you will see, we only need to add some slight modifications. In particular, we
# need to
# 
# 1. wrap data loading and training in functions,
# 2. make some network parameters configurable,
# 3. add checkpointing (optional),
# 4. and define the search space for the model tuning
# 
# :::{note}
# To run this example, you will need to install the following:
# 
# ```bash
# $ pip install ray torch torchvision
# ```
# :::
# 
# ```{contents}
# :backlinks: none
# :local: true
# ```

# ## Setup / Imports
# 
# Let's start with the imports:

# In[1]:


import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler


# Most of the imports are needed for building the PyTorch model. Only the last three
# imports are for Ray Tune.
# 
# ## Data loaders
# 
# We wrap the data loaders in their own function and pass a global data directory.
# This way we can share a data directory between different trials.

# In[2]:


def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


# ## Configurable neural network
# 
# We can only tune those parameters that are configurable. In this example, we can specify
# the layer sizes of the fully connected layers:

# In[3]:


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ## The train function
# 
# Now it gets interesting, because we introduce some changes to the example [from the PyTorch
# documentation](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).
# 
# (communicating-with-ray-tune)=
# 
# The full code example looks like this:

# In[4]:


def train_cifar(config):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # To restore a checkpoint, use `train.get_checkpoint()`.
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    data_dir = os.path.abspath("./data")
    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `train.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        train.report({"loss": (val_loss / val_steps), "accuracy": correct / total}, checkpoint=checkpoint)
    print("Finished Training")


# As you can see, most of the code is adapted directly from the example.
# 
# ## Test set accuracy
# 
# Commonly the performance of a machine learning model is tested on a hold-out test
# set with data that has not been used for training the model. We also wrap this in a
# function:

# In[5]:


def test_best_model(best_result):
    best_trained_model = Net(best_result.config["l1"], best_result.config["l2"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("Best trial test set accuracy: {}".format(correct / total))


# As you can see, the function also expects a `device` parameter, so we can do the
# test set validation on a GPU.
# 
# ## Configuring the search space
# 
# Lastly, we need to define Tune's search space. Here is an example:

# In[6]:


config = {
    "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16]),
}


# The `tune.sample_from()` function makes it possible to define your own sample
# methods to obtain hyperparameters. In this example, the `l1` and `l2` parameters
# should be powers of 2 between 4 and 256, so either 4, 8, 16, 32, 64, 128, or 256.
# The `lr` (learning rate) should be uniformly sampled between 0.0001 and 0.1. Lastly,
# the batch size is a choice between 2, 4, 8, and 16.
# 
# At each trial, Tune will now randomly sample a combination of parameters from these
# search spaces. It will then train a number of models in parallel and find the best
# performing one among these. We also use the `ASHAScheduler` which will terminate bad
# performing trials early.
# 
# You can specify the number of CPUs, which are then available e.g.
# to increase the `num_workers` of the PyTorch `DataLoader` instances. The selected
# number of GPUs are made visible to PyTorch in each trial. Trials do not have access to
# GPUs that haven't been requested for them - so you don't have to care about two trials
# using the same set of resources.
# 
# Here we can also specify fractional GPUs, so something like `gpus_per_trial=0.5` is
# completely valid. The trials will then share GPUs among each other.
# You just have to make sure that the models still fit in the GPU memory.
# 
# After training the models, we will find the best performing one and load the trained
# network from the checkpoint file. We then obtain the test set accuracy and report
# everything by printing.
# 
# The full main function looks like this:

# In[7]:


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_cifar),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)

main(num_samples=2, max_num_epochs=2, gpus_per_trial=0)


# If you run the code, an example output could look like this:
# 
# ```{code-block} bash
# :emphasize-lines: 7
# 
#   Number of trials: 10 (10 TERMINATED)
#   +-------------------------+------------+-------+------+------+-------------+--------------+---------+------------+----------------------+
#   | Trial name              | status     | loc   |   l1 |   l2 |          lr |   batch_size |    loss |   accuracy |   training_iteration |
#   |-------------------------+------------+-------+------+------+-------------+--------------+---------+------------+----------------------|
#   | train_cifar_87d1f_00000 | TERMINATED |       |   64 |    4 | 0.00011629  |            2 | 1.87273 |     0.244  |                    2 |
#   | train_cifar_87d1f_00001 | TERMINATED |       |   32 |   64 | 0.000339763 |            8 | 1.23603 |     0.567  |                    8 |
#   | train_cifar_87d1f_00002 | TERMINATED |       |    8 |   16 | 0.00276249  |           16 | 1.1815  |     0.5836 |                   10 |
#   | train_cifar_87d1f_00003 | TERMINATED |       |    4 |   64 | 0.000648721 |            4 | 1.31131 |     0.5224 |                    8 |
#   | train_cifar_87d1f_00004 | TERMINATED |       |   32 |   16 | 0.000340753 |            8 | 1.26454 |     0.5444 |                    8 |
#   | train_cifar_87d1f_00005 | TERMINATED |       |    8 |    4 | 0.000699775 |            8 | 1.99594 |     0.1983 |                    2 |
#   | train_cifar_87d1f_00006 | TERMINATED |       |  256 |    8 | 0.0839654   |           16 | 2.3119  |     0.0993 |                    1 |
#   | train_cifar_87d1f_00007 | TERMINATED |       |   16 |  128 | 0.0758154   |           16 | 2.33575 |     0.1327 |                    1 |
#   | train_cifar_87d1f_00008 | TERMINATED |       |   16 |    8 | 0.0763312   |           16 | 2.31129 |     0.1042 |                    4 |
#   | train_cifar_87d1f_00009 | TERMINATED |       |  128 |   16 | 0.000124903 |            4 | 2.26917 |     0.1945 |                    1 |
#   +-------------------------+------------+-------+------+------+-------------+--------------+---------+------------+----------------------+
# 
# 
#   Best trial config: {'l1': 8, 'l2': 16, 'lr': 0.0027624906698231976, 'batch_size': 16, 'data_dir': '...'}
#   Best trial final validation loss: 1.1815014744281769
#   Best trial final validation accuracy: 0.5836
#   Best trial test set accuracy: 0.5806
# ```
# 
# As you can see, most trials have been stopped early in order to avoid wasting resources.
# The best performing trial achieved a validation accuracy of about 58%, which could
# be confirmed on the test set.
# 
# So that's it! You can now tune the parameters of your PyTorch models.
# 
# ## See More PyTorch Examples
# 
# - {doc}`/tune/examples/includes/mnist_pytorch`: Converts the PyTorch MNIST example to use Tune with the function-based API.
#   Also shows how to easily convert something relying on argparse to use Tune.
# - {doc}`/tune/examples/includes/pbt_convnet_function_example`: Example training a ConvNet with checkpointing in function API.
# - {doc}`/tune/examples/includes/mnist_pytorch_trainable`: Converts the PyTorch MNIST example to use Tune with Trainable API.
#   Also uses the HyperBandScheduler and checkpoints the model at the end.
