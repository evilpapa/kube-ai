#!/usr/bin/env python
# coding: utf-8

# # Finetuning a Pytorch Image Classifier with Ray Train
# This example fine tunes a pre-trained ResNet model with Ray Train. 
# 
# For this example, the network architecture consists of the intermediate layer output of a pre-trained ResNet model, which feeds into a randomly initialized linear layer that outputs classification logits for our new task.
# 
# 
# 

# ## Load and preprocess finetuning dataset
# This example is adapted from Pytorch's [Finetuning Torchvision Models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) tutorial.
# We will use *hymenoptera_data* as the finetuning dataset, which contains two classes (bees and ants) and 397 total images (across training and validation). This is a quite small dataset and used only for demonstration purposes. 

# In[15]:


# To run full example, set SMOKE_TEST as False
SMOKE_TEST = True


# The dataset is publicly available [here](https://www.kaggle.com/datasets/ajayrana/hymenoptera-data). Note that it is structured with directory names as the labels. Use `torchvision.datasets.ImageFolder()` to load the images and their corresponding labels.

# In[2]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

def download_datasets():
    os.system(
        "wget https://download.pytorch.org/tutorial/hymenoptera_data.zip >/dev/null 2>&1"
    )
    os.system("unzip hymenoptera_data.zip >/dev/null 2>&1")

# Download and build torch datasets
def build_datasets():
    torch_datasets = {}
    for split in ["train", "val"]:
        torch_datasets[split] = datasets.ImageFolder(
            os.path.join("./hymenoptera_data", split), data_transforms[split]
        )
    return torch_datasets


# In[16]:


if SMOKE_TEST:
    from torch.utils.data import Subset

    def build_datasets():
        torch_datasets = {}
        for split in ["train", "val"]:
            torch_datasets[split] = datasets.ImageFolder(
                os.path.join("./hymenoptera_data", split), data_transforms[split]
            )
            
        # Only take a subset for smoke test
        for split in ["train", "val"]:
            indices = list(range(100))
            torch_datasets[split] = Subset(torch_datasets[split], indices)
        return torch_datasets


# ## Initialize Model and Fine-tuning configs

# Next, let's define the training configuration that will be passed into the training loop function later.

# In[4]:


train_loop_config = {
    "input_size": 224,  # Input image size (224 x 224)
    "batch_size": 32,  # Batch size for training
    "num_epochs": 10,  # Number of epochs to train for
    "lr": 0.001,  # Learning Rate
    "momentum": 0.9,  # SGD optimizer momentum
}


# Next, let's define our model. You can either create a model from pre-trained weights or reload the model checkpoint from a previous run.

# In[5]:


import os
import torch
from ray.train import Checkpoint

# Option 1: Initialize model with pretrained weights
def initialize_model():
    # Load pretrained model params
    model = models.resnet50(pretrained=True)

    # Replace the original classifier with a new Linear layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Ensure all params get updated during finetuning
    for param in model.parameters():
        param.requires_grad = True
    return model


# Option 2: Initialize model with an Train checkpoint
# Replace this with your own uri
CHECKPOINT_FROM_S3 = Checkpoint(
    path="s3://air-example-data/finetune-resnet-checkpoint/TorchTrainer_4f69f_00000_0_2023-02-14_14-04-09/checkpoint_000001/"
)


def initialize_model_from_checkpoint(checkpoint: Checkpoint):
    with checkpoint.as_directory() as tmpdir:
        state_dict = torch.load(os.path.join(tmpdir, "checkpoint.pt"))
    resnet50 = initialize_model()
    resnet50.load_state_dict(state_dict["model"])
    return resnet50


# ## Define the Training Loop
# 
# The `train_loop_per_worker` function defines the fine-tuning procedure for each worker.
# 
# **1. Prepare dataloaders for each worker**:
# - This tutorial assumes you are using PyTorch's native `torch.utils.data.Dataset` for data input. {meth}`train.torch.prepare_data_loader() <ray.train.torch.prepare_data_loader>`  prepares your dataLoader for distributed execution. You can also use Ray Data for more efficient preprocessing. For more details on using Ray Data for for images, see the {doc}`Working with Images </data/working-with-images>` Ray Data user guide.
# 
# **2. Prepare your model**:
# - {meth}`train.torch.prepare_model() <ray.train.torch.prepare_model>` prepares the model for distributed training. Under the hood, it converts your torch model to `DistributedDataParallel` model, which synchronize its weights across all workers.
# 
# **3. Report metrics and checkpoint**:
# - {meth}`train.report() <ray.train.report>` will report metrics and checkpoints to Ray Train.
# - Saving checkpoints through {meth}`train.report(metrics, checkpoint=...) <ray.train.report>` will automatically [upload checkpoints to cloud storage](tune-cloud-checkpointing) (if configured), and allow you to easily enable Ray Train worker fault tolerance in the future.

# In[6]:


import os
from tempfile import TemporaryDirectory

import ray.train as train
from ray.train import Checkpoint



def evaluate(logits, labels):
    _, preds = torch.max(logits, 1)
    corrects = torch.sum(preds == labels).item()
    return corrects


def train_loop_per_worker(configs):
    import warnings

    warnings.filterwarnings("ignore")

    # Calculate the batch size for a single worker
    worker_batch_size = configs["batch_size"] // train.get_context().get_world_size()

    # Download dataset once on local rank 0 worker
    if train.get_context().get_local_rank() == 0:
        download_datasets()
    torch.distributed.barrier()

    # Build datasets on each worker
    torch_datasets = build_datasets()

    # Prepare dataloader for each worker
    dataloaders = dict()
    dataloaders["train"] = DataLoader(
        torch_datasets["train"], batch_size=worker_batch_size, shuffle=True
    )
    dataloaders["val"] = DataLoader(
        torch_datasets["val"], batch_size=worker_batch_size, shuffle=False
    )

    # Distribute
    dataloaders["train"] = train.torch.prepare_data_loader(dataloaders["train"])
    dataloaders["val"] = train.torch.prepare_data_loader(dataloaders["val"])

    device = train.torch.get_device()

    # Prepare DDP Model, optimizer, and loss function
    model = initialize_model()
    model = train.torch.prepare_model(model)

    optimizer = optim.SGD(
        model.parameters(), lr=configs["lr"], momentum=configs["momentum"]
    )
    criterion = nn.CrossEntropyLoss()

    # Start training loops
    for epoch in range(configs["num_epochs"]):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # calculate statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += evaluate(outputs, labels)

            size = len(torch_datasets[phase]) // train.get_context().get_world_size()
            epoch_loss = running_loss / size
            epoch_acc = running_corrects / size

            if train.get_context().get_world_rank() == 0:
                print(
                    "Epoch {}-{} Loss: {:.4f} Acc: {:.4f}".format(
                        epoch, phase, epoch_loss, epoch_acc
                    )
                )

            # Report metrics and checkpoint every epoch
            if phase == "val":
                with TemporaryDirectory() as tmpdir:
                    state_dict = {
                        "epoch": epoch,
                        "model": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    torch.save(state_dict, os.path.join(tmpdir, "checkpoint.pt"))
                    train.report(
                        metrics={"loss": epoch_loss, "acc": epoch_acc},
                        checkpoint=Checkpoint.from_directory(tmpdir),
                    )


# Next, setup the TorchTrainer:

# In[7]:


from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

# Scale out model training across 4 GPUs.
scaling_config = ScalingConfig(
    num_workers=4, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
)

# Save the latest checkpoint
checkpoint_config = CheckpointConfig(num_to_keep=1)

# Set experiment name and checkpoint configs
run_config = RunConfig(
    name="finetune-resnet",
    storage_path="/tmp/ray_results",
    checkpoint_config=checkpoint_config,
)


# In[8]:


if SMOKE_TEST:
    scaling_config = ScalingConfig(
        num_workers=8, use_gpu=False, resources_per_worker={"CPU": 1}
    )
    train_loop_config["num_epochs"] = 1


# In[9]:


trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config=train_loop_config,
    scaling_config=scaling_config,
    run_config=run_config,
)

result = trainer.fit()
print(result)


# ## Load the checkpoint for prediction:
# 
#  
#  The metadata and checkpoints have already been saved in the `storage_path` specified in TorchTrainer:

# We now need to load the trained model and evaluate it on test data. The best model parameters have been saved in `log_dir`. We can load the resulting checkpoint from our fine-tuning run using the previously defined `initialize_model_from_checkpoint()` function.

# In[11]:


model = initialize_model_from_checkpoint(result.checkpoint)
device = torch.device("cuda")


# In[12]:


if SMOKE_TEST:
    device = torch.device("cpu")


# Finally, define a simple evaluation loop and check the performance of the checkpoint model.

# In[14]:


model = model.to(device)
model.eval()

download_datasets()
torch_datasets = build_datasets()
dataloader = DataLoader(torch_datasets["val"], batch_size=32, num_workers=4)
corrects = 0
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    preds = model(inputs)
    corrects += evaluate(preds, labels)

print("Accuracy: ", corrects / len(dataloader.dataset))

