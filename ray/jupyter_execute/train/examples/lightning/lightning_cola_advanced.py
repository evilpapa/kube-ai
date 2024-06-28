#!/usr/bin/env python
# coding: utf-8

# (lightning_advanced_example)=
# 
# # Fine-tune a PyTorch Lightning Text Classifier with Ray Data
# 
# :::{note}
# 
# This is an intermediate example demonstrates how to use [Ray Data](data) with PyTorch Lightning in Ray Train.
# 
# If you just want to quickly convert your existing PyTorch Lightning scripts into Ray Train, you can refer to the [Lightning Quick Start Guide](train-pytorch-lightning).
# 
# :::
# 
# This demo introduces how to fine-tune a text classifier on the [CoLA(The Corpus of Linguistic Acceptability)](https://nyu-mll.github.io/CoLA/) dataset using a pre-trained BERT model. In particular, it follows three steps:
# - Preprocess the CoLA dataset with Ray Data.
# - Define a training function with PyTorch Lightning.
# - Launch distributed training with Ray Train's TorchTrainer.

# Run the following line in order to install all the necessary dependencies:

# In[1]:


SMOKE_TEST = True


# In[2]:


get_ipython().system('pip install numpy datasets "transformers>=4.19.1" "pytorch_lightning>=1.6.5"')


# Start by importing the needed libraries:

# In[3]:


import ray
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, load_metric


# 

# ## Pre-process CoLA Dataset
# 
# CoLA is a dataset for binary sentence classification with 10.6K training examples. First, download the dataset and metrics using the Hugging Face datasets API, and create a Ray Dataset for each split accordingly.

# In[ ]:


dataset = load_dataset("glue", "cola")

train_dataset = ray.data.from_huggingface(dataset["train"])
validation_dataset = ray.data.from_huggingface(dataset["validation"])


# Next, tokenize the input sentences and pad the ID sequence to length 128 using the `bert-base-uncased` tokenizer. The {meth}`map_batches <ray.data.Dataset.map_batches>` applies this preprocessing function on all data samples.

# 

# In[5]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_sentence(batch):
    outputs = tokenizer(
        batch["sentence"].tolist(),
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    outputs["label"] = batch["label"]
    return outputs

train_dataset = train_dataset.map_batches(tokenize_sentence, batch_format="numpy")
validation_dataset = validation_dataset.map_batches(tokenize_sentence, batch_format="numpy")


# ## Define a PyTorch Lightning model
# 
# You don't have to make any changes to your `LightningModule` definition. Just copy and paste your code here:

# In[6]:


class SentimentModel(pl.LightningModule):
    def __init__(self, lr=2e-5, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.num_classes = 2
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=self.num_classes
        )
        self.metric = load_metric("glue", "cola")
        self.predictions = []
        self.references = []

    def forward(self, batch):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        logits = self.forward(batch)
        loss = F.cross_entropy(logits.view(-1, self.num_classes), labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        logits = self.forward(batch)
        preds = torch.argmax(logits, dim=1)
        self.predictions.append(preds)
        self.references.append(labels)

    def on_validation_epoch_end(self):
        predictions = torch.concat(self.predictions).view(-1)
        references = torch.concat(self.references).view(-1)
        matthews_correlation = self.metric.compute(
            predictions=predictions, references=references
        )

        # self.metric.compute() returns a dictionary:
        # e.g. {"matthews_correlation": 0.53}
        self.log_dict(matthews_correlation, sync_dist=True)
        self.predictions.clear()
        self.references.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, eps=self.eps)


# ## Define a training function
# 
# Define a [training function](train-overview-training-function) that includes all of your lightning training logic. {class}`TorchTrainer <ray.train.torch.TorchTrainer>` launches this function on each worker in parallel. 
# 

# In[7]:


import ray.train
from ray.train.lightning import (
    prepare_trainer,
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)

train_func_config = {
    "lr": 1e-5,
    "eps": 1e-8,
    "batch_size": 16,
    "max_epochs": 5,
}

def train_func(config):
    # Unpack the input configs passed from `TorchTrainer(train_loop_config)`
    lr = config["lr"]
    eps = config["eps"]
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]

    # Fetch the Dataset shards
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("validation")

    # Create a dataloader for Ray Datasets
    train_ds_loader = train_ds.iter_torch_batches(batch_size=batch_size)
    val_ds_loader = val_ds.iter_torch_batches(batch_size=batch_size)

    # Model
    model = SentimentModel(lr=lr, eps=eps)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        enable_progress_bar=False,
    )

    trainer = prepare_trainer(trainer)

    trainer.fit(model, train_dataloaders=train_ds_loader, val_dataloaders=val_ds_loader)


# To enable distributed training with Ray Train, configure the Lightning Trainer with the following utilities:
# 
# - {class}`~ray.train.lightning.RayDDPStrategy`
# - {class}`~ray.train.lightning.RayLightningEnvironment`
# - {class}`~ray.train.lightning.RayTrainReportCallback`
# 
# 
# To ingest Ray Data with Lightning Trainer, follow these three steps:
# 
# - Feed the full Ray dataset to Ray `TorchTrainer` (details in the next section).
# - Use {meth}`ray.train.get_dataset_shard <ray.train.get_dataset_shard>` to fetch the sharded dataset on each worker.
# - Use {meth}`ds.iter_torch_batches <ray.data.Dataset.iter_torch_batches>` to create a Ray data loader for Lightning Trainer.
# 
# :::{seealso}
# 
# - {ref}`Lightning Quick Start Guide <train-pytorch-lightning>`
# - {ref}`User Guides for Ray Data <data-ingest-torch>`
# 
# :::

# In[8]:


if SMOKE_TEST:
    train_func_config["max_epochs"] = 2
    train_dataset = train_dataset.random_sample(0.1)
    validation_dataset = validation_dataset.random_sample(0.1)


# ## Distributed training with Ray TorchTrainer
# 
# Next, define a {class}`TorchTrainer <ray.train.torch.TorchTrainer>` to launch your training function on 4 GPU workers. 
# 
# You can pass the full Ray dataset to the `datasets` argument of ``TorchTrainer``. TorchTrainer automatically shards the datasets among multiple workers.

# In[9]:


from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, DataConfig


# Save the top-2 checkpoints according to the evaluation metric
# The checkpoints and metrics are reported by `RayTrainReportCallback`
run_config = RunConfig(
    name="ptl-sent-classification",
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="matthews_correlation",
        checkpoint_score_order="max",
    ),
)

# Schedule four workers for DDP training (1 GPU/worker by default)
scaling_config = ScalingConfig(num_workers=4, use_gpu=True)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=train_func_config,
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={"train": train_dataset, "validation": validation_dataset}, # <- Feed the Ray Datasets here
)


# In[10]:


result = trainer.fit()


# :::{note}
# Note that this examples uses Ray Data for data ingestion for faster preprocessing, but you can also continue to use the native `PyTorch DataLoader` or `LightningDataModule`. See {ref}`Train a Pytorch Lightning Image Classifier <lightning_mnist_example>`. 
# 
# :::

# In[11]:


result


# ## See also
# 
# * [Ray Train Examples](train-examples) for more use cases
# 
# * [Ray Train User Guides](train-user-guides) for how-to guides
