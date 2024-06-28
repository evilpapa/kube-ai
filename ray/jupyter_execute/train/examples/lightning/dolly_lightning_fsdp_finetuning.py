#!/usr/bin/env python
# coding: utf-8

# (dolly_lightning_fsdp_finetuning)=
# 
# # Fine-tune `dolly-v2-7b` with Ray Train, PyTorch Lightning and FSDP

# In this example, we demonstrate how to use Ray Train to fine-tune a [`dolly-v2-7b`](https://huggingface.co/databricks/dolly-v2-7b) model. `dolly-v2-7b` is a 7 billion parameter causal language model created by Databricks, derived from EleutherAI’s [Pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b), and fine-tuned on a [~15K record instruction corpus](https://github.com/databrickslabs/dolly/tree/master/data).
# 
# We load the pre-trained model from the HuggingFace model hub into a LightningModule and launch an FSDP fine-tuning job across 16 T4 GPUs with the help of {class}`Ray TorchTrainer <ray.train.torch.TorchTrainer>`. It is also straightforward to fine-tune other similar large language models in a similar manner as shown in this example.
# 
# Before starting this example, we highly recommend reading [Ray Train Key Concepts](train-key-concepts) and [Ray Data Key Concepts](data_key_concepts).

# ## Set up ray cluster 
# In this example, we are using a Ray cluster with a `g4dn.8xlarge` head node and 15 `g4dn.4xlarge` worker nodes. Each instance has one Tesla T4 GPU (16GiB Memory). 
# 
# We define a `runtime_env` to install the necessary Python libraries on each node. You can skip this step if you have already installed all the required packages in your workers' base image. We tested this example with `pytorch_lightning==2.0.2` and `transformers==4.29.2`.

# In[ ]:


import ray

ray.init(
    runtime_env={
        "pip": [
            "datasets",
            "evaluate",
            "transformers>=4.26.0",
            "torch>=1.12.0",
            "pytorch_lightning>=2.0",
        ]
    }
)


# In[2]:


MODEL_NAME = "databricks/dolly-v2-7b"


# ## Prepare your data 
# We are using tiny_shakespeare for fine-tuning, which contains 40,000 lines of Shakespeare from a variety of Shakespeare's plays. Featured in Andrej Karpathy's blog post ['The Unreasonable Effectiveness of Recurrent Neural Networks'](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). 
# 
# Dataset samples:
# ```
# BAPTISTA:
# I know him well: you are welcome for his sake.
# 
# GREMIO:
# Saving your tale, Petruchio, I pray,
# Let us, that are poor petitioners, speak too:
# Baccare! you are marvellous forward.
# 
# PETRUCHIO:
# O, pardon me, Signior Gremio; I would fain be doing.
# ```
# 
# Here, we have adopted similar pre-processing logic from another demo: {ref}`GPT-J-6B Fine-Tuning with Ray Train and DeepSpeed <gptj_deepspeed_finetune>`.

# In[ ]:


import ray
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def split_text(batch: pd.DataFrame) -> pd.DataFrame:
    text = list(batch["text"])
    flat_text = "".join(text)
    split_text = [
        x.strip()
        for x in flat_text.split("\n")
        if x.strip() and not x.strip()[-1] == ":"
    ]
    return pd.DataFrame(split_text, columns=["text"])


def tokenize(batch: pd.DataFrame) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    ret = tokenizer(
        list(batch["text"]),
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="np",
    )
    ret["labels"] = ret["input_ids"].copy()
    return dict(ret)

hf_dataset = load_dataset("tiny_shakespeare")
train_ds = ray.data.from_huggingface(hf_dataset["train"])


# We first split the original paragraphs into multiple sentences, then tokenize them. Here are some samples:

# In[4]:


# First split the dataset into multiple sentences.
train_ds = train_ds.map_batches(split_text, batch_format="pandas")
train_ds.take(10)


# In[5]:


# Then tokenize the dataset.
train_ds = train_ds.map_batches(tokenize, batch_format="pandas")


# ## Define your lightning model
# 
# In this example, we use the [dolly-v2-7b](https://huggingface.co/databricks/dolly-v2-7b) model for finetuning. It is an instruction-following large language model trained on the Databricks machine learning platform that is licensed for commercial use. We load the model weights from Huggingface Model Hub and encapsulate it into a `pl.LightningModule`.
# 
# :::{note}
# Make sure you pass the FSDP wrapped model parameters `self.trainer.model.parameters()` into the optimizer, instead of `self.model.parameters()`. 
# :::
# 

# In[6]:


import torch
import pytorch_lightning as pl

class DollyV2Model(pl.LightningModule):
    def __init__(self, lr=2e-5, eps=1e-8):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.eps = eps
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def forward(self, batch):
        outputs = self.model(
            batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["labels"]
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        if self.global_rank == 0:
            print(self.trainer.model)
        return torch.optim.AdamW(self.trainer.model.parameters(), lr=self.lr, eps=self.eps)


# ## Configure your FSDP strategy
# As `dolly-v2-7b` is a relatively large model, it cannot be properly fit into a single commercial GPU. In this example, we use the FSDP strategy to shard model parameters across multiple workers. This allows us to avoid GPU out-of-memory issues and support a larger global batch size.
# 
# ![](https://user-images.githubusercontent.com/26745457/236892936-d4b91751-4689-421e-ac5f-edfd2eeeb635.png)
# Image source: [Fully Sharded Data Parallel: faster AI training with fewer GPUs](https://engineering.fb.com/2021/07/15/open-source/fsdp/)
# 
# :::{note}
# FSDP is a type of data parallelism that shards model parameters, optimizer states and gradients across DDP ranks. This was inspired by Xu et al. as well as the ZeRO Stage 3 from DeepSpeed. You may refer to these blogs for more information:
# 
# - [Fully Sharded Data Parallel: faster AI training with fewer GPUs](https://engineering.fb.com/2021/07/15/open-source/fsdp/)
# - [Getting Started with Fully Sharded Data Parallel(FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#:~:text=FSDP%20is%20a%20type%20of,sizes%20for%20our%20training%20job.)
# - [PyTorch FSDP Tutorial](https://www.youtube.com/watch?v=8_k76AHu__s&list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT)
# :::
# 
# To start training with Lightning's [FSDPStrategy](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html#lightning.pytorch.strategies.FSDPStrategy), you only need to create a {class}`~ray.train.lightning.RayFSDPStrategy` with the same initialization arguments.
# 

# In[ ]:


import functools
import pytorch_lightning as pl 

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from ray.train.lightning import RayFSDPStrategy


# Define the model sharding policy:
# Wrap every GPTNeoXLayer as its own FSDP instance
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls = {GPTNeoXLayer}
)

fsdp_strategy = RayFSDPStrategy(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    forward_prefetch=True,
    auto_wrap_policy=auto_wrap_policy,
    limit_all_gathers=True,
    activation_checkpointing=[GPTNeoXLayer],
)


# :::{tip}
# 
# Some tips for FSDP configuration:
# - `sharding_strategy`:
#     - `ShardingStrategy.NO_SHARD`: Parameters, gradients, and optimizer states are not sharded. Similar to DDP.
#     - `ShardingStrategy.SHARD_GRAD_OP`: Gradients and optimizer states are sharded during computation, and additionally, parameters are sharded outside computation. Similar to ZeRO stage-2.
#     - `ShardingStrategy.FULL_SHARD`: Parameters, gradients, and optimizer states are sharded. It has minimal GRAM usage among the 3 options. Similar to ZeRO stage-3.
# - `auto_wrap_policy`:
#     - Model layers are often wrapped with FSDP in a layered fashion. This means that only the layers in a single FSDP instance are required to aggregate all parameters to a single device during forwarding or backward calculations.
#     - Use `transformer_auto_wrap_policy` to automatically wrap each Transformer Block into a single FSDP instance. 
# - `backward_prefetch` and `forward_prefetch`:
#     - Overlap the upcoming all-gather while executing the current forward/backward pass. It can improve throughput but may slightly increase peak memory usage.
# :::

# ## Fine-tune with Ray TorchTrainer
# 
# Ray TorchTrainer allows you to scale your PyTorch Lightning training workload over multiple nodes. See {ref}`Configuring Scale and GPUs <train_scaling_config>` for more details.

# In[8]:


num_workers = 16
batch_size_per_worker = 10


# In[9]:


# To accelerate release tests
train_ds = train_ds.limit(num_workers * batch_size_per_worker * 10)  # each worker has 10 batches


# Additionally, remember to define a Lightning callback that saves and reports checkpoints. Ray Train offers a simple implementation, {meth}`~ray.train.lightning.RayTrainReportCallback`, which persists your checkpoint and metrics in remote storage at the end of each training epoch. 
# 
# Note you can also implement your own report callback with customized logics, such as saving customized checkpoint files or reporting at a different frequency.

# In[10]:


from pytorch_lightning.callbacks import TQDMProgressBar

# Create a customized progress bar for Ray Data Iterable Dataset
class DollyV2ProgressBar(TQDMProgressBar):
    def __init__(self, num_iters_per_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_iters_per_epoch = num_iters_per_epoch
    
    def on_train_epoch_start(self, trainer, *_):
        super().on_train_epoch_start(trainer, *_)
        self.train_progress_bar.reset(self.num_iters_per_epoch)

num_iters_per_epoch = train_ds.count() // (num_workers * batch_size_per_worker)
prog_bar = DollyV2ProgressBar(num_iters_per_epoch)


# In[11]:


from ray.train import Checkpoint
from ray.train.lightning import RayLightningEnvironment, RayTrainReportCallback, prepare_trainer

# Training function for each worker
def train_func(config):
    lr = config["lr"]
    eps = config["eps"]
    strategy = config["strategy"]
    batch_size_per_worker = config["batch_size_per_worker"]

    # Model
    model = DollyV2Model(lr=lr, eps=eps)

    # Ray Data Ingestion
    train_ds = ray.train.get_dataset_shard("train")
    train_dataloader = train_ds.iter_torch_batches(batch_size=batch_size_per_worker)

    # Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=1, 
        devices="auto",
        accelerator="auto", 
        precision="16-mixed",
        strategy=strategy,
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        enable_checkpointing=False,
    )

    trainer = prepare_trainer(trainer)

    trainer.fit(model, train_dataloaders=train_dataloader)


# ```{note}
# Since this example runs with multiple nodes, we need to persist checkpoints
# and other outputs to some external storage for access after training has completed.
# **You should set up cloud storage or NFS, then replace `storage_path` with your own cloud bucket URI or NFS path.**
# 
# See the [storage guide](tune-storage-options) for more details.
# ```

# In[12]:


storage_path="s3://your-bucket-here"  # TODO: Set up cloud storage
# storage_path="/mnt/path/to/nfs"     # TODO: Alternatively, set up NFS


# In[13]:


storage_path = "/mnt/cluster_storage"


# In[14]:


from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

# Save Ray Train checkpoints according to the performance on validation set
run_config = RunConfig(
    name="finetune_dolly-v2-7b",
    storage_path=storage_path,
    checkpoint_config=CheckpointConfig(num_to_keep=1),
)

# Scale the FSDP training workload across 16 GPUs
# You can change this config based on your compute resources.
scaling_config = ScalingConfig(
    num_workers=num_workers, use_gpu=True, trainer_resources={"memory": 100 * 1024 ** 3}
)

# Configuration to pass into train_func
train_config = {
    "lr": 2e-5,
    "eps": 1e-8,
    "strategy": fsdp_strategy,
    "batch_size_per_worker": 10
}

# Define a TorchTrainer and launch you training workload
ray_trainer = TorchTrainer(
    train_func,
    train_loop_config=train_config,
    run_config=run_config,
    scaling_config=scaling_config,
    datasets={"train": train_ds},
)
result = ray_trainer.fit()

result


# We finished training in 2877s. The price for an on-demand g4dn.4xlarge instance is `$1.204/hour`, while a g4dn.8xlarge instance costs `$2.176/hour`. The total cost would be `($1.204 * 15 + $2.176) * 2877 / 3600 = $16.17`.

# ## Text-generation with HuggingFace Pipeline
# 
# We can use the [HuggingFace Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) to generate predictions from our fine-tuned model. Let's input some prompts and see if our tuned Dolly can speak like Shakespeare:

# In[ ]:


import os
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")

ckpt_path = os.path.join(result.checkpoint.path, "checkpoint.ckpt")

dolly = DollyV2Model.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))

nlp_pipeline = pipeline(
    task="text-generation", 
    model=dolly.model, 
    tokenizer=tokenizer, 
    device_map="auto"
)


# In[16]:


for prompt in ["This is", "I am", "Once more"]:
    print(nlp_pipeline(prompt, max_new_tokens=20, do_sample=True, pad_token_id=tokenizer.eos_token_id))


# References:
# - [PyTorch FSDP Tutorial](https://www.youtube.com/watch?v=8_k76AHu__s&list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT)
# - [Getting Started with Fully Sharded Data Parallel(FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#:~:text=FSDP%20is%20a%20type%20of,sizes%20for%20our%20training%20job.)
# - [Fully Sharded Data Parallel: faster AI training with fewer GPUs](https://engineering.fb.com/2021/07/15/open-source/fsdp/)
# - [Hugging Face: dolly-v2-7b Model Card](https://huggingface.co/databricks/dolly-v2-7b)
# - [Hugging Face: Handling big models for inference](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
