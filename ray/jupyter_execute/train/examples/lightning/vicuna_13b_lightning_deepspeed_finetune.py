#!/usr/bin/env python
# coding: utf-8

# (vicuna_lightning_deepspeed_finetuning)=
# 
# # Fine-tune `vicuna-13b` with Lightning and DeepSpeed
# 
# In this example, we will demonstrate how to perform full fine-tuning for a [`vicuna-13b-v1.3`](https://huggingface.co/lmsys/vicuna-13b-v1.3) model using Ray Train PyTorch Lightning integrations with the DeepSpeed ZeRO-3 strategy.
# 
# - [DeepSpeed](<https://github.com/microsoft/DeepSpeed>) is an open-source deep learning optimization library for PyTorch. It's designed to reduce computing power and memory usage, and to train large distributed models by leveraging state-of-the-art innovations like ZeRO, 3D-Parallelism, DeepSpeed-MoE, and ZeRO-Infinity. 
# - PyTorch Lightning offers a [DeepSpeed integration](https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.strategies.DeepSpeedStrategy.html), which provides a simple interface to configure the knobs for DeepSpeed and automatically trigger your training process with the DeepSpeed Engine.
# - {class}`Ray TorchTrainer <ray.train.torch.TorchTrainer>` allows you to easily scale your PyTorch Lightning job across multiple nodes in a Ray cluster, without worrying about the underlying cluster management, autoscaling, and distributed process group settings.
# 
# Our demo aims to illustrate how these three tools can be combined effectively to finetune the Vicuna-13B model, leveraging the strengths of each to create an efficient and high-performance deep learning solution.
# 

# ```{note}
# This is an advanced example of Large Language Model fine-tuning with Ray Train. If you're a beginner or new to the concepts of Ray Train and our Lightning integrations, it would be beneficial to first explore the introductory documentation below to build a foundational understanding. 
# - [Ray Train Key Concepts](train-key-concepts) 
# - [Ray Data Key Concepts](data_key_concepts)
# - {ref}`[Basic] Image Classification with PyTorch Lightning and Ray Train <lightning_mnist_example>`
# - {ref}`[Intermediate] Fine-tuning Lightning Modules with with Ray Data <lightning_advanced_example>`
# ```
# 

# ## Cluster Setting
# 
# 
# ### Compute instances
# In this example, we set up a Ray cluster on AWS with the following settings:
# 
# |  | num | instance type | GPU per node | GPU Memory | CPU Memory |
# |-|-|-|-|-|-|
# |Head node|1|g5.16xlarge|1 x A10G | 24 GB | 256 GB|
# |Worker node|15|g5.4xlarge|1 x A10G | 24 GB | 64 GB|
# 
# ```{note}
# In this example, we used 16 A10G GPUs for model training and tuned the DeepSpeed configurations for this setup. If you have a different cluster setup or GPUs with lower memory capacities, you may need to modify the DeepSpeed configurations and batch size to fit the model into the GPUs.
# ```
# 
# ```{tip}
# We selected a GPU instance with additional CPU memory for the head node to demonstrate single-node offline inference. If you are training only, you can still opt for the g5.4xlarge instance for the head node.
# ```
# 
# 
# ### Cloud Storage
# 
# Additionally, since the checkpoint size for this 13B parameter model can be large (~140GB), we choose to store the checkpoints in AWS S3. Thanks to the newly introduced distributed checkpointing feature in Ray 2.5, each worker can upload its own shards individually to the S3 bucket, greatly reducing the latency and network traffic of checkpoint syncing.
# 
# ### Local Storage
# To demonstrate offline inference, we need to download and consolidate the model checkpoint onto the head node. This action requires around 200GB disk storage. Therefore, we mounted the NVMe SSD provided by g5 instances at `/dev/nvme1n1` to `/mnt/local_storage`, and we will save the checkpoints in this folder.
# 
# For more details, see [Amazon EBS and NVMe on Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/nvme-ebs-volumes.html).
# 

# ## Setup Ray Environment
# 
# We define a runtime environment to ensure that the Ray workers have access to all necessary packages. If you have already included these dependencies in your Docker image or installed them on each node, you can ignore the `runtime_env` argument.
# 
# ```{note}
# Note that the codebases of `transformers`, `accelerate`, and `deepspeed` are all rapidly changing, so we have pinned the package versions here to ensure testing stability. You can try other version combinations and feel free to report any issues you encounter.
# ```

# In[ ]:


# TODO(@justinvyu): Remove it
import os
os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"] = "1"


# In[ ]:


import ray

NUM_WORKERS = 16
BATCH_SIZE_PER_WORKER = 8
MODEL_NAME = "lmsys/vicuna-13b-v1.3"

ray.init(
    runtime_env={
        "pip": [
            "datasets==2.13.1",
            "torch>=1.13.0",
            "deepspeed==0.9.4",
            "accelerate==0.20.3",
            "transformers==4.30.2",
            "pytorch_lightning==2.0.3",
        ],
        "env_vars": {"RAY_AIR_NEW_PERSISTENCE_MODE": "1"} # TODO(@justinvyu): Remove it
    }
)


# ## Load and preprocess datasets
# 
# We were impressed by LLM's ability of zero-shot text-generation, while some LLMs may not perform well in code generation due to the lack of code in the training corpus. The CMU [CoNaLa](https://conala-corpus.github.io/)(The Code/Natural Language Challenge) was designed to test systems for generating program snippets from natural language. Each data record contains an intent sentence and a one-line code snippet. The goal is to fine-tune the Vicuna model on this dataset, enabling the model to generate correct and runnable code snippets, thereby achieving natural language intent. Here are some examples:
# 
# | intent | code snippet |
# | - | - |
# | "convert a list of integers into a single integer" | `r = int(''.join(map(str, x)))`|
# | "normalize a pandas dataframe `df` by row" | `df.div(df.sum(axis=1), axis=0)` | 
# | "Convert string '03:55' into datetime.time object" | `datetime.datetime.strptime('03:55', '%H:%M').time()` |
# 
# The CoNaLa team has released a dataset crawled from Stack Overflow, automatically filtered, then curated by annotators, split into 2379 training and 500 test examples. In addition, they also included an automatically-mined dataset with 600k examples. In this demo, we take all the curated data and the top 5000 mined data for fine-tuning.

# Here we preprocess the CoNaLa dataset with Ray Data. You can also use HuggingFace Datasets and pass it directly to `LightningConfigBuilder.fit_params()`.

# In[4]:


import re
import ray
import json
from transformers import AutoTokenizer
from datasets import concatenate_datasets, load_dataset

# Combine the curated dataset and automatically-mined dataset
hf_dataset_curated = load_dataset("neulab/conala")
hf_dataset_mined = load_dataset("neulab/conala", "mined", split="train[:5000]")
hf_dataset_merged = concatenate_datasets(
    [hf_dataset_curated["train"], hf_dataset_mined]
)
print(hf_dataset_merged)

# Convert it into Ray Dataset
ray_ds = ray.data.from_huggingface(hf_dataset_merged)

# Build a prompt template for Vicuna-13b model
PROMPT_TEMPLATE = "Intent: {intent}\nOne-line code snippet: {snippet}"


def fill_prompt(batch):
    batch["input_sentence"] = batch.apply(
        lambda row: PROMPT_TEMPLATE.format(
            intent=row["rewritten_intent"]
            if row["rewritten_intent"]
            else row["intent"],
            snippet=f"`{row['snippet']}`",
        )
        + "</s>",
        axis=1,
    )
    return batch[["input_sentence"]]


# Tokenize input sentences to tensors
def tokenize(batch):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="left", use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    ret = tokenizer(
        list(batch["input_sentence"]),
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="np",
    )
    ret["labels"] = ret["input_ids"].copy()
    return dict(ret)

# Preprocess train dataset
processed_ds = ray_ds.map_batches(fill_prompt, batch_format="pandas").map_batches(tokenize, batch_format="pandas")


# In[ ]:


# To accelerate release tests
processed_ds = processed_ds.limit(16 * 8 * 16)  # each worker has 16 batches


# ## Define a Lightning Module
# 
# Here we load the pre-trained model weights from HuggingFace Model Hub, and wrap them into `pl.LightningModule`. We adopted the efficient model initialization techniques introduced in [Lightning-transformers](https://github.com/Lightning-Universe/lightning-transformers) to avoid unnecessary full weights loading.

# In[5]:


import torch
import transformers
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.ops.adam import DeepSpeedCPUAdam


class ZeRO3Config:
    def __init__(self, pl_module):
        self.config = pl_module.trainer.strategy.config

    def __call__(self, *args, **kwargs):
        return self

    def is_zero3(self) -> bool:
        return True


def enable_transformers_pretrained_deepspeed_sharding(
    pl_module: "pl.LightningModule",
) -> None:
    transformers.deepspeed._hf_deepspeed_config_weak_ref = ZeRO3Config(pl_module)


class Vicuna13BModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Enable tf32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True

    def setup(self, stage) -> None:
        # Defer model initialization to inject deepspeed configs to HF.
        # During initialization, HF transformers can immediately partition 
        # the model across all gpus avoid the overhead in time and memory 
        # copying it on CPU or each GPU first.
        enable_transformers_pretrained_deepspeed_sharding(self)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        if self.global_rank == 0:
            print("DeepSpeed Configs: ", self.trainer.strategy.config)
            print("Model Archetecture: ", self.model)

    def forward(self, batch):
        outputs = self.model(
            batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=2e-5, weight_decay=0.01)


# ## DeepSpeed Configurations
# 
# Before training, let's calculate the memory usage of finetuning a `vicuna-13b` model. Assume we are using FP16 mixed-precision training, and the optimizer is Adam with FP32 states.
# 
# - Model parameters: 13(billion parameters) * 2(FP16) ≈ 26GB
# - Optimizer states: 13(billion parameters)  * 2(momentums per param) * 4 (FP32) ≈ 52GB
# 
# As we can see, the model parameters themselves require 26GB, which cannot fit in a single A10G GPU, let alone the activations and optimizers states. Here, we use ZeRO stage-3 to partition the model, gradients, and optimizer states across 16 nodes. Additionally, we employ optimizer CPU offloading to reduce GRAM usage and increase throughput with larger batch sizes. We also disabled parameter offloading and activation checkpointing to improve the training speed.
# 
# Regarding other knobs such as `reduce_bucket_size`, `stage3_prefetch_bucket_size` and `stage3_param_persistence_threshold`, we kept them as the [default values in HuggingFace](https://huggingface.co/docs/transformers/main_classes/deepspeed#zero3-config). Feel free to further adjust them to speed up the training process.

# In[7]:


from transformers import AutoConfig

config = AutoConfig.from_pretrained(MODEL_NAME)
HIDDEN_SIZE = config.hidden_size

deepspeed_configs = {
    "zero_allow_untested_optimizer": True,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": HIDDEN_SIZE * HIDDEN_SIZE,
        "stage3_prefetch_bucket_size": 0.9 * HIDDEN_SIZE * HIDDEN_SIZE,
        "stage3_param_persistence_threshold": 10 * HIDDEN_SIZE,
    },
}


# ## Define your training function
# 
# Finally, define the training function that will be launched on multiple workers. The training function is generally the same as the pure pytorch Lightning training code, with additional Ray Train utilities:
# 
# - {class}`~ray.train.lightning.RayDeepSpeedStrategy`: Same argument list as Lightning DeepSpeedStrategy but integrated with Ray Train.
# - {class}`~ray.train.lightning.RayLightningEnvironment`: Lightning environments for Ray cluster.
# - {class}`~ray.train.lightning.RayTrainReportCallback`: On each epoch end, it reports the checkpoint from each worker to the ray train (distributed checkpointing).
# - {meth}`~ray.train.lightning.prepare_trainer`: Validate your lightning Trainer configurations.
# 
# For Ray Data ingestion, we fetched the preprocessed and sharded dataset with {meth}`~ray.train.get_dataset_shard`, and created a dataloader with {meth}`~ray.data.Dataset.iter_torch_batches`. It returns a custom iterator that replaces the Torch DataLoader.
# 

# In[ ]:


import ray.train
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    prepare_trainer,
    RayDeepSpeedStrategy, 
    RayLightningEnvironment, 
    RayTrainReportCallback
)


def train_func(config):
    """Training function for each worker."""

    # Unpack the `train_loop_config`
    max_epochs = config["max_epochs"]
    batch_size = config["batch_size"]
    accumulate_grad_batches = config["accumulate_grad_batches"]

    model = Vicuna13BModel()
    
    # Prepare Ray Data Ingestion
    train_ds = ray.train.get_dataset_shard("train")
    train_dataloader = train_ds.iter_torch_batches(batch_size=batch_size)
    
    pl_trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDeepSpeedStrategy(config=deepspeed_configs),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        enable_checkpointing=False, # RayTrainReportCallback will save the checkpoints
        max_epochs=max_epochs,
        precision="bf16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
    )
    pl_trainer = prepare_trainer(pl_trainer)

    pl_trainer.fit(model, train_dataloaders=train_dataloader)
    

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={
        "max_epochs": 1,
        "batch_size": BATCH_SIZE_PER_WORKER,
        "accumulate_grad_batches": 2
    },
    run_config=RunConfig(
        name="vicuna-13b-finetune",
        storage_path="s3://anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7/air-release-tests",
        checkpoint_config=CheckpointConfig(num_to_keep=1),
    ),
    scaling_config=ScalingConfig(
        num_workers=NUM_WORKERS,
        use_gpu=True,
        resources_per_worker={"CPU": 15, "GPU": 1},
    ),
    datasets={"train": processed_ds},
)


# ## Model Fine-tuning
# 
# Once everything is configured in TorchTrainer, training becomes easy. Simply call `trainer.fit()`, and your workload will be scaled to the Ray cluster, initiating ZeRO-3 parallel training.

# In[9]:


result = trainer.fit()


# In summary:
# - Training takes: 36:06 = 2166s
# - Training + initialization + checkpointing takes 2473s
# 
# Model initialization and checkpoint synchronization took 307 seconds. It will be amortized as you have larger datasets and take more time to train.

# In[10]:


result


# ## LLM Inference
# 
# Now, it's time to play with our fine-tuned Vicuna code generator!

# ### Download and Process your checkpoints
# 
# First, download the checkpoints to your local machine using the AWS CLI.
# 
# Note that adding the following configurations can significantly increase the syncing throughput compared to the default configurations. On a g5 instance with NVME SSD, the download speed improved from `200MB/s` to around `1.5GB/s`.

# In[11]:


get_ipython().system('awsv2 configure set s3.max_concurrent_requests 32')
get_ipython().system('awsv2 configure set default.s3.preferred_transfer_client crt')
get_ipython().system('awsv2 configure set default.s3.target_bandwidth 100Gb/s')
get_ipython().system('awsv2 configure set default.s3.multipart_chunksize 8MB')


# In[ ]:


import os

os.system(f"awsv2 s3 sync s3://{result.checkpoint.path} /mnt/local_storage")


# The deepspeed ZeRO-3 checkpoint is a directory containing of k shards (k=16 in our case).
# 
# - `zero_pp_rank_k_mp_rank_00_model_states.pt`: contains the model parameter skeleton of shard k.
# - `bf16_zero_pp_rank_k_mp_rank_00_optim_states.pt`: contains the actual flattened model parameters and optimizer states of shard k.
# 
# Next, we removed the optimizer states and consolidate the checkpoint into a single binary file using DeepSpeed utilities. Also, since we wrapped vicuna-13b within a `LightningModule`, we need to remove the prefix `_forward_module.model.model` so that we can directly load the checkpoint into a HF vicuna model.

# In[14]:


import os
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

def extract_fp32_ckpt_from_zero(zero_ckpt_dir):
    state_dict = get_fp32_state_dict_from_zero_checkpoint(zero_ckpt_dir)
    vicuna_state_dict = {
        k.replace("_forward_module.model.", ""): v for k, v in state_dict.items()
    }
    torch.save(vicuna_state_dict, os.path.join(zero_ckpt_dir, "full_model.pt"))


full_model_ckpt_path = "/mnt/local_storage/checkpoint.ckpt/full_model.pt"
extract_fp32_ckpt_from_zero("/mnt/local_storage/checkpoint.ckpt")


# ### Initialize Generation Pipeline
# 
# Here, we leverage the Accelerate library to efficiently load the model onto a suitable device(GPU and CPU) and generate a HF text generation pipeline. 
# 
# - Initialize an empty model on metadevice
# - Create valid device mappings for the vicuna-13b model
# - Load and distribute model weights to target devices
# 
# This ensures that only 1x model size of RAM is used for model initialization.

# In[ ]:


import torch
import ray
import pytorch_lightning as pl
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)

# Initialize a model on meta device
with init_empty_weights():
    config = AutoConfig.from_pretrained(MODEL_NAME)
    meta_model = AutoModelForCausalLM.from_config(config)
meta_model.tie_weights()

# Define the device mapping
device_map = infer_auto_device_map(
    meta_model,
    max_memory={0: "15GB", "cpu": "60GB"},
    no_split_module_classes=["LlamaDecoderLayer"],
)

# Load the model parameters
model = load_checkpoint_and_dispatch(
    meta_model,
    checkpoint=full_model_ckpt_path,
    device_map=device_map,
)


# In[ ]:


from transformers import pipeline

generator = pipeline(
    "text-generation",
    model=model,
    device_map=device_map,
    tokenizer=AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="left", use_fast=False
    ),
)


# ### Case Study
# 
# We took 3 examples from the CoNaLa's test split for demo:

# In[34]:


testcases = [
    {
        "intent": "replace white spaces in colunm 'col' of dataframe `df` with '_'",
    },
    {
        "intent": "search for occurrences of regex pattern '>.*<' in xml string `line`",
    },
    {
        "intent": "send a signal `signal.SIGUSR1` to the current process",
    },
]


# Let's begin by examining the generated outputs without fine-tuning. In this case study, we utilize [Aviary Explorer](https://aviary.anyscale.com), an open-source multi-LLM serving platform supported by Ray and Anyscale. You can easily select from a variety of open-source LLMs and compare their generation quality, cost, latency, and many other metrics.
# 
# We constructed a prompt in a zero-shot learning manner and feed it into 3 OSS LLMs.
# 
# ![](https://user-images.githubusercontent.com/26745457/250704232-65a20f1b-6752-4d6c-bba1-8296a373162f.png)
# 
# 
# - `vicuna-13b-v1.3` begins to speak Chinese.
# - `mpt-7b-chat` generates a reasonable code snippet, but with multiple lines.
# - `falcon-7b-sft` generates a one line snippet, but it doesn't seem to work.
# 
# As we can see, none of them generate a satisfactory code snippet. 
# 
# Now let's check the performance of our fine-tuned `vicuna-13b-v1.3` model:

# In[35]:


for case in testcases:
    prompt = PROMPT_TEMPLATE.format(intent=case["intent"], snippet="")
    output = generator(prompt, max_new_tokens=30, do_sample=True)
    print(output[0]["generated_text"])


# ### Test the Generated Code Snippets
# 
# The generated code snippets look pretty reasonable. The results covered Pandas operations, regular expressions, and Linux commands. Let's test them one by one.

# In[33]:


import pandas as pd

df = pd.DataFrame.from_dict({"col": ["abc def ghi", " 12 3 456", "     "]})
print("Before\n", df)

df["col"] = df["col"].str.replace(" ", "_")
print("After\n", df)


# In[47]:


import re

line = """
<bookstore>
  <book category="fiction">
    <title>The Great Gatsby</title>
    <author>F. Scott Fitzgerald</author>
    <year>1925</year>
  </book>
  <book category="non-fiction">
    <title>Sapiens: A Brief History of Humankind</title>
    <author>Yuval Noah Harari</author>
    <year>2011</year>
  </book>
</bookstore>
"""
re.findall(">.*<", line)


# Finally, let's hand it over to LLM and let it wrap up the demo:

# In[ ]:


import os, signal

os.kill(os.getpid(), signal.SIGUSR1)  # Terminate the current process~


# ## References:
# 
# - [CoNaLa: The Code/Natural Language Challenge](https://conala-corpus.github.io/)
# - [HuggingFace: DeepSpeed Integration](https://huggingface.co/docs/transformers/main_classes/deepspeed#deepspeed-integration)
# - [HuggingFace: Handling big models for inference](https://huggingface.co/docs/accelerate/main/usage_guides/big_modeling)
# - [Lightning Transformers: DeepSpeed Training with Big Transformer Models](https://lightning-transformers.readthedocs.io/en/latest/)
# - [Aviary: Open Source Multi-LLM Serving](https://www.anyscale.com/blog/announcing-aviary-open-source-multi-llm-serving-solution)
# - Rajbhandari, S., Rasley, J., et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
# - Zheng, L., Chiang, W-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
# 
# 
# 
