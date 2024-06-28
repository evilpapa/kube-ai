#!/usr/bin/env python
# coding: utf-8

# (train_transformers_glue_example)=
# 
# # Fine-tune a Hugging Face Transformers Model

# This notebook is based on an official Hugging Face example, [How to fine-tune a model on text classification](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb). This notebook shows the process of conversion from vanilla HF to Ray Train without changing the training logic unless necessary.
# 
# This notebook consists of the following steps:
# 1. [Set up Ray](#setup)
# 2. [Load the dataset](#load)
# 3. [Preprocess the dataset with Ray Data](#preprocess)
# 4. [Run the training with Ray Train](#train)
# 5. [Optionally, share the model with the community](#share)

# Uncomment and run the following line to install all the necessary dependencies. (This notebook is being tested with `transformers==4.19.1`.):

# In[1]:


#! pip install "datasets" "transformers>=4.19.0" "torch>=1.10.0" "mlflow"


# ## Set up Ray <a name="setup"></a>

# Use `ray.init()` to initialize a local cluster. By default, this cluster contains only the machine you are running this notebook on. You can also run this notebook on an [Anyscale](https://www.anyscale.com/) cluster.

# In[ ]:


from pprint import pprint
import ray

ray.init()


# Check the resources our cluster is composed of. If you are running this notebook on your local machine or Google Colab, you should see the number of CPU cores and GPUs available on the your machine.

# In[3]:


pprint(ray.cluster_resources())


# This notebook fine-tunes a [HF Transformers](https://github.com/huggingface/transformers) model for one of the text classification task of the [GLUE Benchmark](https://gluebenchmark.com/). It runs the training using Ray Train.
# 
# You can change these two variables to control whether the training, which happens later, uses CPUs or GPUs, and how many workers to spawn. Each worker claims one CPU or GPU. Make sure to not request more resources than the resources present. By default, the training runs with one GPU worker.

# In[4]:


use_gpu = True  # set this to False to run on CPUs
num_workers = 1  # set this to number of GPUs or CPUs you want to use


# ## Fine-tune a model on a text classification task

# The GLUE Benchmark is a group of nine classification tasks on sentences or pairs of sentences. To learn more, see the [original notebook](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb).
# 
# Each task has a name that is its acronym, with `mnli-mm` to indicate that it is a mismatched version of MNLI. Each one has the same training set as `mnli` but different validation and test sets.

# In[5]:


GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]


# This notebook runs on any of the tasks in the list above, with any model checkpoint from the [Model Hub](https://huggingface.co/models) as long as that model has a version with a classification head. Depending on the model and the GPU you are using, you might need to adjust the batch size to avoid out-of-memory errors. Set these three parameters, and the rest of the notebook should run smoothly:

# In[6]:


task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16


# ### Loading the dataset <a name="load"></a>

# Use the [HF Datasets](https://github.com/huggingface/datasets) library to download the data and get the metric to use for evaluation and to compare your model to the benchmark. You can do this comparison easily with the `load_dataset` and `load_metric` functions.
# 
# Apart from `mnli-mm` being special code, you can directly pass the task name to those functions.
# 
# Run the normal HF Datasets code to load the dataset from the Hub.

# In[7]:


from datasets import load_dataset

actual_task = "mnli" if task == "mnli-mm" else task
datasets = load_dataset("glue", actual_task)


# The `dataset` object itself is a [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation, and test set, with more keys for the mismatched validation and test set in the special case of `mnli`.

# ### Preprocessing the data with Ray Data <a name="preprocess"></a>

# Before you can feed these texts to the model, you need to preprocess them. Preprocess them with a HF Transformers' `Tokenizer`, which tokenizes the inputs, including converting the tokens to their corresponding IDs in the pretrained vocabulary, and puts them in a format the model expects. It also generates the other inputs that the model requires.
# 
# To do all of this preprocessing, instantiate your tokenizer with the `AutoTokenizer.from_pretrained` method, which ensures that you:
# 
# - Get a tokenizer that corresponds to the model architecture you want to use.
# - Download the vocabulary used when pretraining this specific checkpoint.

# In[8]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


# Pass `use_fast=True` to the preceding call to use one of the fast tokenizers, backed by Rust, from the HF Tokenizers library. These fast tokenizers are available for almost all models, but if you get an error with the previous call, remove the argument.

# To preprocess the dataset, you need the names of the columns containing the sentence(s). The following dictionary keeps track of the correspondence task to column names:

# In[9]:


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


# Instead of using HF Dataset objects directly, convert them to [Ray Data](data). Arrow tables back both of them, so the conversion is straightforward. Use the built-in {meth}`~ray.data.from_huggingface` function.

# In[10]:


import ray.data

ray_datasets = {
    "train": ray.data.from_huggingface(datasets["train"]),
    "validation": ray.data.from_huggingface(datasets["validation"]),
    "test": ray.data.from_huggingface(datasets["test"]),
}
ray_datasets


# You can then write the function that preprocesses the samples. Feed them to the `tokenizer` with the argument `truncation=True`. This configuration ensures that the `tokenizer` truncates and pads to the longest sequence in the batch, any input longer than what the model selected can handle.

# In[11]:


import numpy as np
from typing import Dict


# Tokenize input sentences
def collate_fn(examples: Dict[str, np.array]):
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        outputs = tokenizer(
            list(examples[sentence1_key]),
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
    else:
        outputs = tokenizer(
            list(examples[sentence1_key]),
            list(examples[sentence2_key]),
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

    outputs["labels"] = torch.LongTensor(examples["label"])

    # Move all input tensors to GPU
    for key, value in outputs.items():
        outputs[key] = value.cuda()

    return outputs


# ### Fine-tuning the model with Ray Train <a name="train"></a>

# Now that the data is ready, download the pretrained model and fine-tune it.
# 
# Because all of the tasks involve sentence classification, use the `AutoModelForSequenceClassification` class. For more specifics about each individual training component, see the [original notebook](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb). The original notebook uses the same tokenizer used to encode the dataset in this notebook's preceding example.
# 
# The main difference when using Ray Train is that you need to define the training logic as a function (`train_func`). You pass this [training function](train-overview-training-function) to the {class}`~ray.train.torch.TorchTrainer` to on every Ray worker. The training then proceeds using PyTorch DDP.
# 
# 
# ```{note}
# 
# Be sure to initialize the model, metric, and tokenizer within the function. Otherwise, you may encounter serialization errors.
# 
# ```

# In[12]:


import torch
import numpy as np

from datasets import load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import ray.train
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
metric_name = (
    "pearson"
    if task == "stsb"
    else "matthews_correlation"
    if task == "cola"
    else "accuracy"
)
model_name = model_checkpoint.split("/")[-1]
validation_key = (
    "validation_mismatched"
    if task == "mnli-mm"
    else "validation_matched"
    if task == "mnli"
    else "validation"
)
name = f"{model_name}-finetuned-{task}"

# Calculate the maximum steps per epoch based on the number of rows in the training dataset.
# Make sure to scale by the total number of training workers and the per device batch size.
max_steps_per_epoch = ray_datasets["train"].count() // (batch_size * num_workers)


def train_func(config):
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    metric = load_metric("glue", actual_task)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )

    train_ds = ray.train.get_dataset_shard("train")
    eval_ds = ray.train.get_dataset_shard("eval")

    train_ds_iterable = train_ds.iter_torch_batches(
        batch_size=batch_size, collate_fn=collate_fn
    )
    eval_ds_iterable = eval_ds.iter_torch_batches(
        batch_size=batch_size, collate_fn=collate_fn
    )

    print("max_steps_per_epoch: ", max_steps_per_epoch)

    args = TrainingArguments(
        name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=config.get("learning_rate", 2e-5),
        num_train_epochs=config.get("epochs", 2),
        weight_decay=config.get("weight_decay", 0.01),
        push_to_hub=False,
        max_steps=max_steps_per_epoch * config.get("epochs", 2),
        disable_tqdm=True,  # declutter the output a little
        no_cuda=not use_gpu,  # you need to explicitly set no_cuda if you want CPUs
        report_to="none",
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds_iterable,
        eval_dataset=eval_ds_iterable,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.add_callback(RayTrainReportCallback())

    trainer = prepare_trainer(trainer)

    print("Starting training")
    trainer.train()


# With your `train_func` complete, you can now instantiate the {class}`~ray.train.torch.TorchTrainer`. Aside from calling the function, set the `scaling_config`, which controls the amount of workers and resources used, and the `datasets` to use for training and evaluation.

# In[13]:


from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    datasets={
        "train": ray_datasets["train"],
        "eval": ray_datasets["validation"],
    },
    run_config=RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    ),
)


# Finally, call the `fit` method to start training with Ray Train. Save the `Result` object to a variable so you can access metrics and checkpoints.

# In[14]:


result = trainer.fit()


# You can use the returned `Result` object to access metrics and the Ray Train `Checkpoint` associated with the last iteration.

# In[15]:


result


# ### Tune hyperparameters with Ray Tune <a name="predict"></a>

# To tune any hyperparameters of the model, pass your `TorchTrainer` into a `Tuner` and define the search space.
# 
# You can also take advantage of the advanced search algorithms and schedulers from Ray Tune. This example uses an `ASHAScheduler` to aggresively terminate underperforming trials.

# In[23]:


from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers.async_hyperband import ASHAScheduler

tune_epochs = 4
tuner = Tuner(
    trainer,
    param_space={
        "train_loop_config": {
            "learning_rate": tune.grid_search([2e-5, 2e-4, 2e-3, 2e-2]),
            "epochs": tune_epochs,
        }
    },
    tune_config=tune.TuneConfig(
        metric="eval_loss",
        mode="min",
        num_samples=1,
        scheduler=ASHAScheduler(
            max_t=tune_epochs,
        ),
    ),
    run_config=RunConfig(
        name="tune_transformers",
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    ),
)


# In[24]:


tune_results = tuner.fit()


# View the results of the tuning run as a dataframe, and find the best result.

# In[25]:


tune_results.get_dataframe().sort_values("eval_loss")


# In[26]:


best_result = tune_results.get_best_result()


# ### Share the model <a name="share"></a>

# To share the model with the community, a few more steps follow.
# 
# You conducted the training on the Ray cluster, but want share the model from the local enviroment. This configuration allows you to easily authenticate.
# 
# First, store your authentication token from the Hugging Face website. Sign up [here](https://huggingface.co/join) if you haven't already. Then execute the following cell and input your username and password:

# In[ ]:


from huggingface_hub import notebook_login

notebook_login()


# Then you need to install Git-LFS. Uncomment the following instructions:

# In[21]:


# !apt install git-lfs


# Now, load the model and tokenizer locally, and recreate the HF Transformers `Trainer`:

# In[ ]:


from ray.train.huggingface import LegacyTransformersCheckpoint

checkpoint = LegacyTransformersCheckpoint.from_checkpoint(result.checkpoint)
hf_trainer = checkpoint.get_model(model=AutoModelForSequenceClassification)


# You can now upload the result of the training to the Hub. Execute this instruction:

# In[ ]:





# In[ ]:


hf_trainer.push_to_hub()


# You can now share this model. Others can load it with the identifier `"your-username/the-name-you-picked"`. For example:
# 
# ```python
# from transformers import AutoModelForSequenceClassification
# 
# model = AutoModelForSequenceClassification.from_pretrained("sgugger/my-awesome-model")
# ```

# ## See also
# 
# * {ref}`Ray Train Examples <train-examples>` for more use cases
# * {ref}`Ray Train User Guides <train-user-guides>` for how-to guides
# 
