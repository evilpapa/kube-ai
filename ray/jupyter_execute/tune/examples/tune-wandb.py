#!/usr/bin/env python
# coding: utf-8

# # Using Weights & Biases with Tune
# 
# (tune-wandb-ref)=
# 
# [Weights & Biases](https://www.wandb.ai/) (Wandb) is a tool for experiment
# tracking, model optimizaton, and dataset versioning. It is very popular
# in the machine learning and data science community for its superb visualization
# tools.
# 
# ```{image} /images/wandb_logo_full.png
# :align: center
# :alt: Weights & Biases
# :height: 80px
# :target: https://www.wandb.ai/
# ```
# 
# Ray Tune currently offers two lightweight integrations for Weights & Biases.
# One is the {ref}`WandbLoggerCallback <air-wandb-logger>`, which automatically logs
# metrics reported to Tune to the Wandb API.
# 
# The other one is the {ref}`setup_wandb() <air-wandb-setup>` function, which can be
# used with the function API. It automatically
# initializes the Wandb API with Tune's training information. You can just use the
# Wandb API like you would normally do, e.g. using `wandb.log()` to log your training
# process.
# 
# ```{contents}
# :backlinks: none
# :local: true
# ```
# 
# ## Running A Weights & Biases Example
# 
# In the following example we're going to use both of the above methods, namely the `WandbLoggerCallback` and
# the `setup_wandb` function to log metrics.
# 
# As the very first step, make sure you're logged in into wandb on all machines you're running your training on:
# 
#     wandb login
# 
# We can then start with a few crucial imports:

# In[1]:


import numpy as np

import ray
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb


# Next, let's define an easy `train_function` function (a Tune `Trainable`) that reports a random loss to Tune.
# The objective function itself is not important for this example, since we want to focus on the Weights & Biases
# integration primarily.

# In[2]:


def train_function(config):
    for i in range(30):
        loss = config["mean"] + config["sd"] * np.random.randn()
        train.report({"loss": loss})


# You can define a
# simple grid-search Tune run using the `WandbLoggerCallback` as follows:

# In[3]:


def tune_with_callback():
    """Example for using a WandbLoggerCallback with the function API"""
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        run_config=train.RunConfig(
            callbacks=[WandbLoggerCallback(project="Wandb_example")]
        ),
        param_space={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
        },
    )
    tuner.fit()


# To use the `setup_wandb` utility, you simply call this function in your objective.
# Note that we also use `wandb.log(...)` to log the `loss` to Weights & Biases as a dictionary.
# Otherwise, this version of our objective is identical to its original.

# In[4]:


def train_function_wandb(config):
    wandb = setup_wandb(config, project="Wandb_example")

    for i in range(30):
        loss = config["mean"] + config["sd"] * np.random.randn()
        train.report({"loss": loss})
        wandb.log(dict(loss=loss))


# With the `train_function_wandb` defined, your Tune experiment will set up `wandb` in each trial once it starts!

# In[5]:


def tune_with_setup():
    """Example for using the setup_wandb utility with the function API"""
    tuner = tune.Tuner(
        train_function_wandb,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        param_space={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
        },
    )
    tuner.fit()


# Finally, you can also define a class-based Tune `Trainable` by using the `setup_wandb` in the `setup()` method and storing the run object as an attribute. Please note that with the class trainable, you have to pass the trial id, name, and group separately:

# In[6]:


class WandbTrainable(tune.Trainable):
    def setup(self, config):
        self.wandb = setup_wandb(
            config,
            trial_id=self.trial_id,
            trial_name=self.trial_name,
            group="Example",
            project="Wandb_example",
        )

    def step(self):
        for i in range(30):
            loss = self.config["mean"] + self.config["sd"] * np.random.randn()
            self.wandb.log({"loss": loss})
        return {"loss": loss, "done": True}

    def save_checkpoint(self, checkpoint_dir: str):
        pass

    def load_checkpoint(self, checkpoint_dir: str):
        pass


# Running Tune with this `WandbTrainable` works exactly the same as with the function API.
# The below `tune_trainable` function differs from `tune_decorated` above only in the first argument we pass to
# `Tuner()`:

# In[7]:


def tune_trainable():
    """Example for using a WandTrainableMixin with the class API"""
    tuner = tune.Tuner(
        WandbTrainable,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        param_space={
            "mean": tune.grid_search([1, 2, 3, 4, 5]),
            "sd": tune.uniform(0.2, 0.8),
        },
    )
    results = tuner.fit()

    return results.get_best_result().config


# Since you may not have an API key for Wandb, we can _mock_ the Wandb logger and test all three of our training
# functions as follows.
# If you are logged in into wandb, you can set `mock_api = False` to actually upload your results to Weights & Biases.

# In[8]:


import os

mock_api = True

if mock_api:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_API_KEY", "abcd")
    ray.init(
        runtime_env={"env_vars": {"WANDB_MODE": "disabled", "WANDB_API_KEY": "abcd"}}
    )

tune_with_callback()
tune_with_setup()
tune_trainable()


# This completes our Tune and Wandb walk-through.
# In the following sections you can find more details on the API of the Tune-Wandb integration.
# 
# ## Tune Wandb API Reference
# 
# ### WandbLoggerCallback
# 
# (air-wandb-logger)=
# 
# ```{eval-rst}
# .. autoclass:: ray.air.integrations.wandb.WandbLoggerCallback
#    :noindex:
# ```
# 
# ### setup_wandb
# 
# (air-wandb-setup)=
# 
# ```{eval-rst}
# .. autofunction:: ray.air.integrations.wandb.setup_wandb
#    :noindex:
# ```
