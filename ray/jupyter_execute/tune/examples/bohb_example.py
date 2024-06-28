#!/usr/bin/env python
# coding: utf-8

# # Running Tune experiments with BOHB
# 
# In this tutorial we introduce BOHB, while running a simple Ray Tune experiment.
# Tune’s Search Algorithms integrate with BOHB and, as a result,
# allow you to seamlessly scale up a BOHB optimization
# process - without sacrificing performance.
# 
# Bayesian Optimization HyperBand (BOHB) combines the benefits of Bayesian optimization
# together with Bandit-based methods (e.g. HyperBand). BOHB does not rely on
# the gradient of the objective function,
# but instead, learns from samples of the search space.
# It is suitable for optimizing functions that are non-differentiable,
# with many local minima, or even unknown but only testable.
# Therefore, this approach belongs to the domain of
# "derivative-free optimization" and "black-box optimization".
# 
# In this example we minimize a simple objective to briefly demonstrate the usage of
# BOHB with Ray Tune via `BOHBSearch`. It's useful to keep in mind that despite
# the emphasis on machine learning experiments, Ray Tune optimizes any implicit
# or explicit objective. Here we assume `ConfigSpace==0.4.18` and `hpbandster==0.7.4`
# libraries are installed. To learn more, please refer to the
# [BOHB website](https://github.com/automl/HpBandSter).

# In[1]:


# !pip install ray[tune]
get_ipython().system('pip install ConfigSpace==0.4.18')
get_ipython().system('pip install hpbandster==0.7.4')


# Click below to see all the imports we need for this example.
# You can also launch directly into a Binder instance to run this notebook yourself.
# Just click on the rocket symbol at the top of the navigation.

# In[2]:


import tempfile
import time
from pathlib import Path

import ray
from ray import train, tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS


# Let's start by defining a simple evaluation function.
# We artificially sleep for a bit (`0.1` seconds) to simulate a long-running ML experiment.
# This setup assumes that we're running multiple `step`s of an experiment and try to tune
# two hyperparameters, namely `width` and `height`, and `activation`.

# In[3]:


def evaluate(step, width, height, activation):
    time.sleep(0.1)
    activation_boost = 10 if activation=="relu" else 1
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost


# Next, our `objective` function takes a Tune `config`, evaluates the `score` of your
# experiment in a training loop, and uses `train.report` to report the `score` back to Tune.
# 
# BOHB will interrupt our trials often, so we also need to {ref}`save and restore checkpoints <train-checkpointing>`.

# In[4]:


def objective(config):
    start = 0
    if train.get_checkpoint():
        with train.get_checkpoint().as_directory() as checkpoint_dir:
            start = int((Path(checkpoint_dir) / "data.ckpt").read_text())

    for step in range(start, config["steps"]):
        score = evaluate(step, config["width"], config["height"], config["activation"])
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            (Path(checkpoint_dir) / "data.ckpt").write_text(str(step))
            train.report(
                {"iterations": step, "mean_loss": score},
                checkpoint=train.Checkpoint.from_directory(checkpoint_dir)
            )


# In[ ]:


ray.init(configure_logging=False)


# Next we define a search space. The critical assumption is that the optimal
# hyperparameters live within this space. Yet, if the space is very large,
# then those hyperparameters may be difficult to find in a short amount of time.

# In[6]:


search_space = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu", "tanh"]),
}


# Next we define the search algorithm built from `TuneBOHB`, constrained
# to a maximum of `4` concurrent trials with a `ConcurrencyLimiter`.
# Below `algo` will take care of the BO (Bayesian optimization) part of BOHB,
# while scheduler will take care the HB (HyperBand) part.

# In[7]:


algo = TuneBOHB()
algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=100,
    reduction_factor=4,
    stop_last_trials=False,
)


# The number of samples is the number of hyperparameter combinations
# that will be tried out. This Tune run is set to `1000` samples.
# (you can decrease this if it takes too long on your machine).

# In[8]:


num_samples = 1000


# In[9]:


num_samples = 10


# Finally, we run the experiment to `min`imize the "mean_loss" of the `objective`
# by searching within `"steps": 100` via `algo`, `num_samples` times. This previous
# sentence is fully characterizes the search problem we aim to solve.
# With this in mind, notice how efficient it is to execute `tuner.fit()`.

# In[10]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=num_samples,
    ),
    run_config=train.RunConfig(
        name="bohb_exp",
        stop={"training_iteration": 100},
    ),
    param_space=search_space,
)
results = tuner.fit()


# Here are the hyperparameters found to minimize the mean loss of the defined objective.

# In[11]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# ## Optional: Passing the search space via the TuneBOHB algorithm
# 
# We can define the hyperparameter search space using `ConfigSpace`,
# which is the format accepted by BOHB.

# In[12]:


config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(
    CS.Constant("steps", 100)
)
config_space.add_hyperparameter(
    CS.UniformFloatHyperparameter("width", lower=0, upper=20)
)
config_space.add_hyperparameter(
    CS.UniformFloatHyperparameter("height", lower=-100, upper=100)
)
config_space.add_hyperparameter(
    CS.CategoricalHyperparameter(
        "activation", choices=["relu", "tanh"]
    )
)


# In[13]:


# As we are passing config space directly to the searcher,
# we need to define metric and mode in it as well, in addition
# to Tuner()
algo = TuneBOHB(
    space=config_space,
    metric="mean_loss",
    mode="max",
)
algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)
scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=100,
    reduction_factor=4,
    stop_last_trials=False,
)


# In[17]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=num_samples,
    ),
    run_config=train.RunConfig(
        name="bohb_exp_2",
        stop={"training_iteration": 100},
    ),
)
results = tuner.fit()


# Here again are the hyperparameters found to minimize the mean loss of the
# defined objective.

# In[18]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# In[19]:


ray.shutdown()

