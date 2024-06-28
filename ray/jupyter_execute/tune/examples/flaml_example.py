#!/usr/bin/env python
# coding: utf-8

# # Running Tune experiments with BlendSearch and CFO
# 
# In this tutorial we introduce BlendSearch and CFO, while running a simple Ray Tune
# experiment. Tune’s Search Algorithms integrate with FLAML and, as a result, allow
# you to seamlessly scale up a BlendSearch and CFO optimization
# process - without sacrificing performance.
# 
# Fast Library for Automated Machine Learning & Tuning (FLAML) does not rely on the
# gradient of the objective function, but instead, learns from samples of the
# search space. It is suitable for optimizing functions that are non-differentiable,
# with many local minima, or even unknown but only testable. Therefore, it is
# necessarily belongs to the domain of "derivative-free optimization"
# and "black-box optimization".
# 
# FLAML has two primary algorithms: (1) Frugal Optimization for Cost-related
# Hyperparameters (CFO) begins with a low-cost initial point and gradually moves to
# a high-cost region as needed. It is a local search method that leverages randomized
# direct search method with an adaptive step-size and random restarts.
# As a local search method, it has an appealing provable convergence rate and bounded
# cost but may get trapped in suboptimal local minima. (2) Economical Hyperparameter
# Optimization With Blended Search Strategy (BlendSearch) combines CFO's local search
# with global search, making it less suspectable to local minima traps.
# It leverages the frugality of CFO and the space exploration ability of global search
# methods such as Bayesian optimization.
# 
# In this example we minimize a simple objective to briefly demonstrate the usage of
# FLAML with Ray Tune via `BlendSearch` and `CFO`. It's useful to keep in mind that
# despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit
# or explicit objective. Here we assume `flaml==1.1.1` and `optuna==2.8.0` libraries
# are installed. To learn more, please refer to
# the [FLAML website](https://github.com/microsoft/FLAML/tree/main/flaml/tune).
#   
# Click below to see all the imports we need for this example.
# You can also launch directly into a Binder instance to run this notebook yourself.
# Just click on the rocket symbol at the top of the navigation.

# In[1]:


import time

import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.flaml import BlendSearch, CFO


# Let's start by defining a simple evaluation function.
# We artificially sleep for a bit (`0.1` seconds) to simulate a long-running ML experiment.
# This setup assumes that we're running multiple `step`s of an experiment and try to
# tune three hyperparameters, namely `width` and `height`, and `activation`.

# In[2]:


def evaluate(step, width, height, activation):
    time.sleep(0.1)
    activation_boost = 10 if activation=="relu" else 1
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost


# Next, our `objective` function takes a Tune `config`, evaluates the `score` of your
# experiment in a training loop, and uses `train.report` to report the `score` back to Tune.

# In[3]:


def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"], config["activation"])
        train.report({"iterations": step, "mean_loss": score})


# In[ ]:


ray.init(configure_logging=False)


# ## Running Tune experiments with BlendSearch
# 
# This example demonstrates the usage of Economical Hyperparameter Optimization
# With Blended Search Strategy (BlendSearch) with Ray Tune.
# 
# Now we define the search algorithm built from `BlendSearch`, constrained to a
# maximum of `4` concurrent trials with a `ConcurrencyLimiter`.

# In[5]:


algo = BlendSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)


# The number of samples this Tune run is set to `1000`.
# (you can decrease this if it takes too long on your machine).

# In[6]:


num_samples = 1000


# In[7]:


# If 1000 samples take too long, you can reduce this number.
# We override this number here for our smoke tests.
num_samples = 10


# Next we define a search space. The critical assumption is that the optimal
# hyperparameters live within this space. Yet, if the space is very large, then those
# hyperparameters may be difficult to find in a short amount of time.

# In[8]:


search_config = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu, tanh"])
}


# Finally, we run the experiment to `"min"`imize the "mean_loss" of the `objective` by
# searching `search_config` via `algo`, `num_samples` times. This previous sentence is
# fully characterizes the search problem we aim to solve. With this in mind, observe
# how efficient it is to execute `tuner.fit()`.

# In[9]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_config,
)
results = tuner.fit()


# Here are the hyperparamters found to minimize the mean loss of the defined objective.

# In[10]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# ## Incorporating a time budget to the experiment
# 
# Define the time budget in seconds:

# In[11]:


time_budget_s = 30


# Similarly we define a search space, but this time we feed it as an argument to
# `BlendSearch` rather than `Tuner()`'s `param_space` argument.
# 
# We next define the time budget via `set_search_properties`.
# And once again include the `ConcurrencyLimiter`.

# In[12]:


algo = BlendSearch(
    metric="mean_loss",
    mode="min",
    space={
        "width": tune.uniform(0, 20),
        "height": tune.uniform(-100, 100),
        "activation": tune.choice(["relu", "tanh"]),
    },
)
algo.set_search_properties(config={"time_budget_s": time_budget_s})
algo = ConcurrencyLimiter(algo, max_concurrent=4)


# Now we run the experiment, this time with the `time_budget` included as an argument.
# Note: We allow for virtually infinite `num_samples` by passing `-1`, so that the
# experiment is stopped according to the time budget rather than a sample limit.

# In[13]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=-1,
        time_budget_s=time_budget_s,
    ),
    param_space={"steps": 100},
)
results = tuner.fit()


# In[14]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# ## Running Tune experiments with CFO
# 
# This example demonstrates the usage of Frugal Optimization for Cost-related
# Hyperparameters (CFO) with Ray Tune.
# 
# We now define the search algorithm as built from `CFO`, constrained to a maximum of `4`
# concurrent trials with a `ConcurrencyLimiter`.

# In[15]:


algo = CFO()
algo = ConcurrencyLimiter(algo, max_concurrent=4)


# The number of samples is the number of hyperparameter combinations that will be
# tried out. This Tune run is set to `1000` samples.
# (you can decrease this if it takes too long on your machine).

# In[16]:


num_samples = 1000


# In[17]:


# If 1000 samples take too long, you can reduce this number.
# We override this number here for our smoke tests.
num_samples = 10


# Next we define a search space. The critical assumption is that the optimal
# hyperparameters live within this space. Yet, if the space is very large, then
# those hyperparameters may be difficult to find in a short amount of time.

# In[18]:


search_config = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu, tanh"])
}


# Finally, we run the experiment to `"min"`imize the "mean_loss" of the `objective`
# by searching `search_config` via `algo`, `num_samples` times. This previous sentence
# is fully characterizes the search problem we aim to solve. With this in mind,
# notice how efficient it is to execute `tuner.fit()`.

# In[19]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_config,
)
results = tuner.fit()


# Here are the hyperparameters found to minimize the mean loss of the defined objective.

# In[20]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# In[21]:


ray.shutdown()

