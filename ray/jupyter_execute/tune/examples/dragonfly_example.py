#!/usr/bin/env python
# coding: utf-8

# # Running Tune experiments with Dragonfly
# 
# In this tutorial we introduce Dragonfly, while running a simple Ray Tune experiment.
# Tune’s Search Algorithms integrate with Dragonfly and, as a result,
# allow you to seamlessly scale up a Dragonfly optimization process - without
# sacrificing performance.
# 
# Dragonfly is an open source python library for scalable Bayesian optimization.
# Bayesian optimization is used optimizing black-box functions whose evaluations
# are usually expensive. Beyond vanilla optimization techniques,
# Dragonfly provides an array of tools to scale up Bayesian optimization to expensive
# large scale problems. These include features that are especially suited for high
# dimensional spaces (optimizing with a large number of variables), parallel evaluations
# in synchronous or asynchronous settings (conducting multiple evaluations in parallel),
# multi-fidelity optimization (using cheap approximations to speed up the optimization
# process), and multi-objective optimization (optimizing multiple functions
# simultaneously).
# 
# Bayesian optimization does not rely on the gradient of the objective function,
# but instead, learns from samples of the search space. It is suitable for optimizing
# functions that are non-differentiable, with many local minima, or even unknown but only
# testable. Therefore, it belongs to the domain of "derivative-free optimization" and
# "black-box optimization". In this example we minimize a simple objective to briefly
# demonstrate the usage of Dragonfly with Ray Tune via `DragonflySearch`. It's useful
# to keep in mind that despite the emphasis on machine learning experiments,
# Ray Tune optimizes any implicit or explicit objective. Here we assume
# `dragonfly-opt==0.1.6` library is installed. To learn more, please refer to
# the [Dragonfly website](https://dragonfly-opt.readthedocs.io/).

# In[ ]:


# !pip install ray[tune]
get_ipython().system('pip install dragonfly-opt==0.1.6')


# Click below to see all the imports we need for this example.
# You can also launch directly into a Binder instance to run this notebook yourself.
# Just click on the rocket symbol at the top of the navigation.

# In[ ]:


import numpy as np
import time

import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.dragonfly import DragonflySearch


# Let's start by defining a optimization problem.
# Suppose we want to figure out the proportions of water and several salts to add to an
# ionic solution with the goal of maximizing it's ability to conduct electricity.
# The objective here is explicit for demonstration, yet in practice they often come
# out of a black-box (e.g. a physical device measuring conductivity, or reporting the
# results of a long-running ML experiment). We artificially sleep for a bit
# (`0.02` seconds) to simulate a more typical experiment. This setup assumes that we're
# running multiple `step`s of an experiment and try to tune relative proportions of
# 4 ingredients-- these proportions should be considered as hyperparameters.
# Our `objective` function will take a Tune `config`, evaluates the `conductivity` of
# our experiment in a training loop,
# and uses `session.report` to report the `conductivity` back to Tune.

# In[ ]:


def objective(config):
    """
    Simplistic model of electrical conductivity with added Gaussian
    noise to simulate experimental noise.
    """
    for i in range(config["iterations"]):
        vol1 = config["LiNO3_vol"]  # LiNO3
        vol2 = config["Li2SO4_vol"]  # Li2SO4
        vol3 = config["NaClO4_vol"]  # NaClO4
        vol4 = 10 - (vol1 + vol2 + vol3)  # Water
        conductivity = vol1 + 0.1 * (vol2 + vol3) ** 2 + 2.3 * vol4 * (vol1 ** 1.5)
        conductivity += np.random.normal() * 0.01
        train.report({"timesteps_total": i, "objective": conductivity})
        time.sleep(0.02)


# Next we define a search space. The critical assumption is that the optimal
# hyperparameters live within this space. Yet, if the space is very large, then those
# hyperparameters may be difficult to find in a short amount of time.

# In[ ]:


search_space = {
    "iterations": 100,
    "LiNO3_vol": tune.uniform(0, 7),
    "Li2SO4_vol": tune.uniform(0, 7),
    "NaClO4_vol": tune.uniform(0, 7)
}


# In[ ]:


ray.init(configure_logging=False)


# Now we define the search algorithm from `DragonflySearch`  with `optimizer` and
# `domain` arguments specified in a common way. We also use `ConcurrencyLimiter`
# to constrain to 4 concurrent trials.

# In[ ]:


algo = DragonflySearch(
    optimizer="bandit",
    domain="euclidean",
)
algo = ConcurrencyLimiter(algo, max_concurrent=4)


# The number of samples is the number of hyperparameter combinations that will be
# tried out. This Tune run is set to `1000` samples.
# (you can decrease this if it takes too long on your machine).

# In[ ]:


num_samples = 100


# In[ ]:


# Reducing samples for smoke tests
num_samples = 10


# Finally, we run the experiment to `min`imize the `mean_loss` of the `objective` by
# searching `search_config` via `algo`, `num_samples` times. This previous sentence is
# fully characterizes the search problem we aim to solve. With this in mind,
# notice how efficient it is to execute `tuner.fit()`.

# In[ ]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="objective",
        mode="max",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_space,
)
results = tuner.fit()


# Below are the recommended relative proportions of water and each salt found to
# maximize conductivity in the ionic solution (according to the simple model):

# In[ ]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# In[ ]:


ray.shutdown()

