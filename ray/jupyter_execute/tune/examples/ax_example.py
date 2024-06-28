#!/usr/bin/env python
# coding: utf-8

# # Running Tune experiments with AxSearch
# In this tutorial we introduce Ax, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with Ax and, as a result, allow you to seamlessly scale up a Ax optimization process - without sacrificing performance.
# 
# Ax is a platform for optimizing any kind of experiment, including machine learning experiments, A/B tests, and simulations. Ax can optimize discrete configurations (e.g., variants of an A/B test) using multi-armed bandit optimization, and continuous/ordered configurations (e.g. float/int parameters) using Bayesian optimization. Results of A/B tests and simulations with reinforcement learning agents often exhibit high amounts of noise. Ax supports state-of-the-art algorithms which work better than traditional Bayesian optimization in high-noise settings. Ax also supports multi-objective and constrained optimization which are common to real-world problems (e.g. improving load time without increasing data use). Ax belongs to the domain of  "derivative-free" and "black-box" optimization.
# 
# In this example we minimize a simple objective to briefly demonstrate the usage of AxSearch with Ray Tune via `AxSearch`. It's useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `ax-platform==0.2.4` library is installed withe python version >= 3.7. To learn more, please refer to the [Ax website](https://ax.dev/).

# In[1]:


# !pip install ray[tune]
get_ipython().system('pip install ax-platform==0.2.4')


# Click below to see all the imports we need for this example.
# You can also launch directly into a Binder instance to run this notebook yourself.
# Just click on the rocket symbol at the top of the navigation.

# In[2]:


import numpy as np
import time

import ray
from ray import train, tune
from ray.tune.search.ax import AxSearch


# Let's start by defining a classic benchmark for global optimization.
# The form here is explicit for demonstration, yet it is typically a black-box.
# We artificially sleep for a bit (`0.02` seconds) to simulate a long-running ML experiment.
# This setup assumes that we're running multiple `step`s of an experiment and try to tune 6-dimensions of the `x` hyperparameter.

# In[3]:


def landscape(x):
    """
    Hartmann 6D function containing 6 local minima.
    It is a classic benchmark for developing global optimization algorithms.
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = 0
        for k in range(6):
            t += A[j, k] * ((x[k] - P[j, k]) ** 2)
        y -= alpha_j * np.exp(-t)
    return y


# Next, our `objective` function takes a Tune `config`, evaluates the `landscape` of our experiment in a training loop,
# and uses `train.report` to report the `landscape` back to Tune.

# In[4]:


def objective(config):
    for i in range(config["iterations"]):
        x = np.array([config.get("x{}".format(i + 1)) for i in range(6)])
        train.report(
            {"timesteps_total": i, "landscape": landscape(x), "l2norm": np.sqrt((x ** 2).sum())}
        )
        time.sleep(0.02)


# Next we define a search space. The critical assumption is that the optimal hyperparamters live within this space. Yet, if the space is very large, then those hyperparamters may be difficult to find in a short amount of time.

# In[5]:


search_space = {
    "iterations":100,
    "x1": tune.uniform(0.0, 1.0),
    "x2": tune.uniform(0.0, 1.0),
    "x3": tune.uniform(0.0, 1.0),
    "x4": tune.uniform(0.0, 1.0),
    "x5": tune.uniform(0.0, 1.0),
    "x6": tune.uniform(0.0, 1.0)
}


# In[ ]:


ray.init(configure_logging=False)


# Now we define the search algorithm from `AxSearch`. If you want to constrain your parameters or even the space of outcomes, that can be easily done by passing the argumentsas below.

# In[7]:


algo = AxSearch(
    parameter_constraints=["x1 + x2 <= 2.0"],
    outcome_constraints=["l2norm <= 1.25"],
)


# We also use `ConcurrencyLimiter` to constrain to 4 concurrent trials. 

# In[8]:


algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)


# The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples.
# You can decrease this if it takes too long on your machine, or you can set a time limit easily through `stop` argument in the `train.RunConfig()` as we will show here.

# In[9]:


num_samples = 100
stop_timesteps = 200


# In[10]:


# Reducing samples for smoke tests
num_samples = 10


# Finally, we run the experiment to find the global minimum of the provided landscape (which contains 5 false minima). The argument to metric, `"landscape"`, is provided via the `objective` function's `session.report`. The experiment `"min"`imizes the "mean_loss" of the `landscape` by searching within `search_space` via `algo`, `num_samples` times or when `"timesteps_total": stop_timesteps`. This previous sentence is fully characterizes the search problem we aim to solve. With this in mind, notice how efficient it is to execute `tuner.fit()`.

# In[11]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="landscape",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    run_config=train.RunConfig(
        name="ax",
        stop={"timesteps_total": stop_timesteps}
    ),
    param_space=search_space,
)
results = tuner.fit()


# And now we have the hyperparameters found to minimize the mean loss.

# In[12]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# In[13]:


ray.shutdown()

