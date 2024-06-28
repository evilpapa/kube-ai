#!/usr/bin/env python
# coding: utf-8

# # Running Tune experiments with BayesOpt
# 
# In this tutorial we introduce BayesOpt, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with BayesOpt and, as a result, allow you to seamlessly scale up a BayesOpt optimization process - without sacrificing performance.
# 
# BayesOpt is a constrained global optimization package utilizing Bayesian inference on gaussian processes, where the emphasis is on finding the maximum value of an unknown function in as few iterations as possible. BayesOpt's techniques are particularly suited for optimization of high cost functions, situations where the balance between exploration and exploitation is important. Therefore BayesOpt falls in the domain of "derivative-free" and "black-box" optimization. In this example we minimize a simple objective to briefly demonstrate the usage of BayesOpt with Ray Tune via `BayesOptSearch`, including conditional search spaces. It's useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `bayesian-optimization==1.2.0` library is installed. To learn more, please refer to [BayesOpt website](https://github.com/fmfn/BayesianOptimization).

# In[1]:


# !pip install ray[tune]
get_ipython().system('pip install bayesian-optimization==1.2.0')


# Click below to see all the imports we need for this example.
# You can also launch directly into a Binder instance to run this notebook yourself.
# Just click on the rocket symbol at the top of the navigation.

# In[2]:


import time

import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch


# Let's start by defining a simple evaluation function.
# We artificially sleep for a bit (`0.1` seconds) to simulate a long-running ML experiment.
# This setup assumes that we're running multiple `step`s of an experiment and try to tune two hyperparameters,
# namely `width` and `height`.

# In[3]:


def evaluate(step, width, height):
    time.sleep(0.1)
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


# Next, our ``objective`` function takes a Tune ``config``, evaluates the `score` of your experiment in a training loop,
# and uses `train.report` to report the `score` back to Tune.

# In[4]:


def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"])
        train.report({"iterations": step, "mean_loss": score})


# In[ ]:


ray.init(configure_logging=False)


# Now we define the search algorithm built from `BayesOptSearch`, constrained  to a maximum of `4` concurrent trials with a `ConcurrencyLimiter`.

# In[6]:


algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
algo = ConcurrencyLimiter(algo, max_concurrent=4)


# The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples.
# (you can decrease this if it takes too long on your machine).

# In[7]:


num_samples = 1000


# In[8]:


# If 1000 samples take too long, you can reduce this number.
# We override this number here for our smoke tests.
num_samples = 10


# Next we define a search space. The critical assumption is that the optimal hyperparameters live within this space. Yet, if the space is very large, then those hyperparameters may be difficult to find in a short amount of time.

# In[9]:


search_space = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
}


# Finally, we run the experiment to `"min"`imize the "mean_loss" of the `objective` by searching `search_config` via `algo`, `num_samples` times. This previous sentence is fully characterizes the search problem we aim to solve. With this in mind, notice how efficient it is to execute `tuner.fit()`.

# In[10]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_space,
)
results = tuner.fit()


# Here are the hyperparamters found to minimize the mean loss of the defined objective.

# In[11]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# In[12]:


ray.shutdown()

