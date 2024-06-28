#!/usr/bin/env python
# coding: utf-8

# # Running Tune experiments with Nevergrad
# 
# In this tutorial we introduce Nevergrad, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with Nevergrad and, as a result, allow you to seamlessly scale up a Nevergrad optimization process - without sacrificing performance.
# 
# Nevergrad provides gradient/derivative-free optimization able to handle noise over the objective landscape, including evolutionary, bandit, and Bayesian optimization algorithms. Nevergrad internally supports search spaces which are continuous, discrete or a mixture of thereof. It also provides a library of functions on which to test the optimization algorithms and compare with other benchmarks.
# 
# In this example we minimize a simple objective to briefly demonstrate the usage of Nevergrad with Ray Tune via `NevergradSearch`. It's useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `nevergrad==0.4.3.post7` library is installed. To learn more, please refer to [Nevergrad website](https://github.com/facebookresearch/nevergrad).

# In[1]:


# !pip install ray[tune]
get_ipython().system('pip install nevergrad==0.4.3.post7')


# Click below to see all the imports we need for this example.
# You can also launch directly into a Binder instance to run this notebook yourself.
# Just click on the rocket symbol at the top of the navigation.

# In[2]:


import time

import ray
import nevergrad as ng
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.nevergrad import NevergradSearch


# Let's start by defining a simple evaluation function.
# We artificially sleep for a bit (`0.1` seconds) to simulate a long-running ML experiment.
# This setup assumes that we're running multiple `step`s of an experiment and try to tune two hyperparameters,
# namely `width` and `height`, and `activation`.

# In[3]:


def evaluate(step, width, height, activation):
    time.sleep(0.1)
    activation_boost = 10 if activation=="relu" else 1
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost


# Next, our `objective` function takes a Tune `config`, evaluates the `score` of your experiment in a training loop,
# and uses `train.report` to report the `score` back to Tune.

# In[4]:


def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"], config["activation"])
        train.report({"iterations": step, "mean_loss": score})


# In[ ]:


ray.init(configure_logging=False)


# Now we construct the hyperparameter search space using `ConfigSpace`

# Next we define the search algorithm built from `NevergradSearch`, constrained  to a maximum of `4` concurrent trials with a `ConcurrencyLimiter`. Here we use `ng.optimizers.OnePlusOne`, a simple evolutionary algorithm.

# In[6]:


algo = NevergradSearch(
    optimizer=ng.optimizers.OnePlusOne,
)
algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)


# The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples.
# (you can decrease this if it takes too long on your machine).

# In[7]:


num_samples = 1000


# In[8]:


# If 1000 samples take too long, you can reduce this number.
# We override this number here for our smoke tests.
num_samples = 10


# Finally, all that's left is to define a search space.

# In[9]:


search_config = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu, tanh"])
}


# Finally, we run the experiment to `"min"`imize the "mean_loss" of the `objective` by searching `search_space` via `algo`, `num_samples` times. This previous sentence is fully characterizes the search problem we aim to solve. With this in mind, observe how efficient it is to execute `tuner.fit()`.

# In[10]:


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

# In[11]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# ## Optional: passing the (hyper)parameter space into the search algorithm
# 
# We can also pass the search space into `NevergradSearch` using their designed format.

# In[12]:


space = ng.p.Dict(
    width=ng.p.Scalar(lower=0, upper=20),
    height=ng.p.Scalar(lower=-100, upper=100),
    activation=ng.p.Choice(choices=["relu", "tanh"])
)


# In[13]:


algo = NevergradSearch(
    optimizer=ng.optimizers.OnePlusOne,
    space=space,
    metric="mean_loss",
    mode="min"
)
algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)


# Again we run the experiment, this time with a less passed via the `config` and instead passed through `search_alg`.

# In[14]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
#         metric="mean_loss",
#         mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space={"steps": 100},
)
results = tuner.fit()


# Here are the hyperparamters found to minimize the mean loss of the defined objective. Note that we have to pass the metric and mode here because we don't set it in the TuneConfig.

# In[17]:


print("Best hyperparameters found were: ", results.get_best_result("mean_loss", "min").config)


# In[15]:


ray.shutdown()

