#!/usr/bin/env python
# coding: utf-8

# # Running Tune experiments with SigOpt
# 
# In this tutorial we introduce SigOpt, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with SigOpt and, as a result, allow you to seamlessly scale up a SigOpt optimization process - without sacrificing performance.
# 
# SigOpt is a model development platform with built in hyperparameter optimization algorithms. Their technology is closed source, but is designed for optimizing functions that are nondifferentiable, with many local minima, or even unknown but only testable. Therefore, SigOpt necessarily falls in the domain of "derivative-free optimization" and "black-box optimization". In this example we minimize a simple objective to briefly demonstrate the usage of SigOpt with Ray Tune via `SigOptSearch`. It's useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `sigopt==7.5.0` library is installed and an API key exists. To learn more and to obtain the necessary API key, refer to [SigOpt website](https://sigopt.com/). 
# 

# In[ ]:


# !pip install ray[tune]
get_ipython().system('pip install sigopt==7.5.0')


# Click below to see all the imports we need for this example.
# You can also launch directly into a Binder instance to run this notebook yourself.
# Just click on the rocket symbol at the top of the navigation.

# In[ ]:


import time
import os

import ray
import numpy as np
from ray import train, tune
from ray.tune.search.sigopt import SigOptSearch

if "SIGOPT_KEY" not in os.environ:
    raise ValueError(
        "SigOpt API Key not found. Please set the SIGOPT_KEY "
        "environment variable."
    )


# Let's start by defining a simple evaluation function.
# We artificially sleep for a bit (`0.1` seconds) to simulate a long-running ML experiment.
# This setup assumes that we're running multiple `step`s of an experiment and try to tune two hyperparameters,
# namely `width` and `height`, and `activation`.

# In[ ]:


def evaluate(step, width, height, activation):
    time.sleep(0.1)
    activation_boost = 10 if activation=="relu" else 1
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost


# Next, our ``objective`` function takes a Tune ``config``, evaluates the `score` of your experiment in a training loop,
# and uses `train.report` to report the `score` back to Tune.

# In[ ]:


def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"], config["activation"])
        train.report({"iterations": step, "mean_loss": score})


# In[ ]:


ray.init(configure_logging=False)


# Next we define a search space. The critical assumption is that the optimal hyperparamters live within this space. Yet, if the space is very large, then those hyperparameters may be difficult to find in a short amount of time.

# In[ ]:


#search_config = {
#    "steps": 100,
#    "width": tune.uniform(0, 20),
#    "height": tune.uniform(-100, 100),
#    "activation": tune.choice(["relu, tanh"])
#}


# In[ ]:


space = [
    {
        "name": "width",
        "type": "int",
        "bounds": {"min": 0, "max": 20},
    },
    {
        "name": "height",
        "type": "int",
        "bounds": {"min": -100, "max": 100},
    },
    {
        "name": "activation",
        "type": "categorical",
        "categorical_values": ["relu","tanh"]
    }
]


# Now we define the search algorithm built from `SigOptSearch`, constrained  to a maximum of `1` concurrent trials.

# In[ ]:


algo = SigOptSearch(
    space,
    name="SigOpt Example Experiment",
    max_concurrent=1,
    metric="mean_loss",
    mode="min",
)


# The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples.
# (you can decrease this if it takes too long on your machine).
# 
# ```
# num_samples = 1000
# ```

# In[ ]:


# If 1000 samples take too long, you can reduce this number.
# We override this number here for our smoke tests.
num_samples = 10


# Finally, we run the experiment to `"min"`imize the "mean_loss" of the `objective` by searching `space` provided above to `algo`, `num_samples` times. This previous sentence is fully characterizes the search problem we aim to solve. With this in mind, notice how efficient it is to execute `tuner.fit()`.

# In[ ]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space={"steps": 100},
)
results = tuner.fit()


# Here are the hyperparamters found to minimize the mean loss of the defined objective.

# In[ ]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# ## Multi-objective optimization with Sigopt
# 
# We define another simple objective.

# In[ ]:


np.random.seed(0)
vector1 = np.random.normal(0, 0.1, 100)
vector2 = np.random.normal(0, 0.1, 100)

def evaluate(w1, w2):
    total = w1 * vector1 + w2 * vector2
    return total.mean(), total.std()

def multi_objective(config):
    w1 = config["w1"]
    w2 = config["total_weight"] - w1
    
    average, std = evaluate(w1, w2)
    train.report({"average": average, "std": std, "sharpe": average / std})
    time.sleep(0.1)


# We define the space manually for `SigOptSearch`.

# In[ ]:


space = [
    {
        "name": "w1",
        "type": "double",
        "bounds": {"min": 0, "max": 1},
    },
]

algo = SigOptSearch(
    space,
    name="sigopt_multiobj_exp",
    observation_budget=num_samples,
    max_concurrent=1,
    metric=["average", "std", "sharpe"],
    mode=["max", "min", "obs"],
)


# Finally, we run the experiment using Ray Tune, which in this case requires very little input since most of the construction has gone inside `search_algo`.

# In[ ]:


tuner = tune.Tuner(
    multi_objective,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space={"total_weight": 1},
)
results = tuner.fit()


# And here are they hyperparameters found to minimize the objective on average.

# In[ ]:


print("Best hyperparameters found were: ", results.get_best_result("average", "min").config)


# ## Incorporating prior beliefs with Sigopt
# 
# If we have information about beneficial hyperparameters within the search space, then we can incorporate this bias via a prior distribution. Without explicitly incorporating a prior, the default is a uniform distribution of preference over the search space. Below we highlight the hyperparamters we expect to be better with a Gaussian prior distribution.
# 
# We start with defining another objective.

# In[ ]:


np.random.seed(0)
vector1 = np.random.normal(0.0, 0.1, 100)
vector2 = np.random.normal(0.0, 0.1, 100)
vector3 = np.random.normal(0.0, 0.1, 100)

def evaluate(w1, w2, w3):
    total = w1 * vector1 + w2 * vector2 + w3 * vector3
    return total.mean(), total.std()

def multi_objective_two(config):
    w1 = config["w1"]
    w2 = config["w2"]
    total = w1 + w2
    if total > 1:
        w3 = 0
        w1 /= total
        w2 /= total
    else:
        w3 = 1 - total
    
    average, std = evaluate(w1, w2, w3)
    train.report({"average": average, "std": std})


# Now we begin setting up the SigOpt experiment and algorithm. Incorporating a prior distribution over hyperparameters requires establishing a connection with SigOpt via `"SIGOPT_KEY"` environment variable. Here we create a Gaussian prior over w1 and w2, each independently. 

# In[ ]:


samples = num_samples

conn = Connection(client_token=os.environ["SIGOPT_KEY"])

experiment = conn.experiments().create(
    name="prior experiment example",
    parameters=[
        {
            "name": "w1",
            "bounds": {"max": 1, "min": 0},
            "prior": {"mean": 1 / 3, "name": "normal", "scale": 0.2},
            "type": "double",
        },
        {
            "name": "w2",
            "bounds": {"max": 1, "min": 0},
            "prior": {"mean": 1 / 3, "name": "normal", "scale": 0.2},
            "type": "double",
        },  
    ],
    metrics=[
        dict(name="std", objective="minimize", strategy="optimize"),
        dict(name="average", strategy="store"),
    ],
    observation_budget=samples,
    parallel_bandwidth=1,
)

algo = SigOptSearch(
    connection=conn,
    experiment_id=experiment.id,
    name="sigopt_prior_multi_exp",
    max_concurrent=1,
    metric=["average", "std"],
    mode=["obs", "min"],
)


# Finally, we run the experiment using Ray Tune, where `search_algo` establishes the search space.

# In[ ]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=samples,
    )
)
results = tuner.fit()


# And here are they hyperparameters found to minimize the the objective on average.

# In[ ]:


print("Best hyperparameters found were: ", results.get_best_result("average", "min").config)


# In[ ]:


ray.shutdown()

