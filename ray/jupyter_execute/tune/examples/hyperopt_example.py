#!/usr/bin/env python
# coding: utf-8

# # Running Tune experiments with HyperOpt
# 
# In this tutorial we introduce HyperOpt, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with HyperOpt and, as a result, allow you to seamlessly scale up a Hyperopt optimization process - without sacrificing performance.
# 
# HyperOpt provides gradient/derivative-free optimization able to handle noise over the objective landscape, including evolutionary, bandit, and Bayesian optimization algorithms. Nevergrad internally supports search spaces which are continuous, discrete or a mixture of thereof. It also provides a library of functions on which to test the optimization algorithms and compare with other benchmarks.
# 
# In this example we minimize a simple objective to briefly demonstrate the usage of HyperOpt with Ray Tune via `HyperOptSearch`. It's useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `hyperopt==0.2.5` library is installed. To learn more, please refer to [HyperOpt website](http://hyperopt.github.io/hyperopt).
# 
# We include a important example on conditional search spaces (stringing together relationships among hyperparameters).

# Background information:
# - [HyperOpt website](http://hyperopt.github.io/hyperopt)
# 
# Necessary requirements:
# - `pip install ray[tune]`
# - `pip install hyperopt==0.2.5`

# In[1]:


# !pip install ray[tune]
get_ipython().system('pip install hyperopt==0.2.5')


# Click below to see all the imports we need for this example.
# You can also launch directly into a Binder instance to run this notebook yourself.
# Just click on the rocket symbol at the top of the navigation.

# In[2]:


import time

import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp


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


# While defining the search algorithm, we may choose to provide an initial set of hyperparameters that we believe are especially promising or informative, and
# pass this information as a helpful starting point for the `HyperOptSearch` object.
# 
# We also set the maximum concurrent trials to `4` with a `ConcurrencyLimiter`.

# In[6]:


initial_params = [
    {"width": 1, "height": 2, "activation": "relu"},
    {"width": 4, "height": 2, "activation": "tanh"},
]
algo = HyperOptSearch(points_to_evaluate=initial_params)
algo = ConcurrencyLimiter(algo, max_concurrent=4)


# The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples.
# (you can decrease this if it takes too long on your machine).

# In[7]:


num_samples = 1000


# In[8]:


# If 1000 samples take too long, you can reduce this number.
# We override this number here for our smoke tests.
num_samples = 10


# Next we define a search space. The critical assumption is that the optimal hyperparamters live within this space. Yet, if the space is very large, then those hyperparameters may be difficult to find in a short amount of time.

# In[9]:


search_config = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu", "tanh"])
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
    param_space=search_config,
)
results = tuner.fit()


# Here are the hyperparamters found to minimize the mean loss of the defined objective.

# In[11]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# ## Conditional search spaces
# 
# Sometimes we may want to build a more complicated search space that has conditional dependencies on other hyperparameters. In this case, we pass a nested dictionary to `objective_two`, which has been slightly adjusted from `objective` to deal with the conditional search space.

# In[12]:


def evaluation_fn(step, width, height, mult=1):
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 * mult


# In[13]:


def objective_two(config):
    width, height = config["width"], config["height"]
    sub_dict = config["activation"]
    mult = sub_dict.get("mult", 1)
    
    for step in range(config["steps"]):
        intermediate_score = evaluation_fn(step, width, height, mult)
        train.report({"iterations": step, "mean_loss": intermediate_score})
        time.sleep(0.1)


# In[14]:


conditional_space = {
    "activation": hp.choice(
        "activation",
        [
            {"activation": "relu", "mult": hp.uniform("mult", 1, 2)},
            {"activation": "tanh"},
        ],
    ),
    "width": hp.uniform("width", 0, 20),
    "height": hp.uniform("height", -100, 100),
    "steps": 100,
}


# Now we the define the search algorithm built from `HyperOptSearch` constrained by `ConcurrencyLimiter`. When the hyperparameter search space is conditional, we pass it (`conditional_space`) into `HyperOptSearch`.

# In[15]:


algo = HyperOptSearch(space=conditional_space, metric="mean_loss", mode="min")
algo = ConcurrencyLimiter(algo, max_concurrent=4)


# Now we run the experiment, this time with an empty `config` because we instead provided `space` to the `HyperOptSearch` `search_alg`.

# In[16]:


tuner = tune.Tuner(
    objective_two,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
)
results = tuner.fit()


# Finally, we again show the hyperparameters that minimize the mean loss defined by the score of the objective function above. 

# In[17]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# In[18]:


ray.shutdown()

