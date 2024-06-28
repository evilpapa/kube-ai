#!/usr/bin/env python
# coding: utf-8

# # Running Tune experiments with Optuna
# 
# In this tutorial we introduce Optuna, while running a simple Ray Tune experiment. Tune’s Search Algorithms integrate with Optuna and, as a result, allow you to seamlessly scale up a Optuna optimization process - without sacrificing performance.
# 
# Similar to Ray Tune, Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative ("how" over "what" emphasis), define-by-run style user API. With Optuna, a user has the ability to dynamically construct the search spaces for the hyperparameters. Optuna falls in the domain of "derivative-free optimization" and "black-box optimization".
# 
# In this example we minimize a simple objective to briefly demonstrate the usage of Optuna with Ray Tune via `OptunaSearch`, including examples of conditional search spaces (string together relationships between hyperparameters), and the multi-objective problem (measure trade-offs among all important metrics). It's useful to keep in mind that despite the emphasis on machine learning experiments, Ray Tune optimizes any implicit or explicit objective. Here we assume `optuna==2.9.1` library is installed. To learn more, please refer to [Optuna website](https://optuna.org/).
# 
# Please note that sophisticated schedulers, such as `AsyncHyperBandScheduler`, may not work correctly with multi-objective optimization, since they typically expect a scalar score to compare fitness among trials.
# 

# In[1]:


# !pip install ray[tune]
get_ipython().system('pip install optuna==2.9.1')


# Click below to see all the imports we need for this example.
# You can also launch directly into a Binder instance to run this notebook yourself.
# Just click on the rocket symbol at the top of the navigation.

# In[2]:


import time
from typing import Dict, Optional, Any

import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch


# In[ ]:


ray.init(configure_logging=False)


# Let's start by defining a simple evaluation function.
# An explicit math formula is queried here for demonstration, yet in practice this is typically a black-box function-- e.g. the performance results after training an ML model.
# We artificially sleep for a bit (`0.1` seconds) to simulate a long-running ML experiment.
# This setup assumes that we're running multiple `step`s of an experiment while tuning three hyperparameters,
# namely `width`, `height`, and `activation`.

# In[4]:


def evaluate(step, width, height, activation):
    time.sleep(0.1)
    activation_boost = 10 if activation=="relu" else 0
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost


# Next, our `objective` function to be optimized takes a Tune `config`, evaluates the `score` of your experiment in a training loop,
# and uses `train.report` to report the `score` back to Tune.

# In[5]:


def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"], config["activation"])
        train.report({"iterations": step, "mean_loss": score})


# Next we define a search space. The critical assumption is that the optimal hyperparamters live within this space. Yet, if the space is very large, then those hyperparamters may be difficult to find in a short amount of time.
# 
# The simplest case is a search space with independent dimensions. In this case, a config dictionary will suffice.

# In[6]:


search_space = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu", "tanh"]),
}


# Here we define the Optuna search algorithm:

# In[7]:


algo = OptunaSearch()


# We also constrain the number of concurrent trials to `4` with a `ConcurrencyLimiter`.

# In[8]:


algo = ConcurrencyLimiter(algo, max_concurrent=4)


# The number of samples is the number of hyperparameter combinations that will be tried out. This Tune run is set to `1000` samples.
# (you can decrease this if it takes too long on your machine).

# In[9]:


num_samples = 1000


# In[10]:


# We override here for our smoke tests.
num_samples = 10


# Finally, we run the experiment to `"min"`imize the "mean_loss" of the `objective` by searching `search_space` via `algo`, `num_samples` times. This previous sentence is fully characterizes the search problem we aim to solve. With this in mind, notice how efficient it is to execute `tuner.fit()`.

# In[11]:


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


# And now we have the hyperparameters found to minimize the mean loss.

# In[12]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# ## Providing an initial set of hyperparameters
# 
# While defining the search algorithm, we may choose to provide an initial set of hyperparameters that we believe are especially promising or informative, and
# pass this information as a helpful starting point for the `OptunaSearch` object.

# In[13]:


initial_params = [
    {"width": 1, "height": 2, "activation": "relu"},
    {"width": 4, "height": 2, "activation": "relu"},
]


# Now the `search_alg` built using `OptunaSearch` takes `points_to_evaluate`.

# In[14]:


searcher = OptunaSearch(points_to_evaluate=initial_params)
algo = ConcurrencyLimiter(searcher, max_concurrent=4)


# And run the experiment with initial hyperparameter evaluations:

# In[15]:


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


# We take another look at the optimal hyperparamters.

# In[16]:


print("Best hyperparameters found were: ", results.get_best_result().config)


# ## Conditional search spaces 
# 
# Sometimes we may want to build a more complicated search space that has conditional dependencies on other hyperparameters. In this case, we pass a define-by-run function to the `search_alg` argument in `ray.tune()`.

# In[17]:


def define_by_run_func(trial) -> Optional[Dict[str, Any]]:
    """Define-by-run function to create the search space.

    Ensure no actual computation takes place here. That should go into
    the trainable passed to ``Tuner()`` (in this example, that's
    ``objective``).

    For more information, see https://optuna.readthedocs.io/en/stable\
    /tutorial/10_key_features/002_configurations.html

    This function should either return None or a dict with constant values.
    """

    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    # Define-by-run allows for conditional search spaces.
    if activation == "relu":
        trial.suggest_float("width", 0, 20)
        trial.suggest_float("height", -100, 100)
    else:
        trial.suggest_float("width", -1, 21)
        trial.suggest_float("height", -101, 101)
        
    # Return all constants in a dictionary.
    return {"steps": 100}


# As before, we create the `search_alg` from `OptunaSearch` and `ConcurrencyLimiter`, this time we define the scope of search via the `space` argument and provide no initialization. We also must specific metric and mode when using `space`. 

# In[18]:


searcher = OptunaSearch(space=define_by_run_func, metric="mean_loss", mode="min")
algo = ConcurrencyLimiter(searcher, max_concurrent=4)


# Running the experiment with a define-by-run search space:

# In[19]:


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=num_samples,
    ),
)
results = tuner.fit()


# We take a look again at the optimal hyperparameters.

# In[21]:


print("Best hyperparameters for loss found were: ", results.get_best_result("mean_loss", "min").config)


# ## Multi-objective optimization
# 
# Finally, let's take a look at the multi-objective case.

# In[22]:


def multi_objective(config):
    # Hyperparameters
    width, height = config["width"], config["height"]

    for step in range(config["steps"]):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluate(step, config["width"], config["height"], config["activation"])
        # Feed the score back back to Tune.
        train.report({
           "iterations": step, "loss": intermediate_score, "gain": intermediate_score * width
        })


# We define the `OptunaSearch` object this time with metric and mode as list arguments.

# In[23]:


searcher = OptunaSearch(metric=["loss", "gain"], mode=["min", "max"])
algo = ConcurrencyLimiter(searcher, max_concurrent=4)

tuner = tune.Tuner(
    multi_objective,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=num_samples,
    ),
    param_space=search_space
)
results = tuner.fit()


# Now there are two hyperparameter sets for the two objectives.

# In[24]:


print("Best hyperparameters for loss found were: ", results.get_best_result("loss", "min").config)
print("Best hyperparameters for gain found were: ", results.get_best_result("gain", "max").config)


# We can mix-and-match the use of initial hyperparameter evaluations, conditional search spaces via define-by-run functions, and multi-objective tasks. This is also true of scheduler usage, with the exception of multi-objective optimization-- schedulers typically rely on a single scalar score, rather than the two scores we use here: loss, gain.

# In[25]:


ray.shutdown()

