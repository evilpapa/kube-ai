#!/usr/bin/env python
# coding: utf-8

# (tune-aim-ref)=
# 
# # Using Aim with Tune
# 
# [Aim](https://aimstack.io) is an easy-to-use and supercharged open-source experiment tracker.
# Aim logs your training runs, enables a well-designed UI to compare them, and provides an API to query them programmatically.
# 
# ```{image} /images/aim_logo_full.png
# :align: center
# :alt: Aim
# :width: 100%
# :target: https://aimstack.io
# ```
# 
# Ray Tune currently offers built-in integration with Aim.
# The {ref}`AimLoggerCallback <tune-aim-logger>` automatically logs metrics that are reported to Tune by using the Aim API.
# 
# 
# ```{contents}
# :backlinks: none
# :local: true
# ```
# 
# ## Logging Tune Hyperparameter Configurations and Results to Aim
# 
# The following example demonstrates how the `AimLoggerCallback` can be used in a Tune experiment.
# Begin by installing and importing the necessary modules:

# In[ ]:


get_ipython().run_line_magic('pip', 'install aim')
get_ipython().run_line_magic('pip', 'install ray[tune]')


# In[9]:


import numpy as np

import ray
from ray import train, tune
from ray.tune.logger.aim import AimLoggerCallback


# Next, define a simple `train_function`, which is a [`Trainable`](trainable-docs) that reports a loss to Tune.
# The objective function itself is not important for this example, as our main focus is on the integration with Aim.

# In[2]:


def train_function(config):
    for _ in range(50):
        loss = config["mean"] + config["sd"] * np.random.randn()
        train.report({"loss": loss})


# Here is an example of how you can use the `AimLoggerCallback` with simple grid-search Tune experiment.
# The logger will log each of the 9 grid-search trials as separate Aim runs.

# In[3]:


tuner = tune.Tuner(
    train_function,
    run_config=train.RunConfig(
        callbacks=[AimLoggerCallback()],
        storage_path="/tmp/ray_results",
        name="aim_example",
    ),
    param_space={
        "mean": tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "sd": tune.uniform(0.1, 0.9),
    },
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
    ),
)
tuner.fit()


# When the script executes, a grid-search is carried out and the results are saved to the Aim repo,
# stored at the default location -- the experiment log directory (in this case, it's at `/tmp/ray_results/aim_example`).
# 
# ### More Configuration Options for Aim
# 
# In the example above, we used the default configuration for the `AimLoggerCallback`.
# There are a few options that can be configured as arguments to the callback. For example,
# setting `AimLoggerCallback(repo="/path/to/repo")` will log results to the Aim repo at that
# filepath, which could be useful if you have a central location where the results of multiple
# Tune experiments are stored. Relative paths to the working directory where Tune script is
# launched can be used as well. By default, the repo will be set to the experiment log
# directory. See [the API reference](tune-aim-logger) for more configurations.
# 
# ## Launching the Aim UI
# 
# Now that we have logged our results to the Aim repository, we can view it in Aim's web UI.
# To do this, we first find the directory where the Aim repository lives, then we use
# the Aim CLI to launch the web interface.

# In[7]:


# Uncomment the following line to launch the Aim UI!
#!aim up --repo=/tmp/ray_results/aim_example


# After launching the Aim UI, we can open the web interface at `localhost:43800`.

# ```{image} /images/aim_example_metrics_page.png
# :align: center
# :alt: Aim Metrics Explorer
# :target: https://aimstack.readthedocs.io/en/latest/ui/pages/explorers.html#metrics-explorer
# ```

# The next sections contain more in-depth information on the API of the Tune-Aim integration.
# 
# ## Tune Aim Logger API
# 
# (tune-aim-logger)=
# 
# ```{eval-rst}
# .. autoclass:: ray.tune.logger.aim.AimLoggerCallback
#    :noindex:
# ```

# 
