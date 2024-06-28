#!/usr/bin/env python
# coding: utf-8

# (tune-comet-ref)=
# 
# # Using Comet with Tune
# 
# [Comet](https://www.comet.ml/site/) is a tool to manage and optimize the
# entire ML lifecycle, from experiment tracking, model optimization and dataset
# versioning to model production monitoring.
# 
# ```{image} /images/comet_logo_full.png
# :align: center
# :alt: Comet
# :height: 120px
# :target: https://www.comet.ml/site/
# ```
# 
# ```{contents}
# :backlinks: none
# :local: true
# ```
# 
# ## Example
# 
# To illustrate logging your trial results to Comet, we'll define a simple training function
# that simulates a `loss` metric:

# In[1]:


import numpy as np
from ray import train, tune


def train_function(config):
    for i in range(30):
        loss = config["mean"] + config["sd"] * np.random.randn()
        train.report({"loss": loss})


# Now, given that you provide your Comet API key and your project name like so:

# In[2]:


api_key = "YOUR_COMET_API_KEY"
project_name = "YOUR_COMET_PROJECT_NAME"


# In[3]:


# This cell is hidden from the rendered notebook. It makes the 
from unittest.mock import MagicMock
from ray.air.integrations.comet import CometLoggerCallback

CometLoggerCallback._logger_process_cls = MagicMock
api_key = "abc"
project_name = "test"


# You can add a Comet logger by specifying the `callbacks` argument in your `RunConfig()` accordingly:

# In[4]:


from ray.air.integrations.comet import CometLoggerCallback

tuner = tune.Tuner(
    train_function,
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
    ),
    run_config=train.RunConfig(
        callbacks=[
            CometLoggerCallback(
                api_key=api_key, project_name=project_name, tags=["comet_example"]
            )
        ],
    ),
    param_space={"mean": tune.grid_search([1, 2, 3]), "sd": tune.uniform(0.2, 0.8)},
)
results = tuner.fit()

print(results.get_best_result().config)


# ## Tune Comet Logger
# 
# Ray Tune offers an integration with Comet through the `CometLoggerCallback`,
# which automatically logs metrics and parameters reported to Tune to the Comet UI.
# 
# Click on the following dropdown to see this callback API in detail:
# 
# ```{eval-rst}
# .. autoclass:: ray.air.integrations.comet.CometLoggerCallback
#    :noindex:
# ```
