#!/usr/bin/env python
# coding: utf-8

# # Using MLflow with Tune
# 
# (tune-mlflow-ref)=
# 
# [MLflow](https://mlflow.org/) is an open source platform to manage the ML lifecycle, including experimentation,
# reproducibility, deployment, and a central model registry. It currently offers four components, including
# MLflow Tracking to record and query experiments, including code, data, config, and results.
# 
# ```{image} /images/mlflow.png
# :align: center
# :alt: MLflow
# :height: 80px
# :target: https://www.mlflow.org/
# ```
# 
# Ray Tune currently offers two lightweight integrations for MLflow Tracking.
# One is the {ref}`MLflowLoggerCallback <tune-mlflow-logger>`, which automatically logs
# metrics reported to Tune to the MLflow Tracking API.
# 
# The other one is the {ref}`setup_mlflow <tune-mlflow-setup>` function, which can be
# used with the function API. It automatically
# initializes the MLflow API with Tune's training information and creates a run for each Tune trial.
# Then within your training function, you can just use the
# MLflow like you would normally do, e.g. using `mlflow.log_metrics()` or even `mlflow.autolog()`
# to log to your training process.
# 
# ```{contents}
# :backlinks: none
# :local: true
# ```
# 
# ## Running an MLflow Example
# 
# In the following example we're going to use both of the above methods, namely the `MLflowLoggerCallback` and
# the `setup_mlflow` function to log metrics.
# Let's start with a few crucial imports:

# In[1]:


import os
import tempfile
import time

import mlflow

from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow


# Next, let's define an easy training function (a Tune `Trainable`) that iteratively computes steps and evaluates
# intermediate scores that we report to Tune.

# In[2]:


def evaluation_fn(step, width, height):
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


def train_function(config):
    width, height = config["width"], config["height"]

    for step in range(config.get("steps", 100)):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(step, width, height)
        # Feed the score back to Tune.
        train.report({"iterations": step, "mean_loss": intermediate_score})
        time.sleep(0.1)


# Given an MLFlow tracking URI, you can now simply use the `MLflowLoggerCallback` as a `callback` argument to
# your `RunConfig()`:

# In[3]:


def tune_with_callback(mlflow_tracking_uri, finish_fast=False):
    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(num_samples=5),
        run_config=train.RunConfig(
            name="mlflow",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name="mlflow_callback_example",
                    save_artifact=True,
                )
            ],
        ),
        param_space={
            "width": tune.randint(10, 100),
            "height": tune.randint(0, 100),
            "steps": 5 if finish_fast else 100,
        },
    )
    results = tuner.fit()


# To use the `setup_mlflow` utility, you simply call this function in your training function.
# Note that we also use `mlflow.log_metrics(...)` to log metrics to MLflow.
# Otherwise, this version of our training function is identical to its original.

# In[4]:


def train_function_mlflow(config):
    tracking_uri = config.pop("tracking_uri", None)
    setup_mlflow(
        config,
        experiment_name="setup_mlflow_example",
        tracking_uri=tracking_uri,
    )

    # Hyperparameters
    width, height = config["width"], config["height"]

    for step in range(config.get("steps", 100)):
        # Iterative training function - can be any arbitrary training procedure
        intermediate_score = evaluation_fn(step, width, height)
        # Log the metrics to mlflow
        mlflow.log_metrics(dict(mean_loss=intermediate_score), step=step)
        # Feed the score back to Tune.
        train.report({"iterations": step, "mean_loss": intermediate_score})
        time.sleep(0.1)


# With this new objective function ready, you can now create a Tune run with it as follows:

# In[5]:


def tune_with_setup(mlflow_tracking_uri, finish_fast=False):
    # Set the experiment, or create a new one if does not exist yet.
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name="setup_mlflow_example")

    tuner = tune.Tuner(
        train_function_mlflow,
        tune_config=tune.TuneConfig(num_samples=5),
        run_config=train.RunConfig(
            name="mlflow",
        ),
        param_space={
            "width": tune.randint(10, 100),
            "height": tune.randint(0, 100),
            "steps": 5 if finish_fast else 100,
            "tracking_uri": mlflow.get_tracking_uri(),
        },
    )
    results = tuner.fit()


# If you hapen to have an MLFlow tracking URI, you can set it below in the `mlflow_tracking_uri` variable and set
# `smoke_test=False`.
# Otherwise, you can just run a quick test of the `tune_function` and `tune_decorated` functions without using MLflow.

# In[6]:


smoke_test = True

if smoke_test:
    mlflow_tracking_uri = os.path.join(tempfile.gettempdir(), "mlruns")
else:
    mlflow_tracking_uri = "<MLFLOW_TRACKING_URI>"

tune_with_callback(mlflow_tracking_uri, finish_fast=smoke_test)
if not smoke_test:
    df = mlflow.search_runs(
        [mlflow.get_experiment_by_name("mlflow_callback_example").experiment_id]
    )
    print(df)

tune_with_setup(mlflow_tracking_uri, finish_fast=smoke_test)
if not smoke_test:
    df = mlflow.search_runs(
        [mlflow.get_experiment_by_name("setup_mlflow_example").experiment_id]
    )
    print(df)


# This completes our Tune and MLflow walk-through.
# In the following sections you can find more details on the API of the Tune-MLflow integration.
# 
# ## MLflow AutoLogging
# 
# You can also check out {doc}`here </tune/examples/includes/mlflow_ptl_example>` for an example on how you can
# leverage MLflow auto-logging, in this case with Pytorch Lightning
# 
# ## MLflow Logger API
# 
# (tune-mlflow-logger)=
# 
# ```{eval-rst}
# .. autoclass:: ray.air.integrations.mlflow.MLflowLoggerCallback
#    :noindex:
# ```
# 
# ## MLflow setup API
# 
# (tune-mlflow-setup)=
# 
# ```{eval-rst}
# .. autofunction:: ray.air.integrations.mlflow.setup_mlflow
#    :noindex:
# ```
# 
# ## More MLflow Examples
# 
# - {doc}`/tune/examples/includes/mlflow_ptl_example`: Example for using [MLflow](https://github.com/mlflow/mlflow/)
#   and [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) with Ray Tune.
