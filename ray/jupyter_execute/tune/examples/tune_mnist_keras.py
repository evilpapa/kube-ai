#!/usr/bin/env python
# coding: utf-8

# (tune-mnist-keras)=
# 
# # Using Keras & TensorFlow with Tune
# 
# ```{image} /images/tf_keras_logo.jpeg
# :align: center
# :alt: Keras & TensorFlow Logo
# :height: 120px
# :target: https://keras.io
# ```
# 
# ```{contents}
# :backlinks: none
# :local: true
# ```
# 
# ## Example

# In[1]:


import argparse
import os

from filelock import FileLock
from tensorflow.keras.datasets import mnist

import ray
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.integrations.keras import ReportCheckpointCallback


def train_mnist(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf

    batch_size = 128
    num_classes = 10
    epochs = 12

    with FileLock(os.path.expanduser("~/.data.lock")):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(config["hidden"], activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(lr=config["lr"], momentum=config["momentum"]),
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[ReportCheckpointCallback(metrics={"mean_accuracy": "accuracy"})],
    )


def tune_mnist():
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    tuner = tune.Tuner(
        tune.with_resources(train_mnist, resources={"cpu": 2, "gpu": 0}),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=10,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={"mean_accuracy": 0.99},
        ),
        param_space={
            "threads": 2,
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9),
            "hidden": tune.randint(32, 512),
        },
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

tune_mnist()


# ## More Keras and TensorFlow Examples
# 
# - {doc}`/tune/examples/includes/pbt_memnn_example`: Example of training a Memory NN on bAbI with Keras using PBT.
# - {doc}`/tune/examples/includes/tf_mnist_example`: Converts the Advanced TF2.0 MNIST example to use Tune
#   with the Trainable. This uses `tf.function`.
#   Original code from tensorflow: https://www.tensorflow.org/tutorials/quickstart/advanced
# - {doc}`/tune/examples/includes/pbt_tune_cifar10_with_keras`:
#   A contributed example of tuning a Keras model on CIFAR10 with the PopulationBasedTraining scheduler.
# 
