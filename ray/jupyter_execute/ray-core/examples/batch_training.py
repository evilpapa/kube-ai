#!/usr/bin/env python
# coding: utf-8

# (mmt-core)=
# 
# # Batch Training with Ray Core

# ```{tip}
# The workload showcased in this notebook can be expressed using different Ray components, such as Ray Data, Ray Tune and Ray Core.
# For best practices, see {ref}`ref-use-cases-mmt`.
# ```
# 
# Batch training and tuning are common tasks in simple machine learning use-cases such as time series forecasting. They require fitting of simple models on multiple data batches corresponding to locations, products, etc. This notebook showcases how to conduct batch training on the [NYC Taxi Dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) using only Ray Core and stateless Ray tasks.

# Batch training in the context of this notebook is understood as creating the same model(s) for different and separate datasets or subsets of a dataset. This task is naively parallelizable and can be easily scaled with Ray.
# 
# ![Batch training diagram](./images/batch-training.svg)

# # Contents
# In this tutorial, we will walk through the following steps:
#  1. Reading parquet data,
#  2. Using Ray tasks to preprocess, train and evaluate data batches,
#  3. Dividing data into batches and spawning a Ray task for each batch to be run in parallel,
#  4. Starting batch training,
#  5. [Optional] Optimizing for runtime over memory with centralized data loading.
# 
# # Walkthrough
# 
# We want to analyze the relationship between the dropoff location and the trip duration. The relationship will be very different for each pickup location, therefore we need to have a separate model for each of those. Furthermore, the relationship can change with time. Therefore, our task is to create separate models for each pickup location-month combination. The dataset we are using is already partitioned into months (each file being equal to one), and we can use the `pickup_location_id` column in the dataset to group it into data batches. We will then fit models for each batch and choose the best one.
# 
# Let’s start by importing Ray and initializing a local Ray cluster.

# In[1]:


from typing import Callable, Optional, List, Union, Tuple, Iterable
import time
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import pyarrow as pa
from pyarrow import fs
from pyarrow import dataset as ds
from pyarrow import parquet as pq
import pyarrow.compute as pc


# In[ ]:


import ray

ray.init(ignore_reinit_error=True)


# For benchmarking purposes, we can print the times of various operations. In order to reduce clutter in the output, this is set to False by default.

# In[3]:


PRINT_TIMES = False


def print_time(msg: str):
    if PRINT_TIMES:
        print(msg)


# To speed things up, we'll only use a small subset of the full dataset consisting of two last months of 2019. You can choose to use the full dataset for 2018-2019 by setting the `SMOKE_TEST` variable to False.

# In[4]:


SMOKE_TEST = True


# ## Reading parquet data
# 
# The `read_data` function reads a Parquet file and uses a push-down predicate to extract the data batch we want to fit a model on using the provided index to group the rows. By having each task read the data and extract batches separately, we ensure that memory utilization is minimal - as opposed to requiring each task to load the entire partition into memory first.
# 
# We are using PyArrow to read the file, as it supports push-down predicates to be applied during file reading. This lets us avoid having to load an entire file into memory, which could cause an OOM error with a large dataset. After the dataset is loaded, we convert it to pandas so that it can be used for training with scikit-learn.

# In[5]:


def read_data(file: str, pickup_location_id: int) -> pd.DataFrame:
    return pq.read_table(
        file,
        filters=[("pickup_location_id", "=", pickup_location_id)],
        columns=[
            "pickup_at",
            "dropoff_at",
            "pickup_location_id",
            "dropoff_location_id",
        ],
    ).to_pandas()


# ## Creating Ray tasks to preprocess, train and evaluate data batches
# 
# As we will be using the NYC Taxi dataset, we define a simple batch transformation function to set correct data types, calculate the trip duration and fill missing values.

# In[6]:


def transform_batch(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_at"] = pd.to_datetime(
        df["pickup_at"], format="%Y-%m-%d %H:%M:%S"
    )
    df["dropoff_at"] = pd.to_datetime(
        df["dropoff_at"], format="%Y-%m-%d %H:%M:%S"
    )
    df["trip_duration"] = (df["dropoff_at"] - df["pickup_at"]).dt.seconds
    df["pickup_location_id"] = df["pickup_location_id"].fillna(-1)
    df["dropoff_location_id"] = df["dropoff_location_id"].fillna(-1)
    return df


# We will be fitting scikit-learn models on data batches. We define a Ray task `fit_and_score_sklearn` that fits the model and calculates mean absolute error on the validation set. We will be treating this as a simple regression problem where we want to predict the relationship between the drop-off location and the trip duration.

# In[7]:


# Ray task to fit and score a scikit-learn model.
@ray.remote
def fit_and_score_sklearn(
    train: pd.DataFrame, test: pd.DataFrame, model: BaseEstimator
) -> Tuple[BaseEstimator, float]:
    train_X = train[["dropoff_location_id"]]
    train_y = train["trip_duration"]
    test_X = test[["dropoff_location_id"]]
    test_y = test["trip_duration"]

    # Start training.
    model = model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    error = mean_absolute_error(test_y, pred_y)
    return model, error


# Next, we will define a `train_and_evaluate` Ray task which contains all logic necessary to load a data batch, transform it, split it into train and test and fit and evaluate models on it. We make sure to return the file and location id used so that we can map the fitted models back to them.
# 
# For data loading and processing, we are using the `read_data` and `transform_batch` functions we have defined earlier.
# 

# In[8]:


def train_and_evaluate_internal(
    df: pd.DataFrame, models: List[BaseEstimator], pickup_location_id: int = 0
) -> List[Tuple[BaseEstimator, float]]:
    # We need at least 4 rows to create a train / test split.
    if len(df) < 4:
        print(
            f"Dataframe for LocID: {pickup_location_id} is empty or smaller than 4"
        )
        return None

    # Train / test split.
    train, test = train_test_split(df)

    # We put the train & test dataframes into Ray object store
    # so that they can be reused by all models fitted here.
    # https://docs.ray.io/en/master/ray-core/patterns/pass-large-arg-by-value.html
    train_ref = ray.put(train)
    test_ref = ray.put(test)

    # Launch a fit and score task for each model.
    results = ray.get(
        [
            fit_and_score_sklearn.remote(train_ref, test_ref, model)
            for model in models
        ]
    )
    results.sort(key=lambda x: x[1])  # sort by error
    return results


@ray.remote
def train_and_evaluate(
    file_name: str,
    pickup_location_id: int,
    models: List[BaseEstimator],
) -> Tuple[str, str, List[Tuple[BaseEstimator, float]]]:
    start_time = time.time()
    data = read_data(file_name, pickup_location_id)
    data_loading_time = time.time() - start_time
    print_time(
        f"Data loading time for LocID: {pickup_location_id}: {data_loading_time}"
    )

    # Perform transformation
    start_time = time.time()
    data = transform_batch(data)
    transform_time = time.time() - start_time
    print_time(
        f"Data transform time for LocID: {pickup_location_id}: {transform_time}"
    )

    # Perform training & evaluation for each model
    start_time = time.time()
    results = (train_and_evaluate_internal(data, models, pickup_location_id),)
    training_time = time.time() - start_time
    print_time(
        f"Training time for LocID: {pickup_location_id}: {training_time}"
    )

    return (
        file_name,
        pickup_location_id,
        results,
    )


# ## Dividing data into batches and spawning a Ray task for each batch to be ran in parallel
# 
# The `run_batch_training` driver function generates tasks for each Parquet file it recieves (with each file corresponding to one month). We define the function to take in a list of models, so that we can evaluate them all and choose the best one for each batch. The function blocks when it reaches `ray.get()` and waits for tasks to return their results.

# In[9]:


def run_batch_training(files: List[str], models: List[BaseEstimator]):
    print("Starting run...")
    start = time.time()

    # Store task references
    task_refs = []
    for file in files:
        try:
            locdf = pq.read_table(file, columns=["pickup_location_id"])
        except Exception:
            continue
        pickup_location_ids = locdf["pickup_location_id"].unique()

        for pickup_location_id in pickup_location_ids:
            # Cast PyArrow scalar to Python if needed.
            try:
                pickup_location_id = pickup_location_id.as_py()
            except Exception:
                pass
            task_refs.append(
                train_and_evaluate.remote(file, pickup_location_id, models)
            )

    # Block to obtain results from each task
    results = ray.get(task_refs)

    taken = time.time() - start
    count = len(results)
    # If result is None, then it means there weren't enough records to train
    results_not_none = [x for x in results if x is not None]
    count_not_none = len(results_not_none)

    # Sleep a moment for nicer output
    time.sleep(1)
    print("", flush=True)
    print(f"Number of pickup locations: {count}")
    print(
        f"Number of pickup locations with enough records to train: {count_not_none}"
    )
    print(f"Number of models trained: {count_not_none * len(models)}")
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
    return results


# ## Starting batch training
# 
# We can now tie everything together! First, we obtain the partitions of the dataset from an S3 bucket so that we can pass them to `run`. The dataset is partitioned by year and month, meaning each file represents one month.

# In[10]:


# Obtain the dataset. Each month is a separate file.
dataset = ds.dataset(
    "s3://anonymous@air-example-data/ursa-labs-taxi-data/by_year/",
    partitioning=["year", "month"],
)
starting_idx = -2 if SMOKE_TEST else 0
files = [f"s3://anonymous@{file}" for file in dataset.files][starting_idx:]
print(f"Obtained {len(files)} files!")


# We can now run our script. The output is a list of tuples in the following format: `(file name, partition id, list of models and their MAE scores)`. For brevity, we will print out the first 10 tuples.

# In[11]:


from sklearn.linear_model import LinearRegression

results = run_batch_training(files, models=[LinearRegression()])
print(results[:10])


# Using the output we've gotten, we can now tie each model to the given file (month)-pickup location combination and see their predictive power, as measured by the error. At this stage, we can carry on with further analysis if necessary or use them for inference.
# 
# We can also provide multiple scikit-learn models to our `run` function and the best one will be chosen for each batch. A common use-case here would be to define several models of the same type with different hyperparameters.

# In[12]:


from sklearn.tree import DecisionTreeRegressor

results = run_batch_training(
    files,
    models=[
        LinearRegression(),
        DecisionTreeRegressor(),
        DecisionTreeRegressor(splitter="random"),
    ],
)
print(results[:10])


# ## [Optional] Optimizing for runtime over memory with centralized data loading
# 
# In order to ensure that the data can always fit in memory, each task reads the files independently and extracts the desired data batch. This, however, negatively impacts the runtime. If we have sufficient memory in our Ray cluster, we can instead load each partition once, extract the batches, and save them in the [Ray object store](objects-in-ray), reducing time required dramatically at a cost of higher memory usage. In other words, we perform centralized data loading using Ray object store as opposed to distributed data loading.
# 
# Notice we do not call `ray.get()` on the references of the `read_into_object_store`. Instead, we pass the reference itself as the argument to the `train_and_evaluate.remote` dispatch, [allowing for the data to stay in the object store until it is actually needed](unnecessary-ray-get). This avoids a situation where all the data would be loaded into the memory of the process calling `ray.get()`.
# 
# You can use the Ray Dashboard to compare the memory usage between the previous approach and this one.

# In[13]:


# Redefine the train_and_evaluate task to use in-memory data.
# We still keep file_name and pickup_location_id for identification purposes.
@ray.remote
def train_and_evaluate(
    pickup_location_id_and_data: Tuple[int, pd.DataFrame],
    file_name: str,
    models: List[BaseEstimator],
) -> Tuple[str, str, List[Tuple[BaseEstimator, float]]]:
    pickup_location_id, data = pickup_location_id_and_data
    # Perform transformation
    start_time = time.time()
    # The underlying numpy arrays are stored in the Ray object
    # store for efficient access, making them immutable. We therefore
    # copy the DataFrame to obtain a mutable copy we can transform.
    data = data.copy()
    data = transform_batch(data)
    transform_time = time.time() - start_time
    print_time(
        f"Data transform time for LocID: {pickup_location_id}: {transform_time}"
    )

    return (
        file_name,
        pickup_location_id,
        train_and_evaluate_internal(data, models, pickup_location_id),
    )


# This allows us to create a Ray Task that is also a generator, returning object references.
@ray.remote(num_returns="dynamic")
def read_into_object_store(file: str) -> ray.ObjectRefGenerator:
    print(f"Loading {file}")
    # Read the entire file into memory.
    try:
        locdf = pq.read_table(
            file,
            columns=[
                "pickup_at",
                "dropoff_at",
                "pickup_location_id",
                "dropoff_location_id",
            ],
        )
    except Exception:
        return []

    pickup_location_ids = locdf["pickup_location_id"].unique()

    for pickup_location_id in pickup_location_ids:
        # Each id-data batch tuple will be put as a separate object into the Ray object store.

        # Cast PyArrow scalar to Python if needed.
        try:
            pickup_location_id = pickup_location_id.as_py()
        except Exception:
            pass

        yield (
            pickup_location_id,
            locdf.filter(
                pc.equal(locdf["pickup_location_id"], pickup_location_id)
            ).to_pandas(),
        )


def run_batch_training_with_object_store(
    files: List[str], models: List[BaseEstimator]
):
    print("Starting run...")
    start = time.time()

    # Store task references
    task_refs = []

    # Use a SPREAD scheduling strategy to load each
    # file on a separate node as an OOM safeguard.
    # This is not foolproof though! We can also specify a resource
    # requirement for memory, if we know what is the maximum
    # memory requirement for a single file.
    read_into_object_store_spread = read_into_object_store.options(
        scheduling_strategy="SPREAD"
    )

    # Dictionary of references to read tasks with file names as keys
    read_tasks_by_file = {
        files[file_id]: read_into_object_store_spread.remote(file)
        for file_id, file in enumerate(files)
    }

    for file, read_task_ref in read_tasks_by_file.items():
        # We iterate over references and pass them to the tasks directly
        for pickup_location_id_and_data_batch_ref in iter(ray.get(read_task_ref)):
            task_refs.append(
                train_and_evaluate.remote(
                    pickup_location_id_and_data_batch_ref, file, models
                )
            )

    # Block to obtain results from each task
    results = ray.get(task_refs)

    taken = time.time() - start
    count = len(results)
    # If result is None, then it means there weren't enough records to train
    results_not_none = [x for x in results if x is not None]
    count_not_none = len(results_not_none)

    # Sleep a moment for nicer output
    time.sleep(1)
    print("", flush=True)
    print(f"Number of pickup locations: {count}")
    print(
        f"Number of pickup locations with enough records to train: {count_not_none}"
    )
    print(f"Number of models trained: {count_not_none * len(models)}")
    print(f"TOTAL TIME TAKEN: {taken:.2f} seconds")
    return results


# In[14]:


results = run_batch_training_with_object_store(
    files, models=[LinearRegression()]
)
print(results[:10])


# We can see that this approach allowed us to finish training much faster, but it would not have been possible if the dataset was too large to fit into our cluster memory. Therefore, this pattern is only recommended if the data you are working with is small. Otherwise, it is recommended to load the data inside the tasks right before its used. As always, your mileage may vary - we recommend you try both approaches for your workload and see what works best for you!
