#!/usr/bin/env python
# coding: utf-8

# (lightgbm-example-ref)=
# 
# # Training a model with distributed LightGBM
# In this example we will train a model in Ray Train using distributed LightGBM.

# Let's start with installing our dependencies:

# In[1]:


get_ipython().system('pip install -qU "ray[data,train]" lightgbm_ray')


# Then we need some imports:

# In[2]:


from typing import Tuple

import ray
from ray.data import Dataset, Preprocessor
from ray.data.preprocessors import Categorizer, StandardScaler
from ray.train.lightgbm import LightGBMTrainer
from ray.train import Result, ScalingConfig


# Next we define a function to load our train, validation, and test datasets.

# In[3]:


def prepare_data() -> Tuple[Dataset, Dataset, Dataset]:
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer_with_categorical.csv")
    train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)
    test_dataset = valid_dataset.drop_columns(cols=["target"])
    return train_dataset, valid_dataset, test_dataset


# The following function will create a LightGBM trainer, train it, and return the result.

# In[4]:


def train_lightgbm(num_workers: int, use_gpu: bool = False) -> Result:
    train_dataset, valid_dataset, _ = prepare_data()

    # Scale some random columns, and categorify the categorical_column,
    # allowing LightGBM to use its built-in categorical feature support
    scaler = StandardScaler(columns=["mean radius", "mean texture"])
    categorizer = Categorizer(["categorical_column"])

    train_dataset = categorizer.fit_transform(scaler.fit_transform(train_dataset))
    valid_dataset = categorizer.transform(scaler.transform(valid_dataset))

    # LightGBM specific params
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "binary_error"],
    }

    trainer = LightGBMTrainer(
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        label_column="target",
        params=params,
        datasets={"train": train_dataset, "valid": valid_dataset},
        num_boost_round=100,
        metadata = {"scaler_pkl": scaler.serialize(), "categorizer_pkl": categorizer.serialize()}
    )
    result = trainer.fit()
    print(result.metrics)

    return result


# Once we have the result, we can do batch inference on the obtained model. Let's define a utility function for this.

# In[5]:


import pandas as pd
from ray.train import Checkpoint
from ray.data import ActorPoolStrategy


class Predict:

    def __init__(self, checkpoint: Checkpoint):
        self.model = LightGBMTrainer.get_model(checkpoint)
        self.scaler = Preprocessor.deserialize(checkpoint.get_metadata()["scaler_pkl"])
        self.categorizer = Preprocessor.deserialize(checkpoint.get_metadata()["categorizer_pkl"])

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        preprocessed_batch = self.categorizer.transform_batch(self.scaler.transform_batch(batch))
        return {"predictions": self.model.predict(preprocessed_batch)}


def predict_lightgbm(result: Result):
    _, _, test_dataset = prepare_data()

    scores = test_dataset.map_batches(
        Predict, 
        fn_constructor_args=[result.checkpoint], 
        compute=ActorPoolStrategy(), 
        batch_format="pandas"
    )
    
    predicted_labels = scores.map_batches(lambda df: (df > 0.5).astype(int), batch_format="pandas")
    print(f"PREDICTED LABELS")
    predicted_labels.show()


# Now we can run the training:

# In[6]:


result = train_lightgbm(num_workers=2, use_gpu=False)


# And perform inference on the obtained model:

# In[7]:


predict_lightgbm(result)

