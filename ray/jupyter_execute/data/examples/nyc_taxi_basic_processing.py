#!/usr/bin/env python
# coding: utf-8

# In[1]:


# flake8: noqa
import warnings
import os

# Suppress noisy requests warnings.
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


# # Processing NYC taxi data using Ray Data
# 
# The [NYC Taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) is a popular tabular dataset.  In this example, we demonstrate some basic data processing on this dataset using Ray Data.
# 
# ## Overview
# 
# This tutorial will cover:
#  - Reading Parquet data
#  - Inspecting the metadata and first few rows of a large Ray {class}`Dataset <ray.data.Dataset>`
#  - Calculating some common global and grouped statistics on the dataset
#  - Dropping columns and rows
#  - Adding a derived column
#  - Shuffling the data
#  - Sharding the data and feeding it to parallel consumers (trainers)
#  - Applying batch (offline) inference to the data
# 
# ## Walkthrough
# 
# Let's start by importing Ray and initializing a local Ray cluster.

# In[ ]:


# Import ray and initialize a local Ray cluster.
import ray
ray.init()


# ### Reading and Inspecting the Data
# 
# Next, we read a few of the files from the dataset. This read is lazy, where reading and all future transformations are delayed until a downstream operation triggers execution (e.g. consuming the data with {meth}`ds.take() <ray.data.Dataset.take>`)
# 

# In[3]:


# Read two Parquet files in parallel.
ds = ray.data.read_parquet([
    "s3://anonymous@air-example-data/ursa-labs-taxi-data/downsampled_2009_01_data.parquet",
    "s3://anonymous@air-example-data/ursa-labs-taxi-data/downsampled_2009_02_data.parquet"
])


# We can easily inspect the schema of this dataset. For Parquet files, we don't even have to read the actual data to get the schema; we can read it from the lightweight Parquet metadata!

# In[4]:


# Fetch the schema from the underlying Parquet metadata.
ds.schema()


# Parquet even stores the number of rows per file in the Parquet metadata, so we can get the number of rows in ``ds`` without triggering a full data read.

# In[5]:


ds.count()


# We can get a nice, cheap summary of the ``Dataset`` by leveraging it's informative repr:

# In[6]:


# Display some metadata about the dataset.
ds


# We can also poke at the actual data, taking a peek at a single row. Since this is only returning a row from the first file, reading of the second file is **not** triggered yet.

# In[7]:


ds.take(1)


# To get a better sense of the data size, we can calculate the size in bytes of the full dataset. Note that for Parquet files, this size-in-bytes will be pulled from the Parquet metadata (not triggering a data read), and therefore might be significantly different than the in-memory size!

# In[8]:


ds.size_bytes()


# In order to get the in-memory size, we can trigger full reading of the dataset and inspect the size in bytes.

# In[9]:


ds.materialize().size_bytes()


# #### Advanced Aside - Reading Partitioned Parquet Datasets
# 
# In addition to being able to read lists of individual files, {func}`ray.data.read_parquet() <ray.data.read_parquet>` (as well as other ``ray.data.read_*()`` APIs) can read directories containing multiple Parquet files. For Parquet in particular, reading Parquet datasets partitioned by a particular column is supported, allowing for path-based (zero-read) partition filtering and (optionally) including the partition column value specified in the file paths directly in the read table data.
# 
# For the NYC taxi dataset, instead of reading individual per-month Parquet files, we can read the entire 2009 directory.
# 
# ```{warning}
# This could be a lot of data (downsampled with 0.01 ratio leads to ~50.2 MB on disk, ~147 MB in memory), so be careful triggering full reads on a limited-memory machine! This is one place where Dataset's lazy reading comes in handy: Dataset will not execute any read tasks eagerly and will execute the minimum number of file reads to satisfy downstream operations, which allows us to inspect a subset of the data without having to read the entire dataset.
# ```

# In[10]:


# Read all Parquet data for the year 2009.
year_ds = ray.data.read_parquet("s3://anonymous@air-example-data/ursa-labs-taxi-data/downsampled_2009_full_year_data.parquet")


# The metadata that Dataset prints in its repr is guaranteed to not trigger reads of all files; data such as the row count and the schema is pulled directly from the Parquet metadata.

# In[11]:


year_ds.count()


# That's a lot of rows! Since we're not going to use this full-year data, let's now delete this dataset to free up some memory in our Ray cluster.

# In[12]:


del year_ds


# ### Data Exploration and Cleaning
# 
# Let's calculate some stats to get a better picture of our data.

# In[13]:


# What's the longets trip distance, largest tip amount, and most number of passengers?
ds.max(["trip_distance", "tip_amount", "passenger_count"])


# We don't have any use for the ``store_and_fwd_flag`` or ``mta_tax`` columns, so let's drop those.

# In[16]:


# Drop some columns.
ds = ds.drop_columns(["store_and_fwd_flag", "mta_tax"])


# Let's say we want to know how many trips there are for each passenger count.

# In[17]:


ds.groupby("passenger_count").count().take()


# Again, it looks like there are some more nonsensical passenger counts, i.e. the negative ones. Let's filter those out too.

# In[18]:


# Filter our records with negative passenger counts.
ds = ds.map_batches(lambda df: df[df["passenger_count"] > 0])


# Do the passenger counts influences the typical trip distance?

# In[ ]:


# Mean trip distance grouped by passenger count.
ds.groupby("passenger_count").mean("trip_distance").take()


# See {ref}`Transforming Data <transforming_data>` for more information on how we can process our data with Ray Data.

# #### Advanced Aside - Projection and Filter Pushdown
# 
# Note that Ray Data' Parquet reader supports projection (column selection) and row filter pushdown, where we can push the above column selection and the row-based filter to the Parquet read. If we specify column selection at Parquet read time, the unselected columns won't even be read from disk!
# 
# The row-based filter is specified via
# [Arrow's dataset field expressions](https://arrow.apache.org/docs/6.0/python/generated/pyarrow.dataset.Expression.html#pyarrow.dataset.Expression). See the {ref}`Parquet row pruning tips <parquet_row_pruning>` for more information.

# In[19]:


# Only read the passenger_count and trip_distance columns.
import pyarrow as pa
filter_expr = (
    (pa.dataset.field("passenger_count") <= 10)
    & (pa.dataset.field("passenger_count") > 0)
)

pushdown_ds = ray.data.read_parquet(
    [
        "s3://anonymous@air-example-data/ursa-labs-taxi-data/downsampled_2009_01_data.parquet",
        "s3://anonymous@air-example-data/ursa-labs-taxi-data/downsampled_2009_02_data.parquet",
    ],
    columns=["passenger_count", "trip_distance"],
    filter=filter_expr,
)

# Force full execution of both of the file reads.
pushdown_ds = pushdown_ds.materialize()
pushdown_ds


# In[20]:


# Delete the pushdown dataset. Deleting the Dataset object
# will release the underlying memory in the cluster.
del pushdown_ds


# ### Ingesting into Model Trainers
# 
# Now that we've learned more about our data and we have cleaned up our dataset a bit, we now look at how we can feed this dataset into some dummy model trainers.
# 
# First, let's do a full global random shuffle of the dataset to decorrelate these samples.

# In[22]:


ds = ds.random_shuffle()


# We define a dummy ``Trainer`` actor, where each trainer will consume a dataset shard in batches and simulate model training.
# 
# :::{note}
# In a real training workflow, we would feed ``ds`` to {ref}`Ray Train <train-docs>`, which would do this sharding and creation of training actors for us, under the hood.
# 

# In[23]:


@ray.remote
class Trainer:
    def __init__(self, rank: int):
        pass

    def train(self, shard: ray.data.Dataset) -> int:
        for batch in shard.iter_batches(batch_size=256):
            pass
        return shard.count()

trainers = [Trainer.remote(i) for i in range(4)]
trainers


# Next, we split the dataset into ``len(trainers)`` shards, ensuring that the shards are of equal size.

# In[24]:


shards = ds.split(n=len(trainers), equal=True)
shards


# Finally, we simulate training, passing each shard to the corresponding trainer. The number of rows per shard is returned.

# In[25]:


ray.get([w.train.remote(s) for w, s in zip(trainers, shards)])


# In[26]:


# Delete trainer actor handle references, which should terminate the actors.
del trainers


# ### Parallel Batch Inference
# 
# ```{tip}
# Refer to the blog on [Model Batch Inference in Ray](https://www.anyscale.com/blog/model-batch-inference-in-ray-actors-actorpool-and-datasets) for an overview of batch inference strategies in Ray and additional examples.
# ```
# After we've trained a model, we may want to perform batch (offline) inference on such a tabular dataset. With Ray Data, this is as easy as a {meth}`ds.map_batches() <ray.data.Dataset.map_batches>` call!
# 
# First, we define a callable class that will cache the loading of the model in its constructor.

# In[27]:


import pandas as pd

def load_model():
    # A dummy model.
    def model(batch: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"score": batch["passenger_count"] % 2 == 0})
    
    return model

class BatchInferModel:
    def __init__(self):
        self.model = load_model()
    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        return self.model(batch)


# ``BatchInferModel``'s constructor will only be called once per actor worker when using the actor pool compute strategy in {meth}`ds.map_batches() <ray.data.Dataset.map_batches>`.

# In[28]:


ds.map_batches(BatchInferModel, batch_size=2048, compute=ray.data.ActorPoolStrategy()).take()


# If wanting to perform batch inference on GPUs, simply specify the number of GPUs you wish to provision for each batch inference worker.
# 
# :::{warning}
# This will only run successfully if your cluster has nodes with GPUs!

# In[29]:


ds.map_batches(
    BatchInferModel,
    batch_size=256,
    #num_gpus=1,  # Uncomment this to run this on GPUs!
    compute=ray.data.ActorPoolStrategy(),
).take()


# We can also configure the autoscaling actor pool that this inference stage uses, setting upper and lower bounds on the actor pool size, and even tweak the batch prefetching vs. inference task queueing tradeoff.

# In[30]:


from ray.data import ActorPoolStrategy

# The actor pool will have at least 2 workers and at most 8 workers.
strategy = ActorPoolStrategy(min_size=2, max_size=8)

ds.map_batches(
    BatchInferModel,
    batch_size=256,
    #num_gpus=1,  # Uncomment this to run this on GPUs!
    compute=strategy,
).take()

