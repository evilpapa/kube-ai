#!/usr/bin/env python
# coding: utf-8

# # A Simple MapReduce Example with Ray Core
# 
#  This example demonstrates how to use Ray for a common distributed computing example––counting word occurrences across multiple documents. The complexity lies in the handling of a large corpus, requiring multiple compute nodes to process the data. 
#  The simplicity of implementing MapReduce with Ray is a significant milestone in distributed computing. 
#  Many popular big data technologies, such as Hadoop, are built on this programming model, underscoring the impact 
#  of using Ray Core.
# 
# The MapReduce approach has three phases:

# 1. Map phase
#    The map phase applies a specified function to transform or _map_ elements within a set of data. It produces key-value pairs: the key represents an element and the value is a metric calculated for that element.
#    To count the number of times each word appears in a document,
#    the map function outputs the pair `(word, 1)` every time a word appears, to indicate that it has been found once.
# 2. Shuffle phase
#    The shuffle phase collects all the outputs from the map phase and organizes them by key. When the same key is found on multiple compute nodes, this phase includes transferring or _shuffling_ data between different nodes.
#    If the map phase produces four occurrences of the pair `(word, 1)`, the shuffle phase puts all occurrences of the word on the same node.
# 3. Reduce phase
#    The reduce phase aggregates the elements from the shuffle phase.
#    The total count of each word's occurrences is the sum of occurrences on each node.
#    For example, four instances of `(word, 1)` combine for a final count of `word: 4`.

# The first and last phases are in the MapReduce name, but the middle phase is equally crucial.
# These phases appear straightforward, but their power is in running them concurrently on multiple machines.
# This figure illustrates the three MapReduce phases on a set of documents:

# 
# ![Simple Map Reduce](https://raw.githubusercontent.com/maxpumperla/learning_ray/main/notebooks/images/chapter_02/map_reduce.png)

# ## Loading Data
# 
# We use Python to implement the MapReduce algorithm for the word count and Ray to parallelize the computation.
# We start by loading some sample data from the Zen of Python, a collection of coding guidelines for the Python community. Access to the Zen of Python, according to Easter egg tradition, is by typing `import this` in a Python session. 
# We divide the Zen of Python into three separate "documents" by treating each line as a separate entity
# and then splitting the lines into three partitions.

# In[13]:


import subprocess
zen_of_python = subprocess.check_output(["python", "-c", "import this"])
corpus = zen_of_python.split()

num_partitions = 3
chunk = len(corpus) // num_partitions
partitions = [
    corpus[i * chunk: (i + 1) * chunk] for i in range(num_partitions)
]


# ## Mapping Data
# 
# To determine the map phase, we require a map function to use on each document.
# The output is the pair `(word, 1)` for every word found in a document.
# For basic text documents we load as Python strings, the process is as follows:

# In[14]:


def map_function(document):
    for word in document.lower().split():
        yield word, 1


# We use the `apply_map` function on a large collection of documents by marking it as a task in Ray using the [`@ray.remote`](https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html) decorator.
# When we call `apply_map`, we apply it to three sets of document data (`num_partitions=3`).
# The `apply_map` function returns three lists, one for each partition so that Ray can rearrange the results of the map phase and distribute them to the appropriate nodes.

# In[15]:


import ray

@ray.remote
def apply_map(corpus, num_partitions=3):
    map_results = [list() for _ in range(num_partitions)]
    for document in corpus:
        for result in map_function(document):
            first_letter = result[0].decode("utf-8")[0]
            word_index = ord(first_letter) % num_partitions
            map_results[word_index].append(result)
    return map_results


# For text corpora that can be stored on a single machine, the map phase is not necessasry.
# However, when the data needs to be divided across multiple nodes, the map phase is useful.
# To apply the map phase to the corpus in parallel, we use a remote call on `apply_map`, similar to the previous examples.
# The main difference is that we want three results returned (one for each partition) using the `num_returns` argument.

# In[16]:


map_results = [
    apply_map.options(num_returns=num_partitions)
    .remote(data, num_partitions)
    for data in partitions
]

for i in range(num_partitions):
    mapper_results = ray.get(map_results[i])
    for j, result in enumerate(mapper_results):
        print(f"Mapper {i}, return value {j}: {result[:2]}")


# This example demonstrates how to collect data on the driver with `ray.get`. To continue with another task after the mapping phase, you wouldn't do this. The following section shows how to run all phases together efficiently.

# ## Shuffling and Reducing Data
# 
# The objective for the reduce phase is to transfer all pairs from the `j`-th return value to the same node.
# In the reduce phase we create a dictionary that adds up all word occurrences on each partition:

# In[17]:


@ray.remote
def apply_reduce(*results):
    reduce_results = dict()
    for res in results:
        for key, value in res:
            if key not in reduce_results:
                reduce_results[key] = 0
            reduce_results[key] += value

    return reduce_results


# We can take the j-th return value from each mapper and send it to the j-th reducer using the following method.
# Note that this code works for large datasets that don't fit on one machine because we are passing references
# to the data using Ray objects rather than the actual data itself.
# Both the map and reduce phases can run on any Ray cluster and Ray handles the data shuffling.

# In[18]:


outputs = []
for i in range(num_partitions):
    outputs.append(
        apply_reduce.remote(*[partition[i] for partition in map_results])
    )

counts = {k: v for output in ray.get(outputs) for k, v in output.items()}

sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
for count in sorted_counts:
    print(f"{count[0].decode('utf-8')}: {count[1]}")


# For a thorough understanding of scaling MapReduce tasks across multiple nodes using Ray,
# including memory management, read the [blog post on the topic](https://medium.com/distributed-computing-with-ray/executing-adistributed-shuffle-without-a-mapreduce-system-d5856379426c).
# 
# 

# ## Wrapping up
# 
# This MapReduce example demonstrates how flexible Ray’s programming model is.
# A production-grade MapReduce implementation requires more effort but being able to reproduce common algorithms like this one _quickly_ goes a long way.
# In the earlier years of MapReduce, around 2010, this paradigm was often the only model available for
# expressing workloads.
# With Ray, an entire range of interesting distributed computing patterns
# are accessible to any intermediate Python programmer.
# 
# To learn more about Ray, and Ray Core and particular, see the [Ray Core Examples Gallery](./overview.rst),
# or the ML workloads in our [Use Case Gallery](../../ray-overview/use-cases.rst).
# This MapReduce example can be found in ["Learning Ray"](https://maxpumperla.com/learning_ray/),
# which contains more examples similar to this one.
