#!/usr/bin/env python
# coding: utf-8

# # Image Classification Batch Inference with Huggingface Vision Transformer
# 
# In this example, we will introduce how to use the [Ray Data](data) for **large-scale image classification batch inference with multiple GPU workers.**
# 
# In particular, we will:
# - Load Imagenette dataset from S3 bucket and create a {class}`Ray Dataset <ray.data.dataset.Dataset>`.
# - Load a pretrained Vision Transformer from Huggingface that's been trained on ImageNet.
# - Use [Ray Data](data) to preprocess the dataset and do model inference parallelizing across multiple GPUs
# - Evaluate the predictions and save results to S3/local disk.
# 
# This example will still work even if you do not have GPUs available, but overall performance will be slower.

# To run this example, you will need to install the following:

# In[10]:


get_ipython().system('pip install -q -U "ray[data]" transformers')


# ## Step 1: Reading the Dataset from S3

# [Imagenette](https://github.com/fastai/imagenette) is a subset of Imagenet with 10 classes. We have this dataset hosted publicly in an S3 bucket. Since we are only doing inference here, we load in just the validation split.
# 
# Here, we use {meth}`ray.data.read_images <ray.data.read_images>` to load the validation set from S3. [Ray Data](data) also supports reading from a variety of other [datasources and formats](loading_data).

# In[1]:


import ray

s3_uri = "s3://anonymous@air-example-data-2/imagenette2/val/"

ds = ray.data.read_images(
    s3_uri, mode="RGB"
)
ds


# Inspecting the schema, we can see that there is 1 column in the dataset containing the images stored as Numpy arrays.

# In[2]:


ds.schema()


# ## Step 2: Inference on a single batch

# Next, we can do inference on a single batch of data, using a pre-trained Vision Transformer from Huggingface following [this Huggingface example](https://huggingface.co/docs/transformers/tasks/image_classification#inference). 
# 
# Let’s get a batch of 10 from our dataset. Each image in the batch is represented as a Numpy array.

# In[3]:


single_batch = ds.take_batch(10)


# We can visualize 1 image from this batch.

# In[4]:


from PIL import Image

img = Image.fromarray(single_batch["image"][0])
img


# Now, let’s create a Huggingface Image Classification pipeline from a pre-trained Vision Transformer model.
# 
# We specify the following configurations:
# 1. Set the device to "cuda:0" to use GPU for inference
# 2. We set the batch size to 10 so that we can maximize GPU utilization and do inference on the entire batch at once. 
# 
# We also convert the image Numpy arrays into PIL Images since that's what Huggingface expects.
# 
# From the results, we see that all of the images in the batch are correctly being classified as "tench" which is a type of fish.

# In[33]:


from transformers import pipeline
from PIL import Image

# If doing CPU inference, set device="cpu" instead.
classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device="cuda:0")
outputs = classifier([Image.fromarray(image_array) for image_array in single_batch["image"]], top_k=1, batch_size=10)
del classifier # Delete the classifier to free up GPU memory.
outputs


# ## Step 3: Scaling up to the full Dataset with Ray Data

# By using Ray Data, we can apply the same logic in the previous section to scale up to the entire dataset, leveraging all the GPUs in our cluster.

# There are a couple unique properties about the inference step:
# 1. Model initialization is usually pretty expensive
# 2. We want to do inference in batches to maximize GPU utilization.
# 
# 
# To address 1, we package the inference code in a `ImageClassifier` class. Using a class allows us to put the expensive pipeline loading and initialization code in the `__init__` constructor, which will run only once. 
# The actual model inference logic is in the `__call__` method, which will be called for each batch.
# 
# To address 2, we do our inference in batches, specifying a `batch_size` to the Huggingface Pipeline.
# The `__call__` method takes a batch of data items, instead of a single one. 
# In this case, the batch is a dict that has one key named "image", and the value is a Numpy array of images represented in `np.ndarray` format. This is the same format in step 2, and we can reuse the same inferencing logic from step 2.

# In[46]:


from typing import Dict
import numpy as np

from transformers import pipeline
from PIL import Image

# Pick the largest batch size that can fit on our GPUs
BATCH_SIZE = 1024

class ImageClassifier:
    def __init__(self):
        # If doing CPU inference, set `device="cpu"` instead.
        self.classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device="cuda:0")

    def __call__(self, batch: Dict[str, np.ndarray]):
        # Convert the numpy array of images into a list of PIL images which is the format the HF pipeline expects.
        outputs = self.classifier(
            [Image.fromarray(image_array) for image_array in batch["image"]], 
            top_k=1, 
            batch_size=BATCH_SIZE)
        
        # `outputs` is a list of length-one lists. For example:
        # [[{'score': '...', 'label': '...'}], ..., [{'score': '...', 'label': '...'}]]
        batch["score"] = [output[0]["score"] for output in outputs]
        batch["label"] = [output[0]["label"] for output in outputs]
        return batch


# Then we use the {meth}`map_batches <ray.data.Dataset.map_batches>` API to apply the model to the whole dataset. 
# 
# The first parameter of `map_batches` is the user-defined function (UDF), which can either be a function or a class. Since we are using a class in this case, the UDF will run as long-running [Ray actors](https://docs.ray.io/en/latest/ray-core/key-concepts.html#actors). For class-based UDFs, we use the `compute` argument to specify {class}`ActorPoolStrategy <ray.data.dataset_internal.compute.ActorPoolStrategy>` with the number of parallel actors. And the `batch_size` argument indicates the number of images in each batch.
# 
# The `num_gpus` argument specifies the number of GPUs needed for each `ImageClassifier` instance. In this case, we want 1 GPU for each model replica.

# In[47]:


predictions = ds.map_batches(
    ImageClassifier,
    compute=ray.data.ActorPoolStrategy(size=4), # Use 4 GPUs. Change this number based on the number of GPUs in your cluster.
    num_gpus=1,  # Specify 1 GPU per model replica.
    batch_size=BATCH_SIZE # Use the largest batch size that can fit on our GPUs
)


# ### Verify and Save Results

# Let's take a small batch and verify the results.

# In[48]:


prediction_batch = predictions.take_batch(5)


# We see that all the images are correctly classified as "tench", which is a type of fish.

# In[49]:


from PIL import Image

for image, prediction in zip(prediction_batch["image"], prediction_batch["label"]):
    img = Image.fromarray(image)
    display(img)
    print("Label: ", prediction)


# If the samples look good, we can proceed with saving the results to an external storage, e.g., S3 or local disks. See [Ray Data Input/Output](https://docs.ray.io/en/latest/data/api/input_output.html) for all supported stoarges and file formats.
# 
# ```python
# ds.write_parquet("local://tmp/inference_results")
# ```

# In[ ]:




