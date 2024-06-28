#!/usr/bin/env python
# coding: utf-8

# (convert-torch-to-train)=
# 
# # Convert existing PyTorch code to Ray Train
# 
# If you already have working PyTorch code, you don't have to start from scratch to utilize the benefits of Ray Train. Instead, you can continue to use your existing code and incrementally add Ray Train components as needed.
# 
# Some of the benefits you'll get by using Ray Train with your existing PyTorch training code:
# 
# - Easy distributed data-parallel training on a cluster
# - Automatic checkpointing/fault tolerance and result tracking
# - Parallel data preprocessing
# - Seamless integration with hyperparameter tuning
# 
# This tutorial will show you how to start with Ray Train from your existing PyTorch training code and learn how to **distribute your training**.
# 

# ## The example code
# 
# The example code we'll be using is that of the [PyTorch quickstart tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html). This code trains a neural network classifier on the FashionMNIST dataset.
# 
# You can find the code we used for this tutorial [here on GitHub](https://github.com/pytorch/tutorials/blob/8dddccc4c69116ca724aa82bd5f4596ef7ad119c/beginner_source/basics/quickstart_tutorial.py).

# ## Unmodified
# Let's start with the unmodified code from the example. A thorough explanation of the parts is given in the full tutorial - we'll just focus on the code here.
# 
# We start with some imports:

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
from tempfile import TemporaryDirectory


# Then we download the data: 
# 
# This tutorial assumes that your existing code is using the `torch.utils.data.Dataset` native to PyTorch. It continues to use `torch.utils.data.Dataset` to allow you to make as few code changes as possible. **This tutorial also runs with Ray Data, which gives you the benefits of efficient parallel preprocessing.** For more details on using Ray Data for for images, see the {doc}`Working with Images </data/working-with-images>` Ray Data user guide.

# In[ ]:


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# We can now define the dataloaders:

# In[3]:


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# We can then define and instantiate the neural network:

# In[4]:


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# Define our optimizer and loss:

# In[5]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# And finally our training loop. Note that we renamed the function from `train` to `train_epoch` to avoid conflicts with the Ray Train module later (which is also called `train`):

# In[6]:


def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# And while we're at it, here is our validation loop (note that we sneaked in a `return test_loss` statement and also renamed the function):

# In[7]:


def test_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


# Now we can trigger training and save a model:

# In[8]:


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_epoch(train_dataloader, model, loss_fn, optimizer)
    test_epoch(test_dataloader, model, loss_fn)
print("Done!")


# In[9]:


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


# We'll cover the rest of the tutorial (loading the model and doing batch prediction) later!

# ## Introducing a wrapper function (no Ray Train, yet!)
# The notebook-style from the tutorial is great for tutorials, but in your production code you probably wrapped the actual training logic in a function. So let's do this here, too.
# 
# Note that we do not add or alter any code here (apart from variable definitions) - we just take the loose bits of code in the current tutorial and put them into one function.

# In[10]:


def train_func():
    batch_size = 64
    lr = 1e-3
    epochs = 5
    
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    model = NeuralNetwork().to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        test_epoch(test_dataloader, model, loss_fn)

    print("Done!")


# Let's see it in action again:

# In[11]:


train_func()


# The output should look very similar to the previous ouput.

# ## Starting with Ray Train: Distribute the training
# 
# As a first step, we want to distribute the training across multiple workers. For this we want to
# 
# 1. Use data-parallel training by sharding the training data
# 2. Setup the model to communicate gradient updates across machines
# 3. Report the results back to Ray Train.
# 
# 
# To facilitate this, we only need a few changes to the code:
# 
# 1. We import Ray Train:
# 
#     ```python
#     import ray.train as train
#     ```
# 
# 
# 2. We use a `config` dict to configure some hyperparameters (this is not strictly needed but good practice, especially if you want to o hyperparameter tuning later):
# 
#     ```python
#     def train_func(config: dict):
#         batch_size = config["batch_size"]
#         lr = config["lr"]
#         epochs = config["epochs"]
#     ```
# 
# 3. We dynamically adjust the worker batch size according to the number of workers:
# 
#     ```python
#         batch_size_per_worker = batch_size // train.get_context().get_world_size()
#     ```
# 
# 4. We prepare the data loader for distributed data sharding:
# 
#     ```python
#         train_dataloader = train.torch.prepare_data_loader(train_dataloader)
#         test_dataloader = train.torch.prepare_data_loader(test_dataloader)
#     ```
# 
# 5. We prepare the model for distributed gradient updates:
# 
#     ```python
#         model = train.torch.prepare_model(model)
#     ```
#     :::{note}
#     Note that `train.torch.prepare_model()` also automatically takes care of setting up devices (e.g. GPU training) - so we can get rid of those lines in our current code!
#     :::
# 
# 6. We capture the validation loss and report it to Ray train:
# 
#     ```python
#             test_loss = test(test_dataloader, model, loss_fn)
#             train.report(dict(loss=test_loss))
#     ```
# 
# 7. In the `train_epoch()` and `test_epoch()` functions we divide the `size` by the world size:
# 
#     ```python
#         # Divide by word size
#         size = len(dataloader.dataset) // train.get_context().get_world_size()
#     ```
# 
# 8. In the `train_epoch()` function we can get rid of the device mapping. Ray Train does this for us:
# 
#     ```python
#             # We don't need this anymore! Ray Train does this automatically:
#             # X, y = X.to(device), y.to(device) 
#     ```
# 
# That's it - you need less than 10 lines of Ray Train-specific code and can otherwise continue to use your original code.
# 
# Let's take a look at the resulting code. First the `train_epoch()` function (2 lines changed, and we also commented out the print statement):

# In[12]:


def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) // train.get_context().get_world_size()  # Divide by word size
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # We don't need this anymore! Ray Train does this automatically:
        # X, y = X.to(device), y.to(device)  

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Then the `test_epoch()` function (1 line changed, and we also commented out the print statement):

# In[13]:


def test_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset) // train.get_context().get_world_size()  # Divide by word size
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


# And lastly, the wrapping `train_func()` where we added 4 lines and modified 2 (apart from the config dict):

# In[14]:


import ray.train as train
from ray.train import Checkpoint

def train_func(config: dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    
    batch_size_per_worker = batch_size // train.get_context().get_world_size()
    
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size_per_worker)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_per_worker)
    
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)
    
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for t in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        test_loss = test_epoch(test_dataloader, model, loss_fn)
        
        with TemporaryDirectory() as tmpdir:
            if train.get_context().get_world_rank() == 0:
                state_dict = dict(epoch=t, model=model.state_dict())
                torch.save(state_dict, os.path.join(tmpdir, "checkpoint.bin"))
                checkpoint = Checkpoint.from_directory(tmpdir)
            else:
                checkpoint = None
            train.report(dict(loss=test_loss), checkpoint=checkpoint)

    print("Done!")


# Now we'll use Ray Train's TorchTrainer to kick off the training. Note that we can set the hyperparameters here! In the `scaling_config` we can also configure how many parallel workers to use and if we want to enable GPU training or not.

# In[ ]:


from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig


trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4},
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
)
result = trainer.fit()
print(f"Last result: {result.metrics}")


# Great, this works! You're now training your model in parallel. You could now scale this up to more nodes and workers on your Ray cluster.
# 
# But there are a few improvements we can make to the code in order to get the most of the system. For one, we should enable **checkpointing** to get access to the trained model afterwards. Additionally, we should optimize the **data loading** to take place within the workers.

# ### Enabling checkpointing to retrieve the model
# Enabling checkpointing is pretty easy - we just need to pass a `Checkpoint` object with the model state to the `ray.train.report()` API.
# 
# ```python
#     from ray import train
#     from ray.train import Checkpoint
# 
#     with TemporaryDirectory() as tmpdir:
#         torch.save(
#             {
#                 "epoch": epoch,
#                 "model": model.module.state_dict()
#             },
#             os.path.join(tmpdir, "checkpoint.pt")
#         )
#         train.report(dict(loss=test_loss), checkpoint=Checkpoint.from_directory(tmpdir))
# ```
# 
# ### Move the data loader to the training function
# 
# You may have noticed a warning: `Warning: The actor TrainTrainable is very large (52 MiB). Check that its definition is not implicitly capturing a large array or other object in scope. Tip: use ray.put() to put large objects in the Ray object store.`.
# 
# This is because we load the data outside the training function. Ray then serializes it to make it accessible to the remote tasks (that may get executed on a remote node!). This is not too bad with just 52 MB of data, but imagine this were a full image dataset - you wouldn't want to ship this around the cluster unnecessarily. Instead, you should move the dataset loading part into the `train_func()`. This will then download the data *to disk* once per machine and result in much more efficient data loading.
# 
# The result looks like this:

# In[16]:


from ray.train import Checkpoint

def load_data():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, test_data


def train_func(config: dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    
    batch_size_per_worker = batch_size // train.get_context().get_world_size()
    
    training_data, test_data = load_data()  # <- this is new!
    
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size_per_worker)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_per_worker)
    
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)
    
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        test_loss = test_epoch(test_dataloader, model, loss_fn)
        with TemporaryDirectory() as tmpdir:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict()
                },
                os.path.join(tmpdir, "checkpoint.pt")
            )
            train.report(dict(loss=test_loss), checkpoint=Checkpoint.from_directory(tmpdir))

    print("Done!")


# Let's train again:

# In[ ]:


trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4},
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
)
result = trainer.fit()


# We can see our results here:

# In[18]:


print(f"Last result: {result.metrics}")
print(f"Checkpoint: {result.checkpoint}")


# ## Summary
# 
# This tutorial demonstrated how to turn your existing PyTorch code into code you can use with Ray Train.
# 
# We learned how to
# - enable distributed training using Ray Train abstractions
# - save and retrieve model checkpoints via Ray Train
# 
# In our {ref}`other examples <ref-ray-examples>` you can learn how to do more things with the Ray libraries, such as **serving your model with Ray Serve** or **tune your hyperparameters with Ray Tune.** You can also learn how to perform {ref}`offline batch inference <batch_inference_home>` with Ray Data.
# 
# We hope this tutorial gave you a good starting point to leverage Ray Train. If you have any questions, suggestions, or run into any problems please reach out on [Discuss](https://discuss.ray.io/) or [GitHub](https://github.com/ray-project/ray)!
