! pip install fastai==2.3.1 neptune-client[fastai]
import fastai
import neptune.new as neptune
from fastai.vision.all import *
from neptune.new.integrations.fastai import NeptuneCallback

# Initialize a run with project details and generated API token.
run = neptune.init(
    project="dk-1232/NeptunefastAI",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYmY5MGFiNC05OTQ1LTQ3YTAtODU3Ni0xMzU4NmYwZjY5ZTgifQ==",
) 

# Load the MNIST dataset images
path = untar_data(URLs.MNIST_TINY)
path.ls()

#Load the image dataset into ImageDataLoaders with batch size = 128. Changed to 64 and ran it again to compare two different runs.
dls = ImageDataLoaders.from_csv(path, bs = 128)
dls.show_batch()

#Load the ResNet34 pre-trained model from FastAI to train it with the dataset
learn = cnn_learner(dls, resnet34, cbs=[NeptuneCallback(run,"experiment"), SaveModelCallback()])

# Train it for 100 epochs
learn.fit_one_cycle(100)

# Stop logging.
run.stop()