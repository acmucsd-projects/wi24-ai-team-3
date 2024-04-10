import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import CNN_model_3
from helpers import *
#from melspec.py 


def predict_language():
    with open("melspec.py") as melspec:
        melspec.create_melspecs_output()
    transformations = tf.Compose([tf.Resize([64,64]), tf.ToTensor()])
    outputset = ImageFolder('melspecs/test', transform=transformations)
    classes = outputset.classes
    
    trained_model = torch.load('cnn_model_trained.pt')
    trained_model.eval()

    outputs = trained_model(outputset)
    return outputs

if __name__ =='__main__':
    predict_language()