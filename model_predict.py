import librosa
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
from PIL import Image
from glob import glob
#from melspec.py 


def predict_language():
    ourclasses = ['Arabic', 'English', 'French', 'Japanese']
    theirclasses = ["English", "French", "German", "Italian", "Spanish"]
    classes = []
    ptmodel = ''
    settouse = 'theirs'
    if settouse == 'theirs':
        ptmodel = 'trained_model_3_state.pt'
        classes = theirclasses
    else:
        ptmodel = 'cnn_model_trained.pt'
        classes = ourclasses


    audio_file = glob('input/inputaudio/*.wav')

    transformer = tf.Compose([tf.Resize([64,64]), tf.ToTensor()])
    clip_data, sr = librosa.load(audio_file[0])
    fig = plt.figure(figsize=[0.75,0.75])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    S = librosa.feature.melspectrogram(y=clip_data,
                                sr=sr,
                                n_mels=128 * 2,)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S, sr=sr)

    filename = 'input/inputmelspec/inputmelspectogram'
    #print(filename) # for code testing
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close('all')

    trained_model = CNN_model_3(opt_fun=torch.optim.Adam, lr=0.001)
    trained_model.load_state_dict(torch.load(ptmodel, map_location=torch.device('cpu')))

    image = Image.open('input/inputmelspec/inputmelspectogram.png').convert('RGB')
    image = transformer(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)

    trained_model.eval()
    output = trained_model(image)
    _, predicted = torch.max(output, dim=1)

    print(classes[predicted[0].item()])
    

if __name__ =='__main__':
    predict_language()