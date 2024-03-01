import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import random

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

def create_melspecs():
    audio_files = glob('Dataset/*.wav')
    random.shuffle(audio_files)
    audio_files = audio_files[0:50] # shortened dataset for testing
    twenty_percent = int(len(audio_files) * 0.2)
    counter = 0

    for aud in audio_files:
        clip_data, sr = librosa.load(aud)
        
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

        if counter <= twenty_percent:
            # filename = 'melspecs/' + aud[8:10] + '/' + aud[8:15]
            filename = 'melspecs/test/' + aud[8:10] +  '/' + aud[8:15]
            print(filename) # for code testing
        else:
            filename = 'melspecs/train/' + aud[8:10] + '/' + aud[8:15]
            print(filename) # for code testing

        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')

        counter += 1

if __name__ == '__main__':
    create_melspecs()


