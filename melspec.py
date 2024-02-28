import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

def create_melspecs():
    audio_files = glob('Dataset/*.wav')
    audio_files = audio_files[0:10] # shortened dataset for testing

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

        filename = 'melspecs/' + aud[8:10] + '/' + aud[8:15]
        print(filename)

        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')

create_melspecs()


