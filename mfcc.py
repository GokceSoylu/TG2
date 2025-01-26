import librosa
import numpy as np
import os

def extract_mfcc(file_path, n_mfcc=10, max_pad_len=49):
    audio, sample_rate = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

# Tüm veriler için MFCC özelliklerini hesapla
dataset_path = "/Users/gokcesoylu/Desktop/custom_dataset"
X = []
y = []

label_map = {"yes": 0, "no": 1, "up": 2, "down": 3, "go": 4}

for label, command in label_map.items():
    command_path = os.path.join(dataset_path, label)
    for file in os.listdir(command_path):
        file_path = os.path.join(command_path, file)
        mfcc = extract_mfcc(file_path)
        X.append(mfcc)
        y.append(command)

X = np.array(X)
y = np.array(y)
