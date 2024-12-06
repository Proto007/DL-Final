# Code refactored from https://github.com/minzwon/sota-music-tagging-models/blob/master/preprocessing/mtat_read.py
# Script to convert the mp3 files to npy format
import os
import numpy as np
import glob
import librosa
import tqdm

SAMPLING_RATE = 16000

# assume that the extracted mp3 files are located at 'data/mp3'
files = glob.glob(os.path.join('data','mp3','*/*.mp3'))

# create and set the path for the npy files to be created from this script
npy_path = os.path.join('data','npy')
if not os.path.exists(npy_path):
    os.makedirs(npy_path)

# iterate through the mp3 files
for file in tqdm.tqdm(files):
    # new path and name of the file
    npy_fn = os.path.join(npy_path,file.split('/')[-1][:-3]+'npy')
    # if the file is not already converted to npy, convert the file
    if not os.path.exists(npy_fn):
        try:
            curr, _ = librosa.core.load(file,sr=SAMPLING_RATE)
            np.save(open(npy_fn, 'wb'), curr)
        except:
            # skip corrupted files
            print(file)
            continue

