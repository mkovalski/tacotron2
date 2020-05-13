#!/usr/bin/env python

# Format the speech commands dataset to be used for Tacotron

import os
import sys
import math
import numpy as np
import argparse

parser = argparse.ArgumentParser("Format speech commmands dataset for TacoTron GAN")
parser.add_argument('--datadir', type = str, required = True, help = "Root location of the speech commands dataset folder")
parser.add_argument('--dest', type = str, default = '../filelists/', help = "Where to copy the output file to")
args = parser.parse_args()

assert(os.path.isdir(args.dest)), \
    "Make sure {} exists".format(args.dest)

TRAIN = 0.9
VAL = 0.05
TEST = 0.05

np.random.seed(0)

assert(os.path.isdir(args.datadir)), "Provide valid path to speech commands directory"

paths = [x for x in os.listdir(args.datadir) if '_background_noise_' not in x and os.path.isdir(os.path.join(args.datadir, x))]
print("Going through {}".format(paths))

train = []
val = []
test = []
for word in paths:
    curr_dir = os.path.abspath(os.path.join(args.datadir, word))
    files = [x for x in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, x))]
    indices = np.arange(0, len(files))
    np.random.shuffle(indices)

    train_stop = math.floor(len(indices) * TRAIN)
    val_stop = train_stop + math.floor(len(indices) * VAL)

    for idx in indices[0:train_stop]:
        train.append(os.path.join(curr_dir, files[idx]) + '|' + word + '\n')

    for idx in indices[train_stop:val_stop]:
        val.append(os.path.join(curr_dir, files[idx]) + '|' + word + '\n')

    for idx in indices[val_stop:]:
        test.append(os.path.join(curr_dir, files[idx]) + '|' + word + '\n')
    
train = np.asarray(train)
val = np.asarray(val)
test = np.asarray(test)
 
with open(os.path.join(args.dest, 'sc_audio_text_train_filelist.txt'), 'a') as myFile:
    indices = np.arange(0, len(train))
    np.random.shuffle(indices)
    for idx in indices:
        myFile.write(train[idx])

with open(os.path.join(args.dest, 'sc_audio_text_val_filelist.txt'), 'a') as myFile:
    indices = np.arange(0, len(val))
    np.random.shuffle(indices)
    for idx in indices:
        myFile.write(val[idx])

with open(os.path.join(args.dest, 'sc_audio_text_test_filelist.txt'), 'a') as myFile:
    indices = np.arange(0, len(test))
    np.random.shuffle(indices)
    for idx in indices:
        myFile.write(test[idx])
    
     
 
