#!/usr/bin/env python

# mhk2160 - Sample some ground truth words 
# This will create examples of MFS for a word in our speech commands dataset

import argparse
from data_utils import TextMelLoader
from hparams import create_hparams
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--fp', type = str)
parser.add_argument('--word', type = str)

args = parser.parse_args()

tml = TextMelLoader(args.fp, create_hparams())
with open(args.fp, 'r') as myFile:
    lines = myFile.readlines()

wav_files = []
for line in lines:
    if line.split('|')[1].strip() == args.word:
        wav_files.append(line.split('|')[0])

if len(wav_files) == 0:
    print("No such word {} found!".format(args.word))
    sys.exit(1)

wav_files = np.asarray(wav_files)
np.random.shuffle(wav_files)

fig, axs = plt.subplots(1, 3, figsize = (8, 4))

for idx, f in enumerate(wav_files[0:3]):
    mel = tml.get_mel(f)
    axs[idx].imshow(mel)

fig.suptitle("Ground Truth MFS: " + args.word)
plt.savefig(args.word)
