1. mhk2160 - Michael Kovalski
2. 5/13/2020
3. Using Conditional GANs for Speech Synthesis
4. Over the past few years, neural network based models have been used to make great strides in Text to Speech (TTS) synthesis problems. However, these models are usually able to use create synthetic speech based on a single speaker or from a collection of pre- defined speakers using database methods. This can often mimic a discrete number of speakers, but lacks the ability to generalize towards new, unseen voices. This work aims to use the Generative Adversarial Network (GAN) framework to create a variety of different voices by learning features such as voice, tone, and speed. In this work, we will build on an existing state of the art text to speed model, Tacotron2, and use an adversarial framework to generate a number of voices conditioned on text.
5. Tools - Python (3.6 was used), PyTorch with CUDA support, librosa. Full list of dependencies can be found in requirements.txt file.
    To install, either run `pip install -r requirements.txt` to install all files, or create an anaconda environment by running `conda env 
6. Directories: tacotron2. cd into data_prep and run ./download_sc.sh and python create_sc.py to create dataset. Then cd ../ and run `python train.py -o outputdir -l logdir --n_gpus 1`
7. See step 6 for running details. A full overview can be found in the README.md file. Basic overview
    cd tacotron2/
    cd data_prep/
    ./download_sc.sh
    python create_sc.py --datadir ../data/speech_commands/ --dest ../filelists/
    cd ../
    python train.py -o outputdir -l logdir --n_gpus 1
    
The above shows the pipeline of scripts to run, below is a description of the scripts:

    1. ./download_sc.sh downloads the speech commands dataset and puts it in the correct directory
    2. create_sc.py creates training, validation, and test splits to be input to tacotrons and adds these files to fileists/ directory under tacotron2/
    3. train.py will run the training for predicting mel frequency spectrogram features from text input, a one to many mapping in this case

After training starts, you may view the progress by entering the directory of your output (in this case, outputdir) and running:
    tensorboard --logdir=$PWD --port=5000

Please replace port "5000" with a port you can use.

Datasets used are the speech_commands dataset, which is available by downloading via the above script

Files which have been modified from original tacotron2 code. All code should have the #mhk2160 comment where code was added:

 - model.py
 - hparams.py
 - train.py
 - inference.ipynb
 - logger.py
 - generate_mfs.py

The easiest way to obtain this code is by simply cloning my fork of the repo and using my branch:
    - git clone https://github.com/mkovalski/tacotron2
    - git checkout gan
