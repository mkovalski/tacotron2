#!/bin/bash

wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
mkdir -p ../data/speech_commands
mv speech_commands_v0.01.tar.gz ../data/speech_commands
tar -xzvf ../data/speech_commands/speech_commands_v0.01.tar.gz
