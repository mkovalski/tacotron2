#!/bin/bash

wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
mkdir -p ../data/speech_commands/
mv speech_commands_v0.01.tar.gz ../data/speech_commands/
cd ../data/speech_commands/
tar -xzvf speech_commands_v0.01.tar.gz
