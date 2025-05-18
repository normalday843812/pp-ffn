#!/bin/bash
# This script downloads the GSM8K dataset and preprocesses it for internalization of chain-of-thought (CoT) reasoning.
mkdir -p data
mkdir -p data/gsm8k

wget https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/train.txt -O data/gsm8k/gsm_train.txt
wget https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/valid.txt -O data/gsm8k/gsm_valid.txt
wget https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/test.txt -O data/gsm8k/gsm_test.txt

for split in train valid test; do
  python preprocessing/gsm_icot.py ${split}
  rm data/gsm8k/gsm_${split}.txt
done