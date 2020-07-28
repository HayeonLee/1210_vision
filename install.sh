#!/bin/bash

conda deactivate
conda env create -f env.yml
conda activate 0731_vision
python train.py
