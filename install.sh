#!/bin/bash

conda deactivate
conda env create -f env.yml
conda activate 0731_vision
pip install 'tensorflow-estimator<1.15.0rc0,>=1.14.0rc0' --force-reinstall
pip install -U gast==0.2.2
python train.py

