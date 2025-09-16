#!/bin/bash

python main.py fit --config configs/lora/cifar100-r4-lr-0.05.yaml | tee logs/lora1000.log
python main.py fit --config configs/full/cifar100-lr=0.05.yaml | tee logs/full1000.log
