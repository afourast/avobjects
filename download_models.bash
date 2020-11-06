#!/bin/bash

mkdir checkpoints
cd checkpoints

wget --timestamping http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch
wget --timestamping http://www.robots.ox.ac.uk/~vgg/research/avobjects/pretrained_models/avobjects_loc_sep.pt