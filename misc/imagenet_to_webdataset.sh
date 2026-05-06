#!/bin/bash

# Train
python /c/Projects/AI/new-project/misc/imagenet_to_webdataset.py \
  /c/data/IMAGENET1K \
  /c/data/IMAGENET1K_tar \
  --split train \
  --samples-per-shard 2000

# Validation  
python /c/Projects/AI/new-project/misc/imagenet_to_webdataset.py \
  /c/data/IMAGENET1K \
  /c/data/IMAGENET1K_tar \
  --split val \
  --samples-per-shard 500

# Test (unlabeled, class_id = -1)
python /c/Projects/AI/new-project/misc/imagenet_to_webdataset.py \
  /c/data/IMAGENET1K \
  /c/data/IMAGENET1K_tar \
  --split test \
  --samples-per-shard 1000