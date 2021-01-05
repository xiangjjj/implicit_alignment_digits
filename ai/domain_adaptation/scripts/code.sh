#!/bin/sh

rsync -avzh  \
    --exclude '*.pyc'  --exclude tensorboard \
    --exclude __pycache__ --exclude '*.pt' --exclude '.git' \
    --exclude '*.pkl' \
    /Users/xjiang/GoogleDrive/Colab_Experiments/meta-domain-adaptation \
    ws3:/export/home/xiang.jiang
