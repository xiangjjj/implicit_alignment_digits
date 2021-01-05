#!/bin/sh

rsync -avzh gpu:/home/xjiang/meta-domain-adaptation/ai/domain_adaptation/makefiles/tensorboard \
    /Users/xjiang/GoogleDrive/Colab_Experiments/meta-domain-adaptation/ai/domain_adaptation/makefiles

rsync -avzh ws3:/data/TransientData/xiang/tensorboard \
    /Users/xjiang/GoogleDrive/Colab_Experiments/meta-domain-adaptation/ai/domain_adaptation/makefiles
