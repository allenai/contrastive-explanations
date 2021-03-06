#!/usr/bin/env bash

GPU_IDX=0

for DATASET in bios mnli; do
#    MODEL_PATH_BASE="../contrastive/s3-link/clean"
    MODEL_PATH="experiments/models/${DATASET}/roberta-large"

    CUDA_VISIBLE_DEVICES=${GPU_IDX} allennlp train configs/${DATASET}.jsonnet \
     -s ${MODEL_PATH} -f \
     --include-package=allennlp_lib

#    mkdir ${MODEL_PATH}/encodings
#    find ${MODEL_PATH} -name "*.th" -delete

    python scripts/cache_linear_classifier.py --model-path=${MODEL_PATH}
done
