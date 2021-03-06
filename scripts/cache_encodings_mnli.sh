#!/usr/bin/env bash

CUDA_ID=0

for DATASET in mnli; do
    MODEL_PATH="experiments/models/${DATASET}/roberta-large"
    OUTPUT_PATH=${MODEL_PATH}

     for SPLIT in dev; do
        DATA_PATH="data/${DATASET}/${SPLIT}.jsonl"

        CUDA_VISIBLE_DEVICES=${CUDA_ID} \
        allennlp predict ${MODEL_PATH}/model.tar.gz \
         ${DATA_PATH} \
         --overrides "{model: {output_hidden_states: true}}" \
         --predictor=textual_entailment_fixed \
         --include-package=allennlp_lib > ${OUTPUT_PATH}/encodings/predicted_${DATASET}_${SPLIT}.txt \
         --cuda-device 0

        python scripts/cache_encodings.py \
         -i=${OUTPUT_PATH}/encodings/predicted_${DATASET}_${SPLIT}.txt \
         -o=${OUTPUT_PATH}/encodings \
         -m=${MODEL_PATH}
    done
done
