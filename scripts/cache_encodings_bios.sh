#!/usr/bin/env bash

CUDA_ID=3

for DATASET in bios; do
    MODEL_PATH="experiments/models/${DATASET}/roberta-large"
    OUTPUT_PATH=${MODEL_PATH}
#    OUTPUT_PATH="experiments/models"

    for SPLIT in dev; do
        DATA_PATH="data/${DATASET}/${SPLIT}.jsonl"
        CUDA_VISIBLE_DEVICES=${CUDA_ID} \
        allennlp predict ${MODEL_PATH}/model.tar.gz \
         ${DATA_PATH} \
         --overrides "{model: {output_hidden_states: true}}" \
         --predictor=jsonl_predictor \
         --include-package=allennlp_lib > ${OUTPUT_PATH}/encodings/predicted_${DATASET}_${SPLIT}.txt \
         --cuda-device 0

        python scripts/cache_encodings.py \
         -i=${OUTPUT_PATH}/encodings/predicted_${DATASET}_${SPLIT}.txt \
         -o=${OUTPUT_PATH}/encodings \
         -m=${MODEL_PATH}
    done

    for SPLIT in dev; do
        DATA_PATH="data/${DATASET}/${SPLIT}.jsonl"
        CUDA_VISIBLE_DEVICES=${CUDA_ID} \
        allennlp predict ${MODEL_PATH}/model.tar.gz \
         ${DATA_PATH} \
         --overrides "{model: {output_hidden_states: true}}" \
         --predictor=bios_masked_predictor \
         --include-package=allennlp_lib > ${OUTPUT_PATH}/encodings/predicted_${DATASET}_${SPLIT}_masked.txt \
         --cuda-device 0

        python scripts/cache_encodings.py \
         -i=${OUTPUT_PATH}/encodings/predicted_${DATASET}_${SPLIT}_masked.txt \
         -o=${OUTPUT_PATH}/encodings \
         -m=${MODEL_PATH}
    done
done
