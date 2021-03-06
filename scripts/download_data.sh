#!/usr/bin/env bash

mkdir -p $(pwd)/data/mnli
curl -Lo $(pwd)/data/mnli/train.jsonl https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_train.jsonl
curl -Lo $(pwd)/data/mnli/dev.jsonl https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_matched.jsonl
curl -Lo $(pwd)/data/mnli/test.jsonl https://allennlp.s3.amazonaws.com/datasets/multinli/multinli_1.0_dev_mismatched.jsonl

mkdir -p $(pwd)/data/bios
curl -Lo $(pwd)/data/bios/train.pickle https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle
curl -Lo $(pwd)/data/bios/dev.pickle https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle
curl -Lo $(pwd)/data/bios/test.pickle https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle

python scripts/bios_pickle_to_jsonl.py
