import json
import csv
import os
import pickle

for split in ["train", "dev", "test"]:
    path1 = f"data/bios/{split}.pickle"
    path2 = f"data/bios/{split}.jsonl"

    def find_idx(d: dict):
        with_gender, without_gender = d["hard_text"], d["text_without_gender"]
        masked_gender = []

        with_gender_lst, without_gender_lst = with_gender.split(" "), without_gender.split(" ")
        gender_idx = [i for i, (w_g, w_ng) in enumerate(zip(with_gender_lst, without_gender_lst)) if
                      "_" in w_ng and "_" not in w_g]
        return gender_idx

    if not os.path.exists(os.path.dirname(path2)):
        os.makedirs(os.path.dirname(path2))

    with open(path1, "rb") as f:
        data = pickle.load(f)

    with open(path2, 'w') as fo:
        for line in data:
            if not line:
                continue

            if line['p'] == 'model':
                continue

            indices = find_idx(line)
            # if 'http' not in line['text']:
            #     continue
            # print(line['text'])
            # print(line['text_without_gender'])
            # for i, w in enumerate(line['text_without_gender'].split()):
            #     if i in indices:
            #         print(w)

            ex = {}
            ex['text'] = line['hard_text']
            ex['original_text'] = line['text']
            ex['text_without_gender'] = line['text_without_gender']
            ex['label'] = line['p']
            ex['gender'] = line['g']
            ex['start'] = line['start']
            ex['gender_tokens'] = indices

            fo.write(json.dumps(ex) + "\n")

