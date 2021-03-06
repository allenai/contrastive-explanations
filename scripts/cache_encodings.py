
if __name__ == '__main__':
    import argparse
    from os import listdir
    import os
    from nltk.tree import Tree
    import json
    import re
    from os.path import isfile, join
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', action='store')
    parser.add_argument('-o', '--output-path', action='store')
    parser.add_argument('-m', '--model-path', action='store')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    with open(args.model_path + "/label2index.json", "r") as f:
        label_dict = json.load(f)

    instance = None
    label = None
    # cls_vectors = []
    encoded_representations = []
    labels = []
    preds = []
    with open(args.input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    assert instance is not None and label is not None

                    # cls_vectors.append(label['hidden_layer12_cls'])
                    encoded_representations.append(label['encoded_representations'])
                    if 'label' in instance:
                        labels.append(label_dict[instance['label']])
                    preds.append(label_dict[label['label']])

                    print(len(preds))

                    instance = None
                    label = None

                if line.startswith('input'):
                    line = line[line.index(':')+1:].strip()
                    instance = json.loads(line)
                elif line.startswith('prediction'):
                    line = line[line.index(':')+1:].strip()
                    label = json.loads(line)

    split_name = os.path.basename(args.input_path).split('.')[0]
    # np.save(args.output_path + f"/{split_name}_cls", np.array(cls_vectors))
    np.save(args.output_path + f"/{split_name}_encoded_representations", np.array(encoded_representations))
    if labels:
        np.save(args.output_path + f"/{split_name}_labels", np.array(labels))
    np.save(args.output_path + f"/{split_name}_predictions", np.array(preds))
