if __name__ == '__main__':
    import argparse
    from os import listdir
    import os
    import json
    import re
    from os.path import isfile, join

    import numpy as np
    import json

    from allennlp.common.util import import_module_and_submodules as import_submodules
    from allennlp.models.archival import load_archive

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', action='store')
    args = parser.parse_args()

    import_submodules("allennlp_lib")

    model_path = args.model_path

    archive = load_archive(model_path + '/model.tar.gz')
    model = archive.model
    weight = model._classification_layer.weight.detach().numpy()
    bias = model._classification_layer.bias.detach().numpy()
    # weight = model.model.classifier.out_proj.weight.detach()
    # bias = model.model.classifier.out_proj.bias.detach()
    # label_vocab = model.vocab.get_token_to_index_vocabulary("labels")

    label_vocab = model.vocab.get_token_to_index_vocabulary("labels")

    np.save(model_path + "/w", weight)
    np.save(model_path + "/b", bias)
    with open(model_path + "/label2index.json", "w") as f:
        json.dump(label_vocab, f)
