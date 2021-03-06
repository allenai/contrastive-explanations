if __name__ == '__main__':
    import argparse
    import json
    import numpy as np
    import json
    import os
    import json
    import pandas as pd
    import spacy

    from spacy.tokenizer import Tokenizer
    from spacy.lang.en import English

    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', action='store')
    parser.add_argument('--concept-path', action='store')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.concept_path)):
        os.makedirs(os.path.dirname(args.concept_path))

    with open(args.data_path) as f:
        data = [json.loads(line) for line in f if line.strip() if line.strip()]

    pretok = lambda sen: sen[:-1] if sen[-1] == "." else sen
    preproc = lambda sen: set([str(w) for w in tokenizer(pretok(sen).lower())])
    preproc_sw = lambda sen: set([str(w) for w in sen if nlp.vocab[w].is_stop == False])

    def is_overlap(p, h):
        prem_tokens = preproc_sw(preproc(p))
        hyp_tokens = preproc_sw(preproc(h))
        overlap = hyp_tokens.intersection(prem_tokens)
        if len(prem_tokens) == 0 and len(hyp_tokens) == 0:
            return 1, True, True
        frac = len(overlap) / len(hyp_tokens) if hyp_tokens else len(overlap) / len(prem_tokens)

        return frac, len(overlap) == len(hyp_tokens), len(overlap) == len(prem_tokens)

    for i, ex in enumerate(data):
        frac, ovh, ovp = is_overlap(ex['sentence1'], ex['sentence2'])
        data[i]['overlap_full_h'] = ovh

    concept = np.array([1 if e['overlap_full_h'] else 0 for e in data], dtype=int)

    np.save(args.concept_path, concept)
