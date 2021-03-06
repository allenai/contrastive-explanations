if __name__ == '__main__':
    import argparse
    import json
    import numpy as np
    import json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', action='store')
    parser.add_argument('--concept-path', action='store')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.concept_path)):
        os.makedirs(os.path.dirname(args.concept_path))

    index2label = {0: 'professor',
                   1: 'physician',
                   2: 'attorney',
                   3: 'photographer',
                   4: 'journalist',
                   5: 'psychologist',
                   6: 'nurse',
                   7: 'teacher',
                   8: 'dentist',
                   9: 'surgeon',
                   10: 'architect',
                   11: 'painter',
                   12: 'filmmaker',
                   13: 'software_engineer',
                   14: 'poet',
                   15: 'accountant',
                   16: 'composer',
                   17: 'dietitian',
                   18: 'pastor',
                   19: 'chiropractor',
                   20: 'comedian',
                   21: 'paralegal',
                   22: 'interior_designer',
                   23: 'yoga_teacher',
                   24: 'dj',
                   25: 'personal_trainer',
                   26: 'rapper'}
    label2index = {index2label[k]: k for k in index2label}

    with open(args.data_path) as f:
        data = [json.loads(line) for line in f if line.strip() if line.strip()]

    concept = np.array([ex['gender'] == 'm' for ex in data], dtype=int)

    np.save(args.concept_path, concept)
