# Contrastive Explanations for Model Interpretability

### This is the repository for the paper "Contrastive Explanations for Model Interpretability", about explaining neural model decisions *against alternative decisions*.

#### Authors: Alon Jacovi, Swabha Swayamdipta, Shauli Ravfogel, Yanai Elazar, Yejin Choi, Yoav Goldberg

### Getting Started

#### Setup
```bash
conda create -n contrastive python=3.8
conda activate contrastive
pip install allennlp==1.2.0rc1
pip install allennlp-models==1.2.0rc1.dev20201014
pip install jupyterlab
pip install pandas
bash scripts/download_data.sh
```

#### Contrastive projection

If you're here just to know how we implemented contrastive projection, here it is:
```python
u = classifier_w[fact_idx] - classifier_w[foil_idx]
contrastive_projection = np.outer(u, u) / np.dot(u, u)
```
Very simple :)

`contrastive_projection` is a projection matrix that projects the model's latent representation of some example `h` into the direction of `h` that separates the logits of the fact and foil.

#### Training MNLI/BIOS models
```bash
bash scripts/train_sequence_classification.sh 
```

#### Highlight ranking (Sections 4.3, 5.3)
Run the `notebooks/mnli-highlight-featurerank.ipynb` or `notebooks/bios-highlight-featurerank.ipynb` jupyter notebooks.

These notebooks load the respective models, and then run the highlight ranking procedure.

#### Foil ranking (Section 4.1)

First, cache the model's encodings of the dev set examples:
```bash
bash scripts/cache_encodings_bios.sh
```
Then run the `notebooks/bios-highlight-foilrank.ipynb` notebook.

#### Contrastive decision making (Section 4.4)
First, cache the model's encodings of the dev set examples (skip if already executed):
```bash
bash scripts/cache_encodings_bios.sh
```

Then run the `notebooks/bios-foilpower.ipynb` notebook.

#### Foil ranking for BIOS concepts (Section 4.2)

First, generate concept labels as a numpy matrix from the BIOS dataset:
```bash
python scripts/bios_concepts.py --data-path data/bios/train.jsonl --concept-path experiments/models/bios/roberta-large/concepts/gender-male/train
python scripts/bios_concepts.py --data-path data/bios/dev.jsonl --concept-path experiments/models/bios/roberta-large/concepts/gender-male/dev
python scripts/bios_concepts.py --data-path data/bios/test.jsonl --concept-path experiments/models/bios/roberta-large/concepts/gender-male/test
```

Then, run Amnesic Probing:

* WIP - to be added soon. Alternatively, refer to [the original amnesic probing repository](https://github.com/yanaiela/amnesic_probing) which has the necessary code.


#### Foil ranking for MNLI concepts (Section 5.2)

##### Overlap concept:

First, generate concept labels as a numpy matrix from the BIOS dataset:

```bash
python scripts/mnli_concepts.py --data-path data/mnli/train.jsonl --concept-path experiments/models/mnli/roberta-large/concepts/overlap/train
python scripts/mnli_concepts.py --data-path data/mnli/dev.jsonl --concept-path experiments/models/mnli/roberta-large/concepts/overlap/dev
python scripts/mnli_concepts.py --data-path data/mnli/test.jsonl --concept-path experiments/models/mnli/roberta-large/concepts/overlap/test
```
Then, run Amnesic Probing:

* WIP - to be added soon. Alternatively, refer to [the original amnesic probing repository](https://github.com/yanaiela/amnesic_probing) which has the necessary code.

##### Negation concept:

The examples we used for the negation concept analysis are:
```
data/nli_negation_concept/entailment.jsonl  # entailment instances
data/nli_negation_concept/entailment_with_negation.jsonl  # the above entailment instances, paraphrased with negation words
data/nli_negation_concept/neutral.jsonl  # neutral instances
data/nli_negation_concept/neutral_with_negation.jsonl  # the above neutral instances, paraphrased with negation words
```

To analyze them with respect to the trained MultiNLI model, run the notebook `notebooks/mnli-negation-foilrank.ipynb`.


