from copy import deepcopy
from typing import List, Dict

from overrides import overrides
import numpy
import json
from nltk.tree import Tree

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField


@Predictor.register("jsonl_predictor")
class TextClassifierPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the :class:`~allennlp.models.basic_classifier.BasicClassifier` model
    """

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json(json.loads(sentence))

    @overrides
    def load_line(self, line: str) -> JsonDict:
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        return json.loads(line)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        sentence = json_dict["text"]

        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = deepcopy(instance)
        label = numpy.argmax(outputs["probs"])

        label_vocab = self.model.vocab.get_token_to_index_vocabulary("labels")
        label_vocab = {v: k for k, v in label_vocab.items()}

        new_instance.add_field("label", LabelField(label_vocab[label]))
        return [new_instance]
