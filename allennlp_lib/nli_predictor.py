from typing import List, Dict

import numpy
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField


@Predictor.register("textual_entailment_fixed")
class TextualEntailmentPredictorFixed(Predictor):
    """
    Predictor for the [`DecomposableAttention`](../models/decomposable_attention.md) model.

    Registered as a `Predictor` with name "textual_entailment".
    """

    def predict(self, premise: str, hypothesis: str) -> JsonDict:
        """
        Predicts whether the hypothesis is entailed by the premise text.

        # Parameters

        premise : `str`
            A passage representing what is assumed to be true.

        hypothesis : `str`
            A sentence that may be entailed by the premise.

        # Returns

        `JsonDict`
            A dictionary where the key "label_probs" determines the probabilities of each of
            [entailment, contradiction, neutral].
        """
        return self.predict_json({"sentence1": premise, "sentence2": hypothesis})

    # def predict_json(self, js: JsonDict) -> JsonDict:
    #     return self.predict_json(js)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"premise": "...", "hypothesis": "..."}`.
        """
        premise_text = json_dict["sentence1"]
        hypothesis_text = json_dict["sentence2"]
        return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        # Skip indexing, we have integer representations of the strings "entailment", etc.
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
