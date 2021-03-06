from typing import List, Dict

import numpy
import logging
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField


logger = logging.getLogger(__name__)

@Predictor.register("bios_masked_predictor")
class BiosIrrelevantPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"premise": "...", "hypothesis": "..."}`.
        """
        # premise_text = json_dict["Sentence1"]
        # hypothesis_text = json_dict["Sentence2"]

        # logger.info(str(json_dict))
        # logger.info(list(enumerate(json_dict['text'].split())))
        # logger.info(list(enumerate(json_dict['text_without_gender'].split())))

        text = json_dict["text_without_gender"]
        text = text.split()
        for i in json_dict['gender_tokens']:
            text[i] = text[i].replace('_', '<mask>')
        text = ' '.join(text)
        logger.info(text)

        # text = json_dict["text"]
        # new_text = text.split()
        # for i, w in enumerate(text.split()):
        #     if i not in json_dict['gender_tokens']:
        #         new_text[i] = '<mask>'
        # text = ' '.join(new_text)

        # logger.info(text)

        return self._dataset_reader.text_to_instance(text)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        # Skip indexing, we have integer representations of the strings "entailment", etc.
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
