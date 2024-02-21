import datasets
from datasets.config import importlib_metadata, version
import evaluate

import numpy as np

from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))

_CITATION = """\

"""

_DESCRIPTION = """\
Molecular Property regression metrics.
"""

_KWARGS_DESCRIPTION = """
Args:
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Property_Regression(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/blender-nlp/MolT5/blob/main/evaluation/text_translation_metrics.py"],
            reference_urls=[
                "https://github.com/blender-nlp/MolT5"
            ],
        )

    def _download_and_prepare(self, dl_manager):
        import nltk

        nltk.download("wordnet")
        if NLTK_VERSION >= version.Version("3.6.5"):
            nltk.download("punkt")
        if NLTK_VERSION >= version.Version("3.6.6"):
            nltk.download("omw-1.4")

    def _compute(self, predictions, references, tsv_path="tmp.tsv"):
        references = [references[i][0] for i in range(len(references))]

        maes = []
        mses = []

        for gt, out in zip(references, predictions):
            maes.append(abs(float(gt) - float(out)))
            mses.append((float(gt) - float(out))**2)

        MAE = np.mean(maes)
        MSE = np.mean(mses)

        return {
            "MAE": MAE,
            "MSE": MSE
        }