from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
import torch
from data_provider.stage3_dm import (
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    REACTION_BENCHMARKS,
)
import model.added_tokens as added_tokens
import re
from Levenshtein import distance as lev


def caption_evaluate(predictions, targets, tokenizer, prompts):
    meteor_scores = []
    references = []
    hypotheses = []
    failure_idxs = []

    patterns = {
        "DESCRIPTION": {
            "dual_side": re.compile(r"(?<=<DESCRIPTION>).*?(?=</DESCRIPTION>)"),
            "left_side": re.compile(r"(?<=<DESCRIPTION>).*"),
        },
        "IUPAC": {
            "dual_side": re.compile(r"(?<=<IUPAC>).*?(?=</IUPAC>)"),
            "left_side": re.compile(r"(?<=<IUPAC>).*"),
        },
        "MOLFORMULA": {
            "dual_side": re.compile(r"(?<=<MOLFORMULA>).*?(?=</MOLFORMULA>)"),
            "left_side": re.compile(r"(?<=<MOLFORMULA>).*"),
        },
    }

    for i in range(len(targets)):
        target = targets[i]
        prediction = predictions[i]

        pattern = None
        for key, matching_pattern in patterns.items():
            if matching_pattern["left_side"].search(targets[i]):
                pattern = matching_pattern
                break
        if pattern is None:
            print(targets[i])
            continue
        assert pattern is not None

        if pattern["dual_side"].search(target):
            ref = pattern["dual_side"].search(target).group()
        else:
            ref = pattern["left_side"].search(target).group()
        ref_tokens = tokenizer.tokenize(ref, truncation=False, padding="longest")

        try:
            if pattern["dual_side"].search(prediction):
                pred = pattern["dual_side"].search(prediction).group()
            else:
                pred = pattern["left_side"].search(prediction).group()
            pred_tokens = tokenizer.tokenize(pred, truncation=False, padding="longest")

            references.append([ref_tokens])
            hypotheses.append(pred_tokens)
            mscore = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(mscore)

        except:
            failure_idxs.append(i)
            pred = None
            pred_tokens = None

    if hypotheses:
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        bleu2 *= 100
        bleu4 *= 100
    else:
        bleu2 = 0
        bleu4 = 0

    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    rouge_scores = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    rouge_1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores]) * 100
    evaluation_results = {
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": rouge_1,
        "rouge2": rouge_2,
        "rougeL": rouge_l,
        "meteor": _meteor_score,
    }
    failed_cases = {
        "predictions": [predictions[i] for i in failure_idxs],
        "targets": [targets[i] for i in failure_idxs],
        "prompts": [prompts[i] for i in failure_idxs],
    }
    return evaluation_results, failed_cases


from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies


def molecule_evaluate(predictions, targets, tokenizer, prompts, morgan_r=2):
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    levs = []
    morgan_r = 2

    failure_idxs = []
    exact_matches = []
    ref_selfies_list = []
    ref_smiles_list = []
    pred_selfies_list = []
    pred_smiles_list = []

    for i in tqdm(range(len(targets))):
        target = targets[i].replace(" ", "")
        prediction = predictions[i].replace(" ", "")

        if re.search(r"(?<=<SELFIES>).*?(?=</SELFIES>)", target):
            target_selfies = re.search(
                r"(?<=<SELFIES>).*?(?=</SELFIES>)", target
            ).group()
        else:
            target_selfies = re.search(r"(?<=<SELFIES>).*", target).group()
        target_smiles = selfies.decoder(target_selfies)
        target_mol = Chem.MolFromSmiles(target_smiles)
        target_canonical_smiles = Chem.CanonSmiles(target_smiles)
        target_canonical_selfies = selfies.encoder(target_canonical_smiles)

        try:
            if re.search(r"(?<=<SELFIES>).*?(?=</SELFIES>)", prediction) is not None:
                prediction_selfies = re.search(
                    r"(?<=<SELFIES>).*?(?=</SELFIES>)", prediction
                ).group()
            else:
                prediction_selfies = re.search(r"(?<=<SELFIES>).*", prediction).group()

            assert (
                "<SELFIES>" not in prediction_selfies
                and "</SELFIES>" not in prediction_selfies
            )

            prediction_smiles = selfies.decoder(prediction_selfies)
            prediction_mol = Chem.MolFromSmiles(prediction_smiles)
            prediction_canonical_smiles = Chem.CanonSmiles(prediction_smiles)
            prediction_canonical_selfies = selfies.encoder(prediction_canonical_smiles)

            exact_matches.append(
                Chem.MolToInchi(target_mol) == Chem.MolToInchi(prediction_mol)
            )
        except:
            failure_idxs.append(i)
            prediction_mol = None
            continue

        if prediction_mol is not None:

            levs.append(lev(target_canonical_smiles, prediction_canonical_smiles))

            pred_selfies = tokenizer.tokenize(
                prediction_canonical_selfies, truncation=False, padding="longest"
            )
            pred_smiles = tokenizer.tokenize(
                prediction_canonical_smiles, truncation=False, padding="longest"
            )
            pred_selfies_list.append(pred_selfies)
            pred_smiles_list.append(pred_smiles)

            ref_selfies = tokenizer.tokenize(
                target_canonical_selfies, truncation=False, padding="longest"
            )
            ref_smiles = tokenizer.tokenize(
                target_canonical_smiles, truncation=False, padding="longest"
            )
            ref_selfies_list.append([ref_selfies])
            ref_smiles_list.append([ref_smiles])

            MACCS_sims.append(
                DataStructs.FingerprintSimilarity(
                    MACCSkeys.GenMACCSKeys(target_mol),
                    MACCSkeys.GenMACCSKeys(prediction_mol),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            RDK_sims.append(
                DataStructs.FingerprintSimilarity(
                    Chem.RDKFingerprint(target_mol),
                    Chem.RDKFingerprint(prediction_mol),
                    metric=DataStructs.TanimotoSimilarity,
                )
            )
            morgan_sims.append(
                DataStructs.TanimotoSimilarity(
                    AllChem.GetMorganFingerprint(target_mol, morgan_r),
                    AllChem.GetMorganFingerprint(prediction_mol, morgan_r),
                )
            )

    validity_ratio = 1 - len(failure_idxs) / len(predictions)
    MACCS_sim = np.mean(MACCS_sims)
    RDK_sim = np.mean(RDK_sims)
    morgan_sim = np.mean(morgan_sims)
    exact_match_ratio = np.mean(exact_matches)
    levenshtein_score = np.mean(levs)

    if pred_smiles_list:
        bleu_smiles = corpus_bleu(
            ref_smiles_list, pred_smiles_list, weights=(0.25, 0.25, 0.25, 0.25)
        )
        bleu_smiles *= 100
    else:
        bleu_smiles = 0

    if pred_selfies_list:
        bleu_selfies = corpus_bleu(
            ref_selfies_list, pred_selfies_list, weights=(0.25, 0.25, 0.25, 0.25)
        )
        bleu_selfies *= 100
    else:
        bleu_selfies = 0

    results = {
        "validity_ratio": validity_ratio,
        "MACCS_FTS": MACCS_sim,
        "RDK_FTS": RDK_sim,
        "morgan_FTS": morgan_sim,
        "exact_match_ratio": exact_match_ratio,
        "levenshtein_score": levenshtein_score,
        "bleu_smiles": bleu_smiles,
        "bleu_selfies": bleu_selfies,
    }
    failed_cases = {
        "predictions": [predictions[i] for i in failure_idxs],
        "targets": [targets[i] for i in failure_idxs],
        "prompts": [prompts[i] for i in failure_idxs],
    }
    return results, failed_cases


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def pad_and_concat(tensor_list, fill_value=0):
    """
    concat the first dimension and pad the second dimension
    tensor_list: [[B (diff), N_num, *], ...]
    """
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype
    max_dim1 = max(t.shape[1] for t in tensor_list)
    sum_dim0 = sum(t.shape[0] for t in tensor_list)
    if len(tensor_list[0].shape) == 3:
        out = torch.full(
            (sum_dim0, max_dim1, tensor_list[0].shape[-1]),
            fill_value=fill_value,
            device=device,
            dtype=dtype,
        )
        i = 0
        for t in tensor_list:
            out[i : i + t.shape[0], : t.shape[1]] = t
            i += t.shape[0]
        return out
    elif len(tensor_list[0].shape) == 2:
        out = torch.full(
            (sum_dim0, max_dim1), fill_value=fill_value, device=device, dtype=dtype
        )
        i = 0
        for t in tensor_list:
            out[i : i + t.shape[0], : t.shape[1]] = t
            i += t.shape[0]
        return out
    raise NotImplementedError()


def get_task_specific_list(predictions, targets, tasks, probs, prompts):
    unique_tasks = list(set(tasks))
    task_specific_predictions = {t: [] for t in unique_tasks}
    task_specific_targets = {t: [] for t in unique_tasks}
    task_specific_probs = {t: [] for t in unique_tasks}
    task_specific_prompts = {t: [] for t in unique_tasks}
    for i, t in enumerate(tasks):
        task_specific_predictions[t].append(predictions[i])
        task_specific_targets[t].append(targets[i])
        task_specific_probs[t].append(probs[i])
        task_specific_prompts[t].append(prompts[i])
    return (
        task_specific_predictions,
        task_specific_targets,
        task_specific_probs,
        task_specific_prompts,
    )


def task_specifically_evaluate(predictions, targets, tasks, probs, prompts, tokenizer):
    # get unique items from all_tasks
    unique_tasks = list(set(tasks))
    # remove tasks_to_be_removed
    tasks_to_be_removed = [
        "smol-name_conversion-i2f/smol-name_conversion-i2f",
        "smol-name_conversion-s2f/smol-name_conversion-s2f",
        "smol-name_conversion-i2s/smol-name_conversion-i2s",
        "smol-name_conversion-s2i/smol-name_conversion-s2i",
    ]

    unique_tasks = [t for t in unique_tasks if t not in tasks_to_be_removed]

    evaluation_results = {task: dict() for task in unique_tasks}

    (
        task_specific_predictions,
        task_specific_targets,
        task_specific_probs,
        task_specific_prompts,
    ) = get_task_specific_list(predictions, targets, tasks, probs, prompts)
    failed_cases = {
        "predictions": [],
        "targets": [],
        "prompts": [],
        "tasks": [],
    }

    for t in task_specific_predictions.keys():
        if t in tasks_to_be_removed:
            continue

        task_predictions = task_specific_predictions[t]
        task_targets = task_specific_targets[t]
        task_probs = task_specific_probs[t]
        task_prompts = task_specific_prompts[t]
        task_name = t.split("/")[0]
        if task_name in CLASSIFICATION_BENCHMARKS:
            results = classification_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                probs=task_probs,
            )
        elif task_name in REGRESSION_BENCHMARKS:
            results, _failed_cases = regression_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                prompts=task_prompts,
            )
        elif (
            task_name in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS
        ):  # output is a molecule
            results, _failed_cases = molecule_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                prompts=task_prompts,
            )
        elif task_name in MOL2TEXT_BENCHMARKS:
            results, _failed_cases = caption_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                prompts=task_prompts,
            )
        else:
            raise NotImplementedError("Task not implemented")
        # update number of instances
        results["num_instances"] = len(task_predictions)
        evaluation_results[t] = results
        if task_name not in CLASSIFICATION_BENCHMARKS:
            for k in _failed_cases.keys():
                failed_cases[k].extend(_failed_cases[k])
            failed_cases["tasks"].extend(
                [t.split("/")[0] for _ in range(len(_failed_cases["predictions"]))]
            )

    return evaluation_results, failed_cases


from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def convert_logit2binary_prob(logits, tokenizer):
    true_token_id = tokenizer.convert_tokens_to_ids(["true"])[0]
    True_token_id = tokenizer.convert_tokens_to_ids(["True"])[0]
    false_token_id = tokenizer.convert_tokens_to_ids(["false"])[0]
    False_token_id = tokenizer.convert_tokens_to_ids(["False"])[0]

    total_probs = torch.zeros(len(logits), 2)
    for i, logit in enumerate(logits):
        probs = logit.softmax(dim=-1)
        # in generated answer, 0 th token is <BOOLEAN> and 1 is the prediction, and 2 is </BOOLEAN>
        true_prob = probs[1, true_token_id] + probs[1, True_token_id]
        false_prob = probs[1, false_token_id] + probs[1, False_token_id]
        # normalize the probability for binary answer
        total_probs[i] = torch.cat(
            [false_prob.unsqueeze(0), true_prob.unsqueeze(0)], dim=0
        ).softmax(-1)
    total_probs = [p.tolist() for p in total_probs]
    return total_probs


def classification_evaluate(predictions, targets, probs):
    probs = [torch.tensor(p) for p in probs]

    total_labels = torch.zeros(len(predictions), dtype=torch.long)

    for i in range(len(predictions)):
        label = int("True" in targets[i] or "true" in targets[i])
        total_labels[i] = label

    probs_unsqueezed = [p.unsqueeze(0) for p in probs]
    total_probs = torch.cat(probs_unsqueezed, dim=0)
    total_preds = total_probs.argmax(dim=-1)

    # Convert tensors to numpy arrays for use with scikit-learn metrics
    total_preds_np = total_preds.numpy()
    total_labels_np = total_labels.numpy()

    # Calculate metrics
    acc = accuracy_score(y_true=total_labels_np, y_pred=total_preds_np)
    f1 = f1_score(y_true=total_labels_np, y_pred=total_preds_np)
    prec = precision_score(y_true=total_labels_np, y_pred=total_preds_np)
    rec = recall_score(y_true=total_labels_np, y_pred=total_preds_np)
    try:
        roc_auc = roc_auc_score(
            y_true=total_labels_np,
            y_score=total_probs[
                :, 1
            ].numpy(),  # Use y_score here because roc_auc_score expects probability scores
        )
    except:
        roc_auc = -1.0

    evaluation_results = {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc,
    }
    return evaluation_results


def regression_evaluate(predictions, targets, prompts):

    total_labels = []
    total_predictions = []
    failure_idxs = []

    for i in range(len(predictions)):
        label = (
            re.search(r"(?<=<FLOAT>).*?(?=</FLOAT>)", targets[i])
            .group()
            .replace(" ", "")
        )
        label = label.replace("<|", "").replace("|>", "")
        label = float(label)

        # only calculate metrics if the prediction is a float
        # else, increment the failure count
        try:
            prediction = (
                re.search(r"(?<=<FLOAT>).*?(?=</FLOAT>)", predictions[i])
                .group()
                .replace(" ", "")
            )
            prediction = prediction.replace("<|", "").replace("|>", "")
            prediction = float(prediction)

            total_labels.append(label)
            total_predictions.append(prediction)
        except:
            failure_idxs.append(i)
    failure_rate = len(failure_idxs) / len(predictions)

    # Calculate regression metrics: mae, mse, rmse
    total_labels = np.array(total_labels)
    total_predictions = np.array(total_predictions)

    mae = np.mean(np.abs(total_labels - total_predictions))
    mse = np.mean((total_labels - total_predictions) ** 2)
    rmse = np.mean((total_labels - total_predictions) ** 2) ** 0.5

    evaluation_results = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "failure_rate": failure_rate,
    }
    failed_cases = {
        "predictions": [predictions[i] for i in failure_idxs],
        "targets": [targets[i] for i in failure_idxs],
        "prompts": [prompts[i] for i in failure_idxs],
    }
    return evaluation_results, failed_cases
