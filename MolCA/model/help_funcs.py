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
import ast
from model.save_only_metrics import Text2Mol_translation
import re


def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(
            gt, truncation=True, max_length=text_trunc_length, padding="max_length"
        )

        gt_tokens = list(filter((tokenizer.pad_token).__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(
            out, truncation=True, max_length=text_trunc_length, padding="max_length"
        )

        out_tokens = list(filter((tokenizer.pad_token).__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    bleu2 *= 100
    bleu4 *= 100

    print("BLEU-2 score:", bleu2)
    print("BLEU-4 score:", bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print("Average Meteor score:", _meteor_score)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print("ROUGE score:")
    rouge_1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores]) * 100
    print("rouge1:", rouge_1)
    print("rouge2:", rouge_2)
    print("rougeL:", rouge_l)
    evaluation_results = {
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": rouge_1,
        "rouge2": rouge_2,
        "rougeL": rouge_l,
        "meteor": _meteor_score,
    }
    return evaluation_results


from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import selfies


def molecule_evaluate(predictions, targets, tokenizer, text_trunc_length, morgan_r=2):
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    morgan_r = 2

    failure_idxs = []
    exact_matches = []

    for i in tqdm(range(len(targets))):
        target = targets[i]
        prediction = predictions[i]
        # <REFACTOR> after re preprocessing molinstrunction reaction prediction dataset, prediction would be smiles.
        target_selfies = (
            target.replace(tokenizer.pad_token, "")
            .replace("[START_I_SMILES]", "")
            .replace("[END_I_SMILES]", "")
        )
        prediction_selfies = (
            prediction.replace(tokenizer.pad_token, "")
            .replace("[START_I_SMILES]", "")
            .replace("[END_I_SMILES]", "")
        )

        try:
            target_smiles = selfies.decoder(target_selfies)
            prediction_smiles = selfies.decoder(prediction_selfies)
            # </REFACTOR>
            target_mol = Chem.MolFromSmiles(target_smiles)
            prediction_mol = Chem.MolFromSmiles(prediction_smiles)

            target_canonical_smiles = Chem.CanonSmiles(target_smiles)
            prediction_canonical_smiles = Chem.CanonSmiles(prediction_smiles)
            if target_canonical_smiles == prediction_canonical_smiles:
                exact_matches.append(True)
            else:
                exact_matches.append(False)

        except:
            failure_idxs.append(i)
            continue

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
    results = {
        "validity_ratio": validity_ratio,
        "MACCS_FTS": MACCS_sim,
        "RDK_FTS": RDK_sim,
        "morgan_FTS": morgan_sim,
        "exact_match_ratio": exact_match_ratio,
    }
    return results, failure_idxs


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


def get_task_specific_list(predictions, targets, tasks, probs):
    unique_tasks = list(set(tasks))
    task_specific_predictions = {t: [] for t in unique_tasks}
    task_specific_targets = {t: [] for t in unique_tasks}
    task_specific_probs = {t: [] for t in unique_tasks}
    for i, t in enumerate(tasks):
        task_specific_predictions[t].append(predictions[i])
        task_specific_targets[t].append(targets[i])
        task_specific_probs[t].append(probs[i])
    return task_specific_predictions, task_specific_targets, task_specific_probs


def task_specifically_evaluate(
    predictions, targets, tasks, probs, tokenizer, text_trunc_length
):

    # get unique items from all_tasks
    unique_tasks = list(set(tasks))
    evaluation_results = {task: dict() for task in unique_tasks}

    task_specific_predictions, task_specific_targets, task_specific_probs = (
        get_task_specific_list(predictions, targets, tasks, probs)
    )

    for t in task_specific_predictions.keys():
        task_predictions = task_specific_predictions[t]
        task_targets = task_specific_targets[t]
        task_probs = task_specific_probs[t]
        if t.split("/")[0] in CLASSIFICATION_BENCHMARKS:
            results = classification_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                probs=task_probs,
                tokenizer=tokenizer,
                text_trunc_length=text_trunc_length,
            )
        elif t.split("/")[0] in REGRESSION_BENCHMARKS:
            results = regression_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                text_trunc_length=text_trunc_length,
            )
        elif (
            t.split("/")[0] in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS
        ):  # output is a molecule
            results, failure_idxs = molecule_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                text_trunc_length=text_trunc_length,
            )
            _task_predictions = [
                task_predictions[i]
                for i in range(len(task_predictions))
                if i not in failure_idxs
            ]
            _task_targets = [
                task_targets[i]
                for i in range(len(task_targets))
                if i not in failure_idxs
            ]

            caption_results = caption_evaluate(
                predictions=_task_predictions,
                targets=_task_targets,
                tokenizer=tokenizer,
                text_trunc_length=text_trunc_length,
            )
            results.update(caption_results)
        elif t.split("/")[0] in MOL2TEXT_BENCHMARKS:
            results = caption_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                text_trunc_length=text_trunc_length,
            )
        else:
            raise NotImplementedError("Task not implemented")
        evaluation_results[t] = results

    return evaluation_results


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
    return total_probs


def classification_evaluate(predictions, targets, probs, tokenizer, text_trunc_length):

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


def regression_evaluate(predictions, targets, tokenizer, text_trunc_length):

    total_labels = []
    total_predictions = []
    failure_count = 0
    tens_order_failure_count = 0

    # reg tokens added
    if "<|" in targets[0]:
        for i in range(len(targets)):
            targets[i] = targets[i].replace("<|", "").replace("|>", "")
            predictions[i] = predictions[i].replace("<|", "").replace("|>", "")

    _total_labels = []
    for i in range(len(predictions)):
        label = targets[i]
        label = ast.literal_eval(
            targets[i]
            .replace(added_tokens.FLOAT[0], "")
            .replace(added_tokens.FLOAT[1], "")
            .replace(tokenizer.pad_token, "")
        )
        _total_labels.append(label)
    _total_labels = np.array(_total_labels)
    label_max_abs = np.max(np.abs(_total_labels))

    for i in range(len(predictions)):
        label = targets[i]
        prediction = predictions[i]

        try:
            label = ast.literal_eval(
                targets[i]
                .replace(added_tokens.FLOAT[0], "")
                .replace(added_tokens.FLOAT[1], "")
                .replace(tokenizer.pad_token, "")
            )
            prediction = re.search("\d*?[.]?\d+(?=</FLOAT>)", prediction).group()
            prediction = float(prediction)

            if prediction > label_max_abs * 10:
                tens_order_failure_count += 1
            assert prediction <= label_max_abs * 10

            total_labels.append(label)
            total_predictions.append(prediction)
        except:
            failure_count += 1
    failure_rate = failure_count / len(predictions)

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
        "tens_order_failure_rate": tens_order_failure_count / len(predictions),
    }
    return evaluation_results
