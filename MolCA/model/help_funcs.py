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
    FLOAT_TOKENS,
)
import ast
from model.save_only_metrics import Text2Mol_translation


def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(
            gt, truncation=True, max_length=text_trunc_length, padding="max_length"
        )
        gt_tokens = list(filter(("[PAD]").__ne__, gt_tokens))
        gt_tokens = list(filter(("[CLS]").__ne__, gt_tokens))
        gt_tokens = list(filter(("[SEP]").__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(
            out, truncation=True, max_length=text_trunc_length, padding="max_length"
        )
        out_tokens = list(filter(("[PAD]").__ne__, out_tokens))
        out_tokens = list(filter(("[CLS]").__ne__, out_tokens))
        out_tokens = list(filter(("[SEP]").__ne__, out_tokens))

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


def molecule_evaluate(predictions, targets, tokenizer, text_trunc_length):
    evaluation_results = caption_evaluate(
        predictions=predictions,
        targets=targets,
        tokenizer=tokenizer,
        text_trunc_length=text_trunc_length,
    )
    # TODO: implement this
    metric = Text2Mol_translation()
    molecule_results = metric.compute(predictions=predictions, references=targets)
    evaluation_results.update(molecule_results)
    return evaluation_results



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

def get_task_specific_list(predictions, targets, tasks):
    unique_tasks = list(set(tasks))
    task_specific_predictions = {t: [] for t in unique_tasks}
    task_specific_targets = {t: [] for t in unique_tasks}
    for i, t in enumerate(tasks):
        task_specific_predictions[t].append(predictions[i])
        task_specific_targets[t].append(targets[i])
    return task_specific_predictions, task_specific_targets


def task_specifically_evaluate(
    predictions, targets, tasks, probs, tokenizer, text_trunc_length
):

    # get unique items from all_tasks
    unique_tasks = list(set(tasks))
    evaluation_results = {task: dict() for task in unique_tasks}

    task_specific_predictions, task_specific_targets = get_task_specific_list(
        predictions, targets, tasks
    )


    for t in task_specific_predictions.keys():
        task_predictions = task_specific_predictions[t]
        task_targets = task_specific_targets[t]
        if t.split('/')[0] in CLASSIFICATION_BENCHMARKS:
            results = classification_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                probs=probs,
                tokenizer=tokenizer,
                text_trunc_length=text_trunc_length,
            )
        elif t.split('/')[0] in REGRESSION_BENCHMARKS:
            results = regression_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                text_trunc_length=text_trunc_length,
            )
        elif t.split('/')[0] in TEXT2MOL_BENCHMARKS + REACTION_BENCHMARKS: # output is a molecule
            results = molecule_evaluate(
                predictions=task_predictions,
                targets=task_targets,
                tokenizer=tokenizer,
                text_trunc_length=text_trunc_length,
            )
        elif t.split('/')[0] in MOL2TEXT_BENCHMARKS:
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


def classification_evaluate(
    predictions, targets, probs, tokenizer, text_trunc_length
):

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
    roc_auc = roc_auc_score(
        y_true=total_labels_np,
        y_score=total_probs[
            :, 1
        ].numpy(),  # Use y_score here because roc_auc_score expects probability scores
    )
    # TODO task specific average of metrics

    evaluation_results = {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc,
    }
    return evaluation_results


def regression_evaluate(predictions, targets, tokenizer, text_trunc_length):

    total_labels = torch.zeros(len(predictions), dtype=torch.float32)
    total_predictions = torch.zeros(len(predictions), dtype=torch.float32)
    for i in range(len(predictions)):
        label = ast.literal_eval(
            targets[i].replace(FLOAT_TOKENS[0], "").replace(FLOAT_TOKENS[1], "").replace(tokenizer.pad_token, "")
        )
        total_labels[i] = label
        prediction = (
            predictions[i].replace(FLOAT_TOKENS[0], "").replace(FLOAT_TOKENS[1], "")
        )
        try:
            prediction = float(prediction)
        except:
            prediction = 0.0
        total_predictions[i] = prediction

    # Calculate regression metrics
    mae = torch.mean(torch.abs(total_labels - total_predictions)).item()
    rmse = torch.sqrt(torch.mean((total_labels - total_predictions) ** 2)).item()
    mse = torch.mean((total_labels - total_predictions) ** 2).item()

    evaluation_results = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }
    return evaluation_results


