import torch
from typing import Tuple
from torch.nn import functional as F


def simpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    loss_type="sigmoid",
    beta=1.0,
    gamma_beta_ratio=0.0,
    device="cuda",
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the SimPO loss for a batch of policy model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the SimPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    pi_logratios = pi_logratios.to(device)
    logits = pi_logratios - gamma_beta_ratio
    # avoid overflow
    logits = torch.clamp(logits, min=-10, max=10)

    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = torch.relu(1 - beta * logits)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge']"
        )

    chosen_rewards = beta * policy_chosen_logps.to(device).clone().detach()
    rejected_rewards = beta * policy_rejected_logps.to(device).clone().detach()

    return losses, chosen_rewards, rejected_rewards


from typing import Dict, List, Union
from torch import nn


def concatenated_forward(
    all_logits: torch.FloatTensor,
    all_labels: torch.LongTensor,
    label_pad_token_id: int = -100,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """

    all_logps = get_batch_logps(
        logits=all_logits,
        labels=all_labels,
        average_log_prob=True,
        is_encoder_decoder=False,
        label_pad_token_id=label_pad_token_id,
    )
    len_chosen = all_labels.shape[0] // 2

    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    chosen_labels = all_labels[:len_chosen]

    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels)


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = True,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "Logits (batch and sequence length dim) and labels must have the same shape."
        )

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    target_truncation_mask = torch.where(loss_mask.sum(-1) > 0, True, False)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    # just add one for target truncated instance.
    # the loss is not used for backprop, so it's fine to have a dummy loss for truncated instances.
    numerically_stable_mask = loss_mask.sum(-1) + ~target_truncation_mask * 1e-6

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / numerically_stable_mask
    else:
        return (per_token_logps * loss_mask).sum(-1)


def minimal_get_batch_loss_metrics(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    instance_loss: torch.FloatTensor,
    is_chosen_rejected_different: torch.BoolTensor,
    simpo_weight: float = 1.0,
    beta: float = 1.0,
    gamma_beta_ratio: float = 0.0,
    loss_type="sigmoid",
):
    """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
    metrics = {}
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        chosen_labels,
    ) = concatenated_forward(
        all_logits=logits, all_labels=labels, label_pad_token_id=-100
    )

    losses_simpo, chosen_rewards, rejected_rewards = simpo_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        beta=beta,
        gamma_beta_ratio=gamma_beta_ratio,
        device=logits.device,
        loss_type=loss_type,
    )

    chosen_loss_mask = chosen_labels[:, 1:].clone() != -100
    rejected_labels = labels[chosen_labels.size(0) :]
    rejected_loss_mask = rejected_labels[:, 1:].clone() != -100

    simpo_loss_mask = torch.where(
        (rejected_loss_mask.sum(-1) > 0) & (chosen_loss_mask.sum(-1) > 0), True, False
    )
    simpo_loss_mask = simpo_loss_mask & is_chosen_rejected_different

    loss_simpo = losses_simpo[simpo_loss_mask]
    loss_simpo = loss_simpo.mean()

    chosen_instance_loss = instance_loss[: chosen_labels.size(0)]
    sft_loss = (
        chosen_instance_loss * chosen_loss_mask.sum(-1)
    ).sum() / chosen_loss_mask.sum()

    if simpo_weight > 0.0:
        loss = sft_loss + simpo_weight * loss_simpo
    else:
        loss = sft_loss

    if torch.isnan(loss):
        assert not torch.isnan(loss), "loss is nan"

    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    metrics[f"rewards/chosen"] = chosen_rewards.cpu()
    metrics[f"rewards/rejected"] = rejected_rewards.cpu()
    metrics[f"rewards/accuracies"] = reward_accuracies.cpu()
    metrics[f"rewards/margins"] = (chosen_rewards - rejected_rewards).cpu()

    metrics[f"sft_loss"] = sft_loss.clone().detach().cpu()
    metrics[f"instance_loss"] = chosen_instance_loss.clone().detach().cpu()
    metrics[f"simpo_loss"] = losses_simpo.clone().detach().cpu()
    metrics[f"logps/rejected"] = policy_rejected_logps.clone().detach().cpu()
    metrics[f"logps/chosen"] = policy_chosen_logps.clone().detach().cpu()
    # TODO: activating the below line cause backprop error, but i don't understand.
    # detach is out of place so would not affect returned loss...
    # metrics[f"loss"] = loss.detach().cpu()

    return loss, metrics


def get_batch_loss_metrics(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    simpo_weight: float = 0.0,
    is_encoder_decoder: bool = False,
    beta: float = 1.0,
    gamma_beta_ratio: float = 0.0,
):
    """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
    metrics = {}
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        chosen_labels,
    ) = concatenated_forward(
        all_logits=logits, all_labels=labels, label_pad_token_id=-100
    )
    losses_simpo, chosen_rewards, rejected_rewards = simpo_loss(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        beta=beta,
        gamma_beta_ratio=gamma_beta_ratio,
        device=logits.device,
    )

    if not is_encoder_decoder:
        policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
        chosen_labels = chosen_labels[..., 1:].clone()

    shift_logits = policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1])
    shift_labels = chosen_labels.view(-1)

    # custom forward to get not reduced loss
    loss_fct_not_reduced = nn.CrossEntropyLoss(reduction="none")
    loss_not_reduced = loss_fct_not_reduced(shift_logits, shift_labels).view(
        chosen_labels.size(0), -1
    )
    # normalization exclude default ignore index -100
    instance_non_pad_tokens = torch.where(
        shift_labels != -100,
        torch.tensor(1).to(shift_labels.device),
        torch.tensor(0).to(shift_labels.device),
    ).view(chosen_labels.size(0), -1)
    # cross entropy aggregate not row-wise, but sum of all instances
    sft_loss = (
        loss_not_reduced * instance_non_pad_tokens
    ).sum() / instance_non_pad_tokens.sum()

    chosen_loss_mask = chosen_labels[:, 1:].clone() != -100
    rejected_labels = labels[chosen_labels.size(0) :]
    rejected_loss_mask = rejected_labels[:, 1:].clone() != -100

    simpo_loss_mask = torch.where(
        (rejected_loss_mask.sum(-1) > 0) & (chosen_loss_mask.sum(-1) > 0), True, False
    )

    loss_simpo = losses_simpo[simpo_loss_mask].mean()

    if torch.isnan(loss_simpo):
        assert False, "loss_simpo is nan"

    loss = sft_loss
    if simpo_weight > 0.0:
        loss += simpo_weight * losses_simpo.mean()

    instance_loss = (loss_not_reduced * instance_non_pad_tokens).sum(
        dim=-1
    ) / instance_non_pad_tokens.sum(dim=-1)

    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    metrics[f"rewards/chosen"] = chosen_rewards.cpu()
    metrics[f"rewards/rejected"] = rejected_rewards.cpu()
    metrics[f"rewards/accuracies"] = reward_accuracies.cpu()
    metrics[f"rewards/margins"] = (chosen_rewards - rejected_rewards).cpu()

    metrics[f"sft_loss"] = sft_loss.clone().detach().cpu()
    metrics[f"instance_loss"] = instance_loss.clone().detach().cpu()
    metrics[f"simpo_loss"] = losses_simpo.clone().detach().cpu()
    metrics[f"logps/rejected"] = policy_rejected_logps.clone().detach().cpu()
    metrics[f"logps/chosen"] = policy_chosen_logps.clone().detach().cpu()
    # TODO: activating the below line cause backprop error, but i don't understand.
    # detach is out of place so would not affect returned loss...
    # metrics[f"loss"] = loss.detach().cpu()

    return loss, metrics
