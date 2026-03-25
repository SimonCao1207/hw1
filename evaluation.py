import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


# compute log mean exp, log (\sum_{i=1}^n exp(x_i) / n)
def logmeanexp(x: torch.Tensor, dim=-1):
    return torch.logsumexp(x, dim=dim) - np.log(x.shape[dim])


def accuracy(pred_logits: torch.Tensor, y: torch.Tensor):
    return (pred_logits.argmax(-1) == y).float().mean()


# compute expected calibration error
# pred_logits: num_data * num_classes tensor where softmax(pred_logits) = classification probs
def ECE(pred_logits, y, num_bins=20):

    pred_py = torch.softmax(pred_logits, -1)
    confidence = pred_py.max(-1)[0]
    bdrs = torch.linspace(0.0, 1.0, num_bins + 1).to(pred_logits.device)
    bins = torch.bucketize(confidence, bdrs[1:-1])

    ece = torch.zeros((), device=pred_logits.device)
    for b in range(num_bins):

        # fill in the blank
        #####################################################################################
        pred_y = pred_py.argmax(-1)
        accuracy = (pred_y == y).float()
        mask = bins == b
        if mask.any():
            bin_prob = mask.float().mean()
            bin_conf = confidence[mask].mean()
            bin_acc = accuracy[mask].mean()
            ece += bin_prob * (bin_acc - bin_conf).abs()
        #####################################################################################

    return ece

# compute entropy of a classification vector
# H[p] = - \sum_{j=1}^k p_j log p_j
def entropy(logits):
    probs = torch.softmax(logits, -1) + 1.0e-10
    log_probs = probs.log()
    return torch.sum(-probs * log_probs, -1)

# compute AUROC of OOD detection based on estimated uncertainties
# id_pred_logits: num_data * num_classes classification logits for in-distribution data
# ood_pred_logits: num_data * num_classes classification logits for out-of-distribution data
# do binary classification based on the entropies of the classification probabilities
def OOD_AUROC(id_pred_logits, ood_pred_logits):
    id_entropy = entropy(id_pred_logits)
    ood_entropy = entropy(ood_pred_logits)

    # fill in the blank.
    # see the documentation for roc_auc_score function 
    # (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
    #####################################################################################
    y_true = torch.cat(
        [
            torch.zeros_like(id_entropy, dtype=torch.long),
            torch.ones_like(ood_entropy, dtype=torch.long),
        ]
    ).cpu().numpy()
    y_score = torch.cat([id_entropy, ood_entropy]).cpu().numpy()
    #####################################################################################

    return roc_auc_score(y_true, y_score)


def evaluate(pred_logits, ood_pred_logits, targets):

    ens_pred_logits = logmeanexp(pred_logits, 0)
    ens_ood_pred_logits = logmeanexp(ood_pred_logits, 0)
    num_samples = pred_logits.shape[0]

    indiv_accs = []
    indiv_nlls = []
    indiv_eces = []
    indiv_aurocs = []
    for i in range(num_samples):
        indiv_accs.append(accuracy(pred_logits[i], targets).cpu().item())
        indiv_nlls.append(F.cross_entropy(pred_logits[i], targets).cpu().item())
        indiv_eces.append(ECE(pred_logits[i], targets).cpu().item())
        indiv_aurocs.append(OOD_AUROC(pred_logits[i], ood_pred_logits[i]))

    metrics = {}
    metrics["indiv"] = {}
    metrics["ens"] = {}

    metrics["indiv"]["acc"] = np.mean(indiv_accs)
    metrics["indiv"]["nll"] = np.mean(indiv_nlls)
    metrics["indiv"]["ece"] = np.mean(indiv_eces)
    metrics["indiv"]["auroc"] = np.mean(indiv_aurocs)

    metrics["ens"]["acc"] = accuracy(ens_pred_logits, targets).cpu().item()
    metrics["ens"]["nll"] = F.cross_entropy(ens_pred_logits, targets).cpu().item()
    metrics["ens"]["ece"] = ECE(ens_pred_logits, targets).cpu().item()
    metrics["ens"]["auroc"] = OOD_AUROC(ens_pred_logits, ens_ood_pred_logits)

    return metrics
