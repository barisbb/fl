import itertools

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score, classification_report

import random


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def start_cuda(cuda_no):
    torch.cuda.set_device(cuda_no)
    print("Using GPU No. {}".format(torch.cuda.current_device()))

    # CUDA
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True


def get_keys(num_cls):
    """Helper Function to generate keys from sklearn report."""
    primary_keys = list(map(str, range(num_cls))) + ["micro avg", "macro avg", "weighted avg"]
    secondary_keys = ["precision", "recall", "f1-score", "support"]
    return primary_keys, secondary_keys


def init_results(num_cls):
    """
    Generate empty template for tracking results coming from the sklearn
    classification report metric.
    """
    primary_keys, secondary_keys = get_keys(num_cls)

    def init_result_dict():
        return dict(zip(secondary_keys, [[] for i in range(len(secondary_keys))]))

    result_dict_list = [init_result_dict() for i in range(len(primary_keys))]
    scores = dict(zip(primary_keys, result_dict_list))
    # hard code other metrics
    scores["ap_mic"] = []
    scores["ap_mac"] = []
    return scores


def update_results(scores, eval_result, num_cls):
    primary_keys, secondary_keys = get_keys(num_cls)
    for prim_key, sec_key in list(itertools.product(*[primary_keys, secondary_keys])):
        scores[prim_key][sec_key].append(eval_result[prim_key][sec_key])
    # hard code other scores
    scores["ap_mic"].append(eval_result["ap_mic"])
    scores["ap_mac"].append(eval_result["ap_mac"])
    return scores


def get_empty_report(y_labels):
    """Helper Function to generate an empty sklearn report."""
    report = {}
    primary_keys, secondary_keys = get_keys(y_labels.shape[1])
    for pkey in primary_keys:
        report[pkey] = {}
        for skey in secondary_keys:
            report[pkey][skey] = 0.0
    return report


def get_classification_report(
    y_true, y_predicted, predicted_probs, dataset_subset=None
):
    """
    Subset Serbia does not contain classes with ID 3 & 7. In this case
    exclude them from evaluation. Can be done analogous with any filter
    applied to BigEarthNet.
    """
    try:
        report = classification_report(y_true, y_predicted, output_dict=True)
    except:
        print("Bug fix for empty classification report.")
        report = get_empty_report(y_true)
    if dataset_subset == "ireland":
        ireland_indices = np.delete(np.arange(0, 19), [3, 7])
        y_true = y_true[:, ireland_indices]
        predicted_probs = predicted_probs[:, ireland_indices]
    ap_mic = average_precision_score(y_true, predicted_probs, average="micro")
    ap_mac = average_precision_score(y_true, predicted_probs, average="macro")
    report.update({"ap_mic": ap_mic, "ap_mac": ap_mac})
    return report


def print_micro_macro(report):
    micros, macros = report["micro avg"], report["macro avg"]
    micro_str = " | ".join(
        list(map(lambda x: "{}: {:.4f}".format(x[0], x[1]), list(micros.items())))
    )
    macro_str = " | ".join(
        list(map(lambda x: "{}: {:.4f}".format(x[0], x[1]), list(macros.items())))
    )
    print(
        "micro    ", micro_str[:-5] + " | mAP: {:.4f}".format(report["ap_mic"])
    )  # cheap hack to avoid float for int value support
    print("macro    ", macro_str[:-5] + " | mAP: {:.4f}".format(report["ap_mac"]))
    print()


class MetricTracker(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
