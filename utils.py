import random
import subprocess
import numpy as np
import torch
from collections import Counter
import os

# ===== Set the random seed for various processes ==== #
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


# ===== Helper functions required for purposes of evaluating the model ==== #
def evaluate_tacred(true_labels, predicted_labels):
    NO_RELATION=1
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == NO_RELATION and predicted_label == NO_RELATION:
            pass
        elif true_label == NO_RELATION and predicted_label != NO_RELATION:
            guessed_by_relation[predicted_label] ++ 1
        elif true_label != NO_RELATION and predicted_label == NO_RELATION:
            gold_by_relation[true_label] += 1
        elif true_label != NO_RELATION and predicted_label != NO_RELATION:
            guessed_by_relation[predicted_label] += 1
            gold_by_relation[true_label] += 1
            if true_label == predicted_label:
                correct_by_relation[predicted_label] += 1
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro

def compute_confusion_matrix(preds, out_label_ids, result):
    tp,fp,tn,fn = 0,0,0,0
    for pred, label in zip(preds, out_label_ids):
        if label == pred:
            if label == 0:
                tp += 1
            else:
                tn += 1
        else:
            if label == 0:
                fn += 1
            else:
                fp += 1
    result['tp'] = tp
    result['fp'] = fp
    result['tn'] = tn
    result['fn'] = fn
    return result


# ========== Utils that read various parts of / add to the args datastructure ========== #

def obtain_bin_files(args):
    train_name = args.train_file.split(".")[0]
    dev_name = args.dev_file.split(".")[0]
    if args.train_distributed > 0:
        train_names = [f'{train_name}_{i}' for i in range(args.train_distributed)]
    else:
        train_names = [train_name]
    if args.dev_distributed > 0:
        dev_names = [f'{dev_name}_{i}' for i in range(args.dev_distributed)]
    else:
        dev_names = [dev_name]

    train_bin_file = [f'{args.exp_dir}/{train_name}.bin' for train_name in train_names]
    dev_bin_file = [f'{args.exp_dir}/{dev_name}.bin' for dev_name in dev_names]
    train_bin_file = train_bin_file if len(train_bin_file) > 1 else train_bin_file[0]
    dev_bin_file = dev_bin_file if len(dev_bin_file) > 1 else dev_bin_file[0]
    return train_bin_file, dev_bin_file
