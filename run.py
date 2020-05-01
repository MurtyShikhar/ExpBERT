import argparse
import json
import itertools
import yaml
import logging, os

import sys
sys.path.append('data_utils')

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from sklearn.metrics import f1_score


from classifiers import *
from data_utils.dataloaders import FeaturesWithPatternsDataset, create_dataset
from utils import set_seed, compute_confusion_matrix, obtain_bin_files, evaluate_tacred



logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

# for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def sweep_hyperparams(args, train_dataset, dev_dataset):
    hyperparams = yaml.load(open(args.hyperparam_config), Loader=yaml.FullLoader)
    hyperparam_names = [name for name in hyperparams]
    all_values = [hyperparams[name] for name in hyperparam_names]
    all_assignments = list(itertools.product(*all_values))

    scores = []
    for assignment in all_assignments:
        name_assn_tuple = list(zip(hyperparam_names, assignment))
        for name, val in name_assn_tuple:
            setattr(args, name, val)
        model = ExplanationFeatureConcatenatorClassifier(args.num_classes, args.num_explanations, args.feature_dim,\
                args.projection_dim, args.hidden_dim, args.num_layers, args.dropout, args.regex_features)
        model.to(args.device)
        logger.info('Running with the following hyperparameters:')
        logger.info(name_assn_tuple)
        best_dev_curr = train(args, model, train_dataset, dev_dataset, verbose=False)
        logger.info(f"Best Dev Score with these hyperparameters = {best_dev_curr}")
        scores.append(best_dev_curr)

    idx, best_dev = max(enumerate(scores), key = lambda idx, score : score)
    return best_dev, list(zip(hyperparam_names, all_assignments[idx]))


def eval(args, model, eval_dataset, verbose=True):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=64)
    # Eval!
    if verbose:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable= not verbose):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            outputs = model(batch[:-1])
            logits = outputs
        nb_eval_steps += 1
        gold_labels = batch[-1]
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = gold_labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, gold_labels.detach().cpu().numpy(), axis=0)
    pred_labels = preds.argmax(axis=1)
    accuracy = (pred_labels == out_label_ids).mean()
    if args.num_classes == 2:
        result = {'f1' : f1_score(out_label_ids, pred_labels, pos_label=0), 'acc' : accuracy}
        result =  compute_confusion_matrix(pred_labels, out_label_ids, result)
    elif args.task_name == 'tacred':
        prec_micro, recall_micro, f1_micro = evaluate_tacred(out_label_ids, pred_labels)
        result = {'f1' : f1_micro, 'acc' : accuracy, 'prec' : prec_micro, 'recall' : recall_micro}
    else:
        result = {'f1' : f1_score(out_label_ids, pred_labels, average='micro'), 'acc' : accuracy}

    if verbose:
        logger.info("***** Eval results {} *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return result


def train(args, model, train_dataset, dev_dataset, verbose=True):
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    criterion = nn.CrossEntropyLoss()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) * args.num_train_epochs
    optimizer = optim.Adam(model.parameters(), weight_decay = args.weight_decay, lr=args.learning_rate)

    # evaluate twice every epoch
    args.logging_steps = int(len(train_dataloader) / 2)

    if verbose:
        logger.info(f'setting logging_steps to {args.logging_steps}')
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size = %d", args.train_batch_size)
        logger.info("  Total train batch size = %d", args.train_batch_size)
        logger.info("  Total optimization steps = %d", t_total)

    patience = 100

    global_step = 0
    best_dev_metric = 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=not verbose)

    output_dir = args.output_dir
    if args.save_model and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    curr_patience = 0
    for _ in train_iterator:
        if curr_patience > patience:
            train_iterator.close(); break
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable= not verbose)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            model.train()
            outputs = model.forward(batch[:-1])
            gold_labels = batch[-1] #(batch_size, )

            loss = criterion(outputs, gold_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            optimizer.zero_grad()
            global_step += 1
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                results = eval(args, model, dev_dataset, verbose)
                if results['f1'] > best_dev_metric:
                    best_dev_metric = results['f1']
                    if args.save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        if verbose:
                            logger.info("Saving model checkpoint to %s", output_dir)
                    curr_patience = 0
                else:
                    curr_patience += 1
                    if curr_patience > patience:
                        epoch_iterator.close(); break;
    return best_dev_metric


def get_args():
    parser = argparse.ArgumentParser(description='Entry script to run all models')
    # Whether we are in train / eval mode
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    # === Random seed used for initialization ===
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # === pointers to various directories === #
    parser.add_argument('--data_dir', type=str, default='data', help='path where data is stored')
    parser.add_argument('--output_dir', type=str, help='path where model is stored')
    parser.add_argument('--exp_dir', type=str, help='building a classifier on top of explanation features located here', default='orig_exp')
    parser.add_argument('--feat_dir', type=str, help='building a classifier on top of single dimensional regex/babble labble features located here', default='')

    # === other dataset specific arguments === #
    parser.add_argument('--percent_train', type=float, default=1.0, help='percentage of the training data used')
    parser.add_argument('--train_distributed', type=int, default=0, help='if this is >0, BERT train features are stored in these many files')
    parser.add_argument('--dev_distributed', type=int, default=0, help='if this is >0, BERT dev features are stored in these many files')
    parser.add_argument('--num_classes', type=int, default=2,help='number of classes in the dataset')

    # === Files within directories for train/dev/test sets ===#
    parser.add_argument('--train_file', type=str, default='train.txt', help='Train File')
    parser.add_argument('--dev_file', type=str,default='dev.txt', help='Dev File')

    # === Model hyperparameters === #
    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--projection_dim', type=int, default=768)

    # === Optimization specific hyperparameters
    parser.add_argument('--hyperparam_config', type=str, help='can use this to tune hyperparameters if required', default='')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument("--max_grad_norm", default=10.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="L2 penalty on training")

    # === Which model to use for training === #
    parser.add_argument('--classifier_type', default='logit_classifier',
        choices=['logit_classifier', 'key_phrase_classifier', 'feature_average', 'feature_concat', 'feature_cnn'])


    parser.add_argument('--task_name', type=str, default='spouse')
    parser.add_argument('--save_model', action='store_true')

    # === These arguments are used for the experiment ablation experiments from the paper
    parser.add_argument('--exp_list', nargs='*', type=int, default=[])
    parser.add_argument('--keep', action='store_true')

    args = parser.parse_args()
    # Use GPU if available
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.n_gpu = 1
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0
    return args


# ExpBERT/Baseline : reads in features from exp_dir
# Regex / Semparse / ExpBERT + Logits: reads in features from exp_dir + feat_dir
# labels are stored in exp_dir
def main():
    args = get_args()
    # === Set random seed
    set_seed(args)
    # === Set the number of explanations based on specified config #
    if os.path.exists(os.path.join(args.data_dir, args.exp_dir, 'dev.bin')):
        exp_features = torch.load(os.path.join(args.data_dir, args.exp_dir, 'dev.bin'))
    else:
        # the dev features might be distributed among several files
        exp_features = torch.load(os.path.join(args.data_dir, args.exp_dir, 'dev_0.bin'))

    args.num_explanations = exp_features.shape[1]
    # adjust based on which explanations we use
    if len(args.exp_list) != 0:
        if args.keep:
            args.num_explanations = len(args.exp_list)
        else:
            args.num_explanations -= len(args.exp_list)

    # === set the number of additional features here. These correspond to Regex/ SemParse / Logit features
    if args.feat_dir != '':
        regex_features = torch.load(os.path.join(args.data_dir, args.feat_dir, 'dev.bin'))
        args.regex_features = regex_features.shape[1]
    else:
        args.regex_features = 0


    # === Create dataset variables here
    train_bin_file, dev_bin_file = obtain_bin_files(args)
    args.train_bin_file = train_bin_file
    args.dev_bin_file = dev_bin_file
    logger.info(f'Binary Files: {train_bin_file}, {dev_bin_file}')

    train_name = args.train_file.split(".")[0]
    dev_name = args.dev_file.split(".")[0]

    # === Create the model

    if args.hyperparam_config != '':
        train_dataset = create_dataset(args, train_name, args.train_bin_file, True)
        dev_dataset = create_dataset(args, dev_name, args.dev_bin_file, False)
        best_dev_metric, best_hyperparams = sweep_hyperparams(args, train_dataset, dev_dataset)
        logger.info("Best dev score: {}".format(best_dev_metric))
        logger.info("Best hyperparameters: ")
        logger.info(best_hyperparams)
    else:
        model = ExplanationFeatureConcatenatorClassifier(args.num_classes, args.num_explanations,
            args.feature_dim, args.projection_dim, args.hidden_dim, args.num_layers, args.dropout, args.regex_features)
        logger.info(model)
        # === Train the model / Evaluate a saved model!
        if args.train:
            train_dataset = create_dataset(args, train_name, args.train_bin_file, True)
            dev_dataset = create_dataset(args, dev_name, args.dev_bin_file, False)
            model.to(args.device)
            best_dev_metric = train(args, model, train_dataset, dev_dataset)
            logger.info("Best dev score: {}".format(best_dev_metric))
        if args.eval:
            dev_dataset = create_dataset(args, dev_name, args.dev_bin_file, False)
            # load model
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'weights.bin')))
            model.to(args.device)
            eval(args, model, dev_dataset)

if __name__ == '__main__':
    main()






