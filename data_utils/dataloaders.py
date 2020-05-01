import os

import torch
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)


# ====== Helper functions required by dataloaders ===== #
def obtain_original_labels(file_name, num_explanations):
    orig_features = torch.load(file_name)
    labels = []
    num_examples = int(len(orig_features) / num_explanations)
    for i in range(num_examples):
        #TODO: assert that the whole slice has the same label
        curr_label = orig_features[i*num_explanations].label
        labels.append(curr_label)
    return labels

def obtain_distributed(data_dir, bin_files):
    # assume distributed into 10 separate files
    all_tensors = []
    for bin_file in bin_files:
        curr_tensor = torch.load(os.path.join(data_dir, bin_file))
        all_tensors.append(torch.tensor(curr_tensor, dtype=torch.float))
    return torch.cat(all_tensors, 0)

def get_tacred_labels(args, file_name):
    relations = open(os.path.join(args.data_dir, 'labels.txt')).readlines()
    relations = [relation.strip() for relation in relations]
    relation_to_idx = {relation : idx for (idx, relation) in enumerate(relations)}
    labels = []
    with open(os.path.join(args.data_dir, file_name)) as f:
        data = json.load(f)
        labels = [relation_to_idx[dat['relation']] for dat in data]
    return labels

def create_dataset(args, file_name, bin_file, is_train):
    labels = torch.load(f'{args.data_dir}/{file_name}_labels.bin')
    if args.regex_features > 0:
        regex_file = f'{args.feat_dir}/{file_name}.bin'
        dataset = FeaturesWithPatternsDataset(args, labels, bin_file, regex_file, train=is_train)
    else:
        dataset = LogitDataset(args, labels, bin_file, train=is_train, exp_list = args.exp_list)
    return dataset

def create_dataset_old(args, file_name, file_path, bin_file, is_train):
    if args.regex_features > 0:
        regex_features = f'{args.feat_dir}/{file_name}.bin'
        dataset = FeaturesWithPatternsDataset(args, file_path, bin_file, regex_features, train=is_train)
    else:
        if args.task_name == 'tacred':
            labels = get_tacred_labels(args, file_path)
        else:
            labels = None
        dataset = LogitDataset(args, file_path, bin_file,  train=is_train, labels = labels, exp_list = args.exp_list)
    return dataset

class FeaturesWithPatternsDataset(Dataset):
    def __init__(self, args, labels, bin_file, regex_file, train=False):
        train_name = args.train_file.split(".")[0]
        self.labels = labels

        num_examples = len(self.labels)
        if args.classifier_type in ['logit_classifier', 'key_phrase_classifier']:
            logit_features = torch.load(os.path.join(args.data_dir, bin_file))
            self.features = torch.tensor(logit_features, dtype=torch.float)
        else:
            # training features can be distributed
            if type(bin_file) == list:
                pre_classifier_features = obtain_distributed(args.data_dir, bin_file)
            else:
                pre_classifier_features = torch.load(os.path.join(args.data_dir, bin_file))
            self.features = torch.tensor(pre_classifier_features, dtype=torch.float)

        regex_features = torch.load(os.path.join(args.data_dir,regex_file))
        self.regex_features = torch.tensor(regex_features, dtype=torch.float)

        print('feature shape')
        print(self.features.shape)
        assert(args.regex_features == regex_features.shape[1])
        assert(len(self.labels) == len(self.features))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.regex_features[index], self.labels[index]



class LogitDataset(Dataset):
    def __init__(self, args, labels, bin_file, train=False, exp_list = []):
        train_name = args.train_file.split(".")[0]
        self.labels = labels
        num_examples = len(self.labels)
        if args.classifier_type in ['logit_classifier', 'key_phrase_classifier']:
            logit_features = torch.load(os.path.join(args.data_dir, bin_file))
            self.features = torch.tensor(logit_features, dtype=torch.float)
        else:
            # training features can be distributed
            if type(bin_file) == list:
                pre_classifier_features = obtain_distributed(args.data_dir, bin_file)
            else:
                pre_classifier_features = torch.load(os.path.join(args.data_dir, bin_file))
            pre_classifier_features = pre_classifier_features.reshape(len(self.labels), -1, args.feature_dim)
            if len(exp_list) != 0:
                if args.keep:
                    print(f'keeping {exp_list} explanations')
                    pre_classifier_features = pre_classifier_features[:, exp_list, :]
                else:
                    print(f'removing {exp_list} explanations')
                    remaining = []
                    bs, num_exp, _ = pre_classifier_features.shape
                    for i in range(num_exp):
                        if i not in exp_list:
                            remaining.append(i)
                    pre_classifier_features = pre_classifier_features[:, remaining, :]
            self.features = torch.tensor(pre_classifier_features, dtype=torch.float)

        if train == True and args.percent_train != 1.0:
            idx = int(len(self.features)*args.percent_train)
            self.features = self.features[:idx]
            self.labels = self.labels[:idx]

        print('feature shape')
        print(self.features.shape)
        if len(self.labels) != len(self.features):
            num_feats = len(self.features)
            self.labels = self.labels[: num_feats]
        assert(len(self.labels) == len(self.features))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
