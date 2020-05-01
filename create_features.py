# Top level script for creating features #
import sys
import shutil
sys.path.append('feature_factory')
sys.path.append('data_utils')
import argparse
import os
import yaml
import torch
from feature_factory import RegexInterpreter, BertInterpreter, obtain_bert
from data_utils import read_babble, read_tacred


# might need to have a class for this too
def read_inputs(input_file):
    pass

def read_explanations(explanation_file):
    with open(explanation_file, 'r') as reader:
        lines = reader.readlines()
        explanations = [line.strip() for line in lines]
        return explanations

# save data if it doesn't exist
def save_if(data, file_name):
    if not os.path.exists(file_name):
        torch.save(data, file_name)


def main():
    parser = argparse.ArgumentParser('Script for creating features to train classifiers')
    parser.add_argument('--interpreter')
    # contains a config.yaml file containing details of the explanations, and the interpreter that must be used to compile these explanations.
    parser.add_argument('--exp_config')

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    exp_config = yaml.load(open(args.exp_config), Loader=yaml.BaseLoader)

    interpreter_config = exp_config['interpreter']
    if interpreter_config['type'] == 'regex':
        n_grams = int(interpreter_config['ngrams'])
        interpreter = RegexInterpreter(k=n_grams)
    elif interpreter_config['type'] == 'bert':
        bert_model, tokenizer = obtain_bert(interpreter_config['path'])
        use_logits = interpreter_config['use_logits'] == 'True'
        interpreter = BertInterpreter(bert_model, tokenizer, args.device, use_logits)
    else:
        raise ValueError("this interpreter is not available. To create this interpreter, add it to interpreter.py")
    path_config = exp_config['paths']
    data_dir = path_config['data_dir']


    if exp_config['data_reader'] == 'babble_reader':
        reader = read_babble
        label_dict = {'entailment' : 0, 'not_entailment' : 1}
        train_inputs = reader(os.path.join(data_dir, 'train.txt'), label_dict)
        dev_inputs = reader(os.path.join(data_dir, 'dev.txt'), label_dict)
        test_inputs = reader(os.path.join(data_dir, 'test.txt'), label_dict)
    else:
        reader = read_tacred
        label_dict = {label.strip() : idx  for (idx, label) in
                enumerate(open(os.path.join(data_dir, 'labels.txt')).readlines())}
        train_inputs = reader(os.path.join(data_dir, 'train.json'), label_dict)
        dev_inputs = reader(os.path.join(data_dir, 'dev.json'), label_dict)
        test_inputs = reader(os.path.join(data_dir, 'test.json'), label_dict)

    save_dir = path_config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    shutil.copy2(args.exp_config, save_dir)

    # contains all the explanations required
    explanations = read_explanations(os.path.join(data_dir, path_config['exp_dir'], 'explanations.txt'))

    train_features = interpreter.batch_interpret(train_inputs, explanations)
    dev_features = interpreter.batch_interpret(dev_inputs, explanations)
    test_features = interpreter.batch_interpret(test_inputs, explanations)

    # also save the labels in data_dir as an easy to load torch serialized object
    train_labels = torch.tensor([label for (_, label) in train_inputs])
    dev_labels =  torch.tensor([label for (_, label) in dev_inputs])
    test_labels = torch.tensor([label for (_, label) in test_inputs])

    save_if(train_labels, os.path.join(data_dir, 'train_labels.bin'))
    save_if(dev_labels, os.path.join(data_dir, 'dev_labels.bin'))
    save_if(test_labels, os.path.join(data_dir, 'test_labels.bin'))

    torch.save(train_features, os.path.join(save_dir, 'train.bin'))
    torch.save(dev_features, os.path.join(save_dir, 'dev.bin'))
    torch.save(test_features, os.path.join(save_dir, 'test.bin'))

if __name__ == '__main__':
    main()
