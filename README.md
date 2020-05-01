# ExpBERT

This repository contains code, scripts, data and checkpoints for running experiments in the following paper:
> Shikhar Murty, Pang Wei Koh, Percy Liang
>
> [ExpBERT: Representation Engineering with Natural Language Explanations]

The experiments uses datasets and precomputed features which can be downloaded here:
- [Spouse][]
- [Disease][]
- For TACRED, contact the authors directly.

## Abstract

## Dependencies

Install all dependencies using `conda`:
```
conda env create -f environment.yml
conda activate lang-supevision
pip install -e .
```

## Setup

To run our code, first download the data/features into `$DATA_DIR`. The main point of entry to the code is `run.py`. Below we provide commands to train models on the `Spouse` dataset:

### NoExp

`python run.py --data_dir $DATA_DIR/spouse --train --num_train_epochs 100 --task_name spouse --classifier_type feature_concat --exp_dir input-features --num_classes 2 --train_distributed 0 --dev_distributed 0 --save_model --output_dir $outdir`

### Semparse (ProgExp) / Semparse (LangExp) / REGEX

`python run.py --data_dir $DATA_DIR/spouse --train --num_train_epochs 100 --task_name spouse --classifier_type feature_concat --exp_dir input-features --feat_dir $feat --num_classes 2 --train_distributed 0 --dev_distributed 0 --save_model --output_dir $outdir`

where `$feat` is semparse-progexp-features, semparse-langexp-features or regex-features based on the interpreter needed.

### ExpBERT

`python run.py --data_dir $DATA_DIR/spouse --train --num_train_epochs 100 --task_name spouse --classifier_type feature_concat --exp_dir input-features --feat_dir expbert-features --num_classes 2 --train_distributed 10 --dev_distributed 0 --save_model --output_dir $outdir`

Note that `train_distributed` is set to 10 here since inside `spouse/expbert-features` there are 10 files corresponding to the training features. This sharding is done to parallelize the creation of expbert features.

## Feature Pipeline
While we provide saved features used in this work, it is also possible to use this codebase to create ExpBERT like features for other explanations on any dataset. Here, we give an example of how one can create features for a dataset located in a directory `fictional-dataset`. The explanations that need to be interpreted should be inside `fictional-dataset/explanations/explanations.txt`. 

First, start by creating a `config.yaml` file (an example can be found in `spouse/input-features`). This file contains various paths as well as the interpreter to be used. Then, run the following command to produce features:

`python create_features.py --exp_config`

