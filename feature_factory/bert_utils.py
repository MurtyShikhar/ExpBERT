from tqdm import tqdm
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

def obtain_bert(model_path):
    if 'scibert' in model_path:
        tokenizer = BertTokenizer.from_pretrained('/u/scr/smurty/scibert_scivocab_uncased/vocab.txt', do_lower_case=True)
        config = BertConfig.from_pretrained('/u/scr/smurty/scibert_scivocab_uncased', num_labels=2, finetuning_task='mnli')
    else:
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2, finetuning_task='mnli')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    config.output_sequence = False
    model = BertForSequenceClassification.from_pretrained(os.path.join(model_path), config=config)
    return model, tokenizer

def convert_examples_to_features(examples, tokenizer, max_length, pad_token):
    features = []
    for example in tqdm(examples):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncate_first_sequence=True  # We're truncating the first sequence in priority
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
        # The label value here is just a dummy
        feature = {'input_ids' : input_ids,
                   'attention_mask' : attention_mask,
                   'token_type_ids' : token_type_ids}
        features.append(feature)
    return features


def convert_to_tensors(features):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    return dataset

def run_bert(model, tokenizer, dataset, use_logits=False, device=torch.device('cuda'), bs=1024):
    '''
    Helper function to run bert and obtain features
    '''
    eval_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=bs)
    if use_logits:
        feats = np.zeros((len(dataset), 2))
    else:
        feats = np.zeros((len(dataset), 768))
    curr_st = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]}
            outputs, features = model(**inputs)
            logits = outputs[0]
        features = features.detach().cpu().numpy()
        logits = torch.nn.functional.softmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        curr_en = min(curr_st + bs, len(feats))
        if use_logits:
            feats[curr_st : curr_en] = logits
        else:
            feats[curr_st : curr_en] = features
        curr_st += bs
    return feats

