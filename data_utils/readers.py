# Readers for reading Spouse/CDR/TACRED data : returns a tuple containing x = (s, e_1, e_2) and y
import json


def read_babble(data_file, label_dict):
    '''
    readers for datasets from BabbleLabble - namely Spouse/CDR
    '''
    data = []
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                sent, e1, e2, doc_name, label = line.strip().split('\t')
            except:
                print(line)
                raise Exception("error reading file")
            sent = sent.strip()
            label = int(label)
            if label == 1:
                label = 'entailment'
            else:
                label = 'not_entailment'
            x = (sent, e1, e2)
            y = label_dict[label]
            data.append((x,y))
    return data

def read_tacred(data_file, label_dict):
    data = []
    with open(data_file, 'r') as f:
        json_dicts = json.load(f)
        for json_dict in json_dicts:
            relation = json_dict['relation']
            sent_tokens = json_dict['token']
            e1_st = json_dict['subj_start']
            e1_en = json_dict['subj_end']

            e2_st = json_dict['obj_start']
            e2_en = json_dict['obj_end']

            e1 = " ".join(sent_tokens[e1_st : e1_en + 1])
            e2 = " ".join(sent_tokens[e2_st : e2_en + 1])
            sent = " ".join(sent_tokens)
            x = (sent, e1, e2)
            y = label_dict[relation]
            data.append((x,y))
    return data
