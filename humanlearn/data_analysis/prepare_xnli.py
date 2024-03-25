import os, sys
sys.path.append("/home1/mmhamdi/x-continuous-learning")
from src.transformers_config import MODELS_dict
import pandas as pd
import csv

trans_model="BertBaseMultilingualCased"
model_root="/project/jonmay_231/meryem/Models/"

model_name, tokenizer_alias, model_trans_alias, config_alias = MODELS_dict[trans_model]
model_load_alias = os.path.join(model_root, model_name)

tokenizer = tokenizer_alias.from_pretrained(model_load_alias,
                                            do_lower_case=True,
                                            do_basic_tokenize=False)

max_input_length = 256
def tokenize_bert(sentence):
    tokens = tokenizer.tokenize(sentence) 
    return tokens

def split_and_cut(sentence):
    tokens = sentence.strip().split(" ")
    tokens = tokens[:max_input_length]
    return tokens

def trim_sentence(sent):
    try:
        sent = sent.split()
        sent = sent[:128]
        return " ".join(sent)
    except:
        return sent

#Get list of 0s 
def get_sent1_token_type(sent):
    try:
        return [0]* len(sent)
    except:
        return []
#Get list of 1s
def get_sent2_token_type(sent):
    try:
        return [1]* len(sent)
    except:
        return []
#combine from lists
def combine_seq(seq):
    return " ".join(seq)
#combines from lists of int
def combine_mask(mask):
    mask = [str(m) for m in mask]
    return " ".join(mask)

# To convert back attention mask and token type ids to integer.
def convert_to_int(tok_ids):
    tok_ids = [int(x) for x in tok_ids]
    return tok_ids

cls_token_idx = tokenizer.cls_token_id
sep_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

#load dataset

data_path="/project/jonmay_231/meryem/Datasets/NLI/XNLI/PERLANG/en/"

# with open(data_path+'train.tsv', "r", encoding="utf-8-sig") as f:
#     df_train = list(csv.reader(f, delimiter="\t", quotechar=None))

# with open(data_path+'eval.tsv', "r", encoding="utf-8-sig") as f:
#     df_dev = list(csv.reader(f, delimiter="\t", quotechar=None))

# with open(data_path+'test.tsv', "r", encoding="utf-8-sig") as f:
#     df_test = list(csv.reader(f, delimiter="\t", quotechar=None))


# print("df_train:", df_train)
df_train = pd.read_csv(data_path+'train.tsv', sep="\t", header=None, index_col=False, names=['sentence1','sentence2', 'gold_label'])
df_dev = pd.read_csv(data_path+'eval.tsv', sep="\t", header=None, index_col=False, names=['sentence1','sentence2', 'gold_label'])
df_test = pd.read_csv(data_path+'test.tsv', sep="\t", header=None, index_col=False, names=['sentence1','sentence2', 'gold_label'])

print(df_train["gold_label"])

#Get neccesary columns
df_train = df_train[['sentence1','sentence2', 'gold_label']]
df_dev = df_dev[['sentence1','sentence2', 'gold_label']]
df_test = df_test[['sentence1','sentence2', 'gold_label']]

#Trim each sentence upto maximum length
df_train['sentence1'] = df_train['sentence1'].apply(trim_sentence)
df_train['sentence2'] = df_train['sentence2'].apply(trim_sentence)
df_dev['sentence1'] = df_dev['sentence1'].apply(trim_sentence)
df_dev['sentence2'] = df_dev['sentence2'].apply(trim_sentence)
df_test['sentence1'] = df_test['sentence1'].apply(trim_sentence)
df_test['sentence2'] = df_test['sentence2'].apply(trim_sentence)

#Add [CLS] and [SEP] tokens
df_train['sent1'] = '[CLS] ' + df_train['sentence1'] + ' [SEP] '
df_train['sent2'] = df_train['sentence2'] + ' [SEP]'
df_dev['sent1'] = '[CLS] ' + df_dev['sentence1'] + ' [SEP] '
df_dev['sent2'] = df_dev['sentence2'] + ' [SEP]'
df_test['sent1'] = '[CLS] ' + df_test['sentence1'] + ' [SEP] '
df_test['sent2'] = df_test['sentence2'] + ' [SEP]'

#Apply Bert Tokenizer for tokeinizing
df_train['sent1_t'] = df_train['sent1'].apply(tokenize_bert)
df_train['sent2_t'] = df_train['sent2'].apply(tokenize_bert)
df_dev['sent1_t'] = df_dev['sent1'].apply(tokenize_bert)
df_dev['sent2_t'] = df_dev['sent2'].apply(tokenize_bert)
df_test['sent1_t'] = df_test['sent1'].apply(tokenize_bert)
df_test['sent2_t'] = df_test['sent2'].apply(tokenize_bert)


#Get Topen type ids for both sentence
df_train['sent1_token_type'] = df_train['sent1_t'].apply(get_sent1_token_type)
df_train['sent2_token_type'] = df_train['sent2_t'].apply(get_sent2_token_type)
df_dev['sent1_token_type'] = df_dev['sent1_t'].apply(get_sent1_token_type)
df_dev['sent2_token_type'] = df_dev['sent2_t'].apply(get_sent2_token_type)
df_test['sent1_token_type'] = df_test['sent1_t'].apply(get_sent1_token_type)
df_test['sent2_token_type'] = df_test['sent2_t'].apply(get_sent2_token_type)

#Combine both sequences
df_train['sequence'] = df_train['sent1_t'] + df_train['sent2_t']
df_dev['sequence'] = df_dev['sent1_t'] + df_dev['sent2_t']
df_test['sequence'] = df_test['sent1_t'] + df_test['sent2_t']


#Get attention mask
df_train['attention_mask'] = df_train['sequence'].apply(get_sent2_token_type)
df_dev['attention_mask'] = df_dev['sequence'].apply(get_sent2_token_type)
df_test['attention_mask'] = df_test['sequence'].apply(get_sent2_token_type)

#Get combined token type ids for input
df_train['token_type'] = df_train['sent1_token_type'] + df_train['sent2_token_type']
df_dev['token_type'] = df_dev['sent1_token_type'] + df_dev['sent2_token_type']
df_test['token_type'] = df_test['sent1_token_type'] + df_test['sent2_token_type']

#Now make all these inputs as sequential data to be easily fed into torchtext Field.
df_train['sequence'] = df_train['sequence'].apply(combine_seq)
df_dev['sequence'] = df_dev['sequence'].apply(combine_seq)
df_test['sequence'] = df_test['sequence'].apply(combine_seq)
df_train['attention_mask'] = df_train['attention_mask'].apply(combine_mask)
df_dev['attention_mask'] = df_dev['attention_mask'].apply(combine_mask)
df_test['attention_mask'] = df_test['attention_mask'].apply(combine_mask)
df_train['token_type'] = df_train['token_type'].apply(combine_mask)
df_dev['token_type'] = df_dev['token_type'].apply(combine_mask)
df_test['token_type'] = df_test['token_type'].apply(combine_mask)
df_train = df_train[['gold_label', 'sequence', 'attention_mask', 'token_type']]
df_dev = df_dev[['gold_label', 'sequence', 'attention_mask', 'token_type']]
df_test = df_test[['gold_label', 'sequence', 'attention_mask', 'token_type']]
df_train = df_train.loc[df_train['gold_label'].isin(['entailment','contradiction','neutral'])]
df_dev = df_dev.loc[df_dev['gold_label'].isin(['entailment','contradiction','neutral'])]
df_test = df_test.loc[df_test['gold_label'].isin(['entailment','contradiction','neutral'])]

#Save prepared data as csv file
df_train.to_csv(data_path+'processed_train.csv', index=False)
df_dev.to_csv(data_path+'processed_eval.csv', index=False)
df_test.to_csv(data_path+'processed_test.csv', index=False)