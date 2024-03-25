from transformers import BertTokenizer
import os, csv
import pandas as pd
import numpy as np

def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

languages = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
data_path = "/project/jonmay_231/meryem/Datasets/XNLI/XNLI-1.0/"
def split_langs_split_train_dev():
    for split in ["dev", "test"]:
        data = {lang: {"id": [], "text_a": [], "text_b": [], "label": []} for lang in languages}
        # with open(data_path+"xnli.dev.tsv", "r") as f_reader: 
        #     reader = csv.reader(f_reader, delimiter='\t')
        # df = pd.read_csv(data_path+"xnli."+split+".tsv", delimiter="\t",  index_col=False)
        # language	gold_label	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	promptID	pairID	genre	label1	label2	label3	label4	label5	sentence1_tokenized	sentence2_tokenized	match
        lines = _read_tsv(os.path.join(data_path, "xnli."+split+".tsv"))
        for i, line in enumerate(lines):
            if i == 0:
                continue

            lang = line[0]
            text_a = line[6]
            text_b = line[7]
            label = line[1]

            if not isinstance(text_a, str):
                raise ValueError(f"Training input {text_a} is not a string")
            if not isinstance(text_b, str):
                raise ValueError(f"Training input {text_b} is not a string")
            if not isinstance(label, str):
                raise ValueError(f"Training label {label} is not a string")
            
            data[lang]["id"].append(i-1)
            data[lang]["text_a"].append(text_a)
            data[lang]["text_b"].append(text_b)
            data[lang]["label"].append(label)

        if split == "dev": # Split that into train/dev splits
            for lang in languages:
                df = pd.DataFrame()
                for key in data[lang]:
                    df[key] = data[lang][key]

                split_train_dev(df, data_path, lang)

        else:
            for lang in languages:
                df = pd.DataFrame()
                for key in data[lang]:
                    df[key] = data[lang][key]

                lang_dir = os.path.join(data_path, lang)

                if not os.path.isdir(lang_dir):
                    os.makedirs(lang_dir)

                df.to_csv(os.path.join(lang_dir, split + ".tsv"), sep="\t", index=False)

def split_train_dev(df, lang):
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= 0.7

    train = df[msk]
    dev = df[~msk]

    train.drop(columns=["split"])
    dev.drop(columns=["split"])

    lang_dir = os.path.join(data_path, lang)

    if not os.path.isdir(lang_dir):
        os.makedirs(lang_dir)

    train.to_csv(os.path.join(lang_dir,  "train.tsv"), sep="\t", index=False)
    dev.to_csv(os.path.join(lang_dir, "eval.tsv"), sep="\t", index=False)
