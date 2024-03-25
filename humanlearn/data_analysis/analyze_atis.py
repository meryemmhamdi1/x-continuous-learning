from data_utils import _parse_multi_atis
from transformers import BertTokenizer
import os, csv
import pandas as pd
import numpy as np

root_data = "/project/jonmay_231/meryem/Datasets/NLU/ATIS/"
def normalize_atis():
    data_path = root_data+"multi_atis_all/"
    for lang in ["tr", "hi"]:
        for split in ["eval", "test", "train"]:
            ids = []
            utts = []
            slots = []
            intents = []
            id_ = 1
            # df_read = pd.read_csv(data_path + lang + "/" + split + "_old.tsv", delimiter="\t",  index_col=False)

            with open(data_path + lang + "/" + split + "_old.tsv", "r") as f_reader: 
                reader = csv.reader(f_reader, delimiter='\t')
                for row in reader:
                    if split == "train":
                        intent, noneng_utt, noneng_slots = row[4], row[5], row[6]
                    else:
                        intent, noneng_utt, noneng_slots = row[3], row[4], row[5]
                    intent = 'atis_' + intent
                    ids.append(id_)
                    utts.append(noneng_utt)
                    slots.append(noneng_slots)
                    intents.append(intent)
                    id_ += 1
                    # new_rows.append((id_, noneng_utt, noneng_slots, intent))

                
            df = pd.DataFrame()
            df["id"] = ids
            df["utterance"] = utts
            df["slot_labels"] = slots
            df["intent"] = intents

            # df["id"] = [i for i in range(len(df_read))]
            # df["utterance"] = df_read[4]
            # df["slot_labels"] = df_read[5]
            # df["intent"] = df_read[3]
            df.to_csv(data_path + lang + "/" + split + ".tsv", sep="\t", index=False)

            # with open(data_path + lang + "/" + split + ".tsv", "w", newline='') as f_writer: 
            #     writer = csv.writer(f_writer)
            #     writer.writerows(new_rows)

def split_atis_seven():
    data_path = root_data+"atis_seven_languages/data/"

    print("Creation of the dev dataset")
    splits = ["train"]
    languages = ["de", "en", "es", "fr", "ja", "pt", "zh"]
    for split in splits:
        for lang in languages:
            file_path = os.path.join(data_path , split+"_"+lang.upper()+".tsv")

            df = pd.read_csv(file_path, delimiter="\t",  index_col=False)

            df['split'] = np.random.randn(df.shape[0], 1)

            msk = np.random.rand(len(df)) <= 0.7

            train = df[msk]
            dev = df[~msk]

            print(lang, len(train), len(dev))

            language = lang
            train.to_csv(root_data+"multi_atis_all/train_"+language.upper()+".tsv", sep="\t", index=False)
            dev.to_csv(root_data+"multi_atis_all/dev_"+language.upper()+".tsv", sep="\t", index=False)

    for lang in languages:
        file_path = os.path.join(data_path , "test_"+lang.upper()+".tsv")

        df = pd.read_csv(file_path, delimiter="\t", index_col=False)
        df.to_csv(root_data+"multi_atis_all/test_"+lang.upper()+".tsv", sep="\t", index=False)


def split_multiatis_hi_tr():
    data_path = root_data+"multilingual_atis/data"

    print("Creation of the dev dataset")
    splits = ["train"]
    languages = ["Hindi", "Turkish"]
    # languages = ["hi", "tr"]
    for split in splits:
        for lang in languages:
            file_path = os.path.join(data_path , lang+"-"+split+".tsv")


            df = pd.read_csv(file_path, delimiter="\t",  index_col=False)

            df['split'] = np.random.randn(df.shape[0], 1)

            msk = np.random.rand(len(df)) <= 0.7

            train = df[msk]
            dev = df[~msk]

            print(lang, len(train), len(dev))

            with open(data_path) as tsv_file:
                reader = csv.reader(tsv_file, delimiter="\t")
                next(reader)
                for i, line in enumerate(reader):
                    print(line)

            if lang == "Hindi":
                language = "hi"
            else:
                language = "tr"
  
            train.to_csv(root_data+"multi_atis_all/train_"+language.upper()+".tsv", sep="\t", index=False)
            dev.to_csv(root_data+"multi_atis_all/dev_"+language.upper()+".tsv", sep="\t", index=False)


def data_statistics():
    data_path = root_data+"atis_seven_languages/data/"

    # test_DE.tsv  test_ES.tsv  test_JA.tsv  test_ZH.tsv   train_EN.tsv  train_FR.tsv  train_PT.tsv
    # test_EN.tsv  test_FR.tsv  test_PT.tsv  train_DE.tsv  train_ES.tsv  train_JA.tsv  train_ZH.tsv

    # data_path = "/project/jonmay_231/meryem/Datasets/multilingual_atis/data"

    # Hindi-test.tsv        Hindi-train.tsv  train.tsv         Turkish-train_638.tsv
    # Hindi-train_1600.tsv  test.tsv         Turkish-test.tsv  Turkish-train.tsv

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                            do_lower_case=True,
                                            do_basic_tokenize=False)

    splits = ["train", "test"]
    languages = ["de", "en", "es", "fr", "ja", "pt", "zh"] #["hi", "tr"]
    all_intent_sets = set()
    all_slot_sets = set()
    for split in splits:
        for lang in languages:
            file_path = os.path.join(data_path , split+"_"+lang.upper()+".tsv")
            process_egs, process_egs_dict, intent_set, slot_set = _parse_multi_atis(file_path, tokenizer, split, lang, intent_set=[], slot_set=["O", "X"])

            print("len(intent_set):", len(intent_set), " len(slot_set):", len(slot_set))
            for intent in intent_set:
                all_intent_sets.add(intent)

            for slot in slot_set:
                all_slot_sets.add(slot)


    print("++++++++++++++++++++++++++++++++++++++++++++")
    print("all_intent_sets:", all_intent_sets, len(all_intent_sets))
    print("--------------------------------------------")
    print("all_slot_sets:", all_slot_sets, " len(all_slot_sets):", len(all_slot_sets))

normalize_atis()