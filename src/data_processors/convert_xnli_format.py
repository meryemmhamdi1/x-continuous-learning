import csv, os, random

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

random.seed(42)

def read_xnli_data():
    data_path = "/project/jonmay_231/meryem/Datasets/XNLI/XNLI-1.0/"
    split_names = ["dev", "test"]
    languages = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    data = {split: {lang: [] for lang in languages} for split in split_names}
    for split in split_names:
        lines = _read_tsv(os.path.join(data_path, "xnli."+split+".tsv"))
        for i, line in enumerate(lines):
            if i == 0:
                continue

            language = line[0]
            label = line[1]
            sentence1 = line[6]
            sentence2 = line[7]

            data[split][language].append([sentence1, sentence2, label])
            print("sentence1:", sentence1)

    # Shuffle and split the dev into train and eval
    random.shuffle(data["dev"][language])
    num_train = int(len(data["dev"][language]) * 0.80)
    data.update({"train": {language: data["dev"][language][:num_train] for language in languages}})
    data.update({"eval": {language: data["dev"][language][num_train:] for language in languages}})

    save_dir = "/project/jonmay_231/meryem/Datasets/XNLI/PERLANG/"
    for language in languages:
        path_dir = os.path.join(save_dir, language)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)

        for split in ["train", "eval", "test"]:
            with open(os.path.join(path_dir, split+'.tsv'), 'w') as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerows(data[split][language])

def split_mt_data():
    data_path = "/project/jonmay_231/meryem/Datasets/MT/Europarl"
    split_names = ["train", "valid", "test"]
    languages = ["bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hu", "it", "lt", "lv", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
    for lang in languages:
        print("lang:", lang)
        # data_src = {split: [] for split in split_names}
        # data_trg = {split: [] for split in split_names}
        with open(os.path.join(data_path, "europarl-v7."+lang+"-en."+lang)) as file:
            src_sents = file.read().splitlines()

        with open(os.path.join(data_path, "europarl-v7."+lang+"-en.en")) as file:
            trg_sents = file.read().splitlines()

        indices = list(range(len(src_sents)))

        random.shuffle(indices)

        num_train = int(len(src_sents) * 0.60)
        num_dev = int(len(src_sents) * 0.20)

        indices_splits = {"train": indices[:num_train], "valid": indices[num_train:num_train+num_dev], "test": indices[num_train+num_dev:]}

        for split in split_names:
            print("split:", split)
            data_src = []
            data_trg = []
            for idx in indices_splits[split]:
                data_src.append(src_sents[idx])
                data_trg.append(trg_sents[idx])

            with open(os.path.join(data_path, split, lang+"-en."+lang), 'w') as file:
                for line in data_src:
                    file.write(f"{line}\n")

            with open(os.path.join(data_path, split, lang+"-en.en"), 'w') as file:
                for line in data_trg:
                    file.write(f"{line}\n")

    
# read_xnli_data()
split_mt_data()