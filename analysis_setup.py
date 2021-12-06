from data_utils import *
from parser_args import *
from utils import *
import argparse
from transformers_config import MODELS_dict
from consts import INTENT_TYPES, SLOT_TYPES
import random


def create_equal_distribution_splits():
    merged_intents = {}
    for split in sorted_intents_all.keys():
        for sorted_intents in sorted_intents_all[split]:
            for intent, count in sorted_intents.items():
                if intent not in merged_intents:
                    merged_intents.update({intent: []})
                merged_intents[intent].append(count)

    included_intents = []
    for intent in merged_intents:
        if len(merged_intents[intent]) == len(languages)*3:
            included_intents.append(intent)

    cleaned_merged_intents = {"train": {inc_intent: 0 for inc_intent in included_intents},
                              "dev": {inc_intent: 0 for inc_intent in included_intents},
                              "test": {inc_intent: 0 for inc_intent in included_intents}}

    for split in cleaned_merged_intents.keys():
        # Compute the minimum per split across languages
        all_counts_per_split = {inc_intent: [] for inc_intent in included_intents}
        for i in range(len(sorted_intents_all[split])):
            for inc_intent in included_intents:
                all_counts_per_split[inc_intent].append(sorted_intents_all[split][i][inc_intent])

        for inc_intent in cleaned_merged_intents[split]:
            cleaned_merged_intents[split][inc_intent] = min(all_counts_per_split[inc_intent])

    new_intents_map = [INTENT_TYPES[intent] for intent in included_intents]

    # Sampling data using the balanced counts ===>
    all_sents = {split: {lang: [] for lang in languages} for split in datastreams.keys()}
    for split, datastream in datastreams.items():
        for lang in languages:
            for intent, count in cleaned_merged_intents[split].items():
                # Sample from each intent in datastream[lang]
                random_sents = random.sample(intents_sent_all[split][lang][intent], count)
                # Adjust to new intent mapping
                new_sents = []
                for sent in random_sents:
                    tokens, sub_tokens, intent, sub_slots, id = sent
                    new_intent = new_intents_map.index(INTENT_TYPES[intent])
                    # print("intent:", intent, " INTENT_TYPES[intent]:", INTENT_TYPES[intent],
                    #       " new_intent:", new_intent, " new_intents_map[new_intent]:", new_intents_map[new_intent])
                    new_sents.append((tokens, sub_tokens, new_intent, sub_slots, id))
                all_sents[split][lang].extend(new_sents)

    with open(os.path.join("analysis_data", "cll-equal", "all.pickle"), "wb") as output_file:
        pickle.dump({"all_sents": all_sents, "new_intents_map": new_intents_map}, output_file)

    for split, datastream in datastreams.items():
        for lang in languages:
            with open(os.path.join("analysis_data", "cll-equal", split+"_"+lang+".txt"), "w") as output_file:
                for sent in all_sents[split][lang]:
                    tokens, sub_tokens, intent, sub_slots, id = sent
                    writable_sent = "\t".join([tokens, " ".join(sub_tokens), str(intent), " ".join(sub_slots), str(id)])
                    output_file.write(writable_sent+"\n")

    return cleaned_merged_intents, included_intents


def create_n_ways_splits():
    random.shuffle(sorted_intents)
    task_1 = sorted_intents[:4]
    task_2 = sorted_intents[4:12]
    task_3 = sorted_intents[12:24]
    task_4 = sorted_intents[24:40]
    task_5 = sorted_intents[24:60]
    print(lang, sorted_intents, count)

    print(task_1)
    print(task_2)
    print(task_3)
    print(task_4)
    print(task_5)


def create_k_shot_splits(lang):
    # Get merged intents across all splits for the language
    split_count_intents = {}
    for split in sorted_intents_all:
        sorted_intents = sorted_intents_all[split][languages.index(lang)]
        for intent, count in sorted_intents.items():
            if intent not in split_count_intents:
                split_count_intents.update({intent: []})
            split_count_intents[intent].append(count)


    # Get the list of intents that intersect across all splits
    included_intents = []
    for intent, count_list in split_count_intents.items():
        if len(count_list) == 3:
            included_intents.append(intent)

    new_intents_map = [INTENT_TYPES[intent] for intent in included_intents]

    # Get list of intents sents and given only the
    k_shot_sents = {split: [] for split in sorted_intents_all.keys()}
    counts_coverage_tasks = {split:  {task: {intent: 0 for intent in included_intents} for task in range(5)} for split in sorted_intents_all.keys()}

    for split in sorted_intents_all:
        start = min_intent
        end = 2 * min_intent
        intent_tasks = {intent: [] for intent in included_intents}
        for intent in included_intents:
            count = sorted_intents_all[split][languages.index(lang)][intent]  # count in the split, the lang and the intent
            print("intent:", intent, " count:", count)
            for inc in range(0, max_intent, min_intent):
                if count >= start + inc and count < end + inc:
                    random.shuffle(intents_sent_all[split][lang][intent])  # randomly shuffled sentences
                    # Sample start + inc partitioned to product * ( 2 + 4 + 6 + 8 + 10)
                    product = (start + inc) // min_intent
                    # 0 + 2 -> 2
                    # 2 + 4 -> 6
                    # 6 + 6 -> 12
                    # 12 + 8 -> 20
                    # 20 + 10 -> 30

                    res = 0
                    for n in range(0, 10, 2):  # 5 tasks
                        before = res
                        res = res + product * (n + 2)
                        picked_sents = intents_sent_all[split][lang][intent][before: res]
                        new_sents = []
                        for sent in picked_sents:
                            tokens, sub_tokens, intent, sub_slots, id = sent
                            new_intent = new_intents_map.index(INTENT_TYPES[intent])
                            # print("intent:", intent, " INTENT_TYPES[intent]:", INTENT_TYPES[intent],
                            #       " new_intent:", new_intent, " new_intents_map[new_intent]:", new_intents_map[new_intent])
                            new_sents.append((tokens, sub_tokens, new_intent, sub_slots, id))
                        intent_tasks[intent].append(new_sents)

        for task_i in range(5):  # 5 tasks
            task_data = []
            for intent in included_intents:
                task_data.extend(intent_tasks[intent][task_i])
                counts_coverage_tasks[split][task_i][intent] = len(intent_tasks[intent][task_i])

            k_shot_sents[split].append(task_data)

    print("lang:", lang, " counts_coverage_tasks:", counts_coverage_tasks)

    with open(os.path.join("analysis_data", "cll-k-shots", lang+".pickle"), "wb") as output_file:
        pickle.dump({"all_sents": k_shot_sents, "new_intents_map": new_intents_map}, output_file)

    for split in k_shot_sents:
        for task_i in range(5):
            with open(os.path.join("analysis_data", "cll-k-shots", split+"_"+lang+"_"+str(task_i)+".txt"), "w") as output_file:
                for sent in k_shot_sents[split][task_i]:
                    tokens, sub_tokens, intent, sub_slots, id = sent
                    writable_sent = "\t".join([tokens, " ".join(sub_tokens), str(intent), " ".join(sub_slots), str(id)])
                    output_file.write(writable_sent+"\n")

    return k_shot_sents, included_intents


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./main.py", description="Different options/arguments for running "
                                                              "continuous learning algorithms.")
    add_path_arguments(parser)
    add_setup_arguments(parser)
    add_dataset_arguments(parser)
    add_checkpoint_arguments(parser)
    add_base_model_arguments(parser)
    add_freezing_arguments(parser)
    add_model_expansion_arguments(parser)
    cont_learn_arguments(parser)
    add_meta_learning_setup(parser)
    args = parser.parse_args()

    args = get_config_params(args)

    model_name, tokenizer_alias, model_trans_alias, config_alias = MODELS_dict[args.trans_model]

    model_from_disk_dir = os.path.join(args.model_root, model_name)

    if os.path.isdir(model_from_disk_dir):
        model_load_alias = model_from_disk_dir
    else:
        model_load_alias = model_name

    config = config_alias.from_pretrained(model_load_alias,
                                          output_hidden_states=args.use_adapters)

    tokenizer = tokenizer_alias.from_pretrained(model_load_alias,
                                                do_lower_case=True,
                                                do_basic_tokenize=False)

    model_trans = model_trans_alias.from_pretrained(model_load_alias,
                                                    config=config)

    languages = ["en", "de", "fr", "hi", "es", "th"]


    print("args.setup_opt:", args.setup_opt)
    dataset = NLUDataset(args.data_root,
                         args.setup_opt,
                         args.setup_cillia,
                         args.multi_head_out,
                         args.use_mono,
                         tokenizer,
                         args.data_format,
                         args.use_slots,
                         args.seed,
                         languages,
                         args.order_class,
                         args.order_lang,
                         args.order_lst,
                         args.num_intent_tasks,
                         args.num_lang_tasks,
                         args.max_mem_sz,
                         intent_types=INTENT_TYPES,
                         slot_types=SLOT_TYPES)

    max_intent = 0  # finds the maximum count per intent
    min_intent = 30  # FIXES the minimum count per intent
    datastreams = {"train": dataset.train_set, "dev": dataset.dev_set, "test": dataset.test_set}
    sorted_intents_all = {split: [] for split in datastreams.keys()}
    intents_sent_all = {split: {lang: [] for lang in languages} for split in datastreams.keys()}
    for split, datastream in datastreams.items():
        for lang in languages:
            intents_count = {}
            intents_sent = {}
            for i in range(len(datastream[lang])):
                intent = datastream[lang][i][2]
                if intent not in intents_count:
                    intents_count.update({intent: 0})
                    intents_sent.update({intent: []})

                intents_count[intent] += 1
                intents_sent[intent].append(datastream[lang][i])
            intents_sent_all[split][lang] = intents_sent
            count = 0
            for intent in intents_count:
                if intents_count[intent] > max_intent:
                    max_intent = intents_count[intent]
                if intents_count[intent] >= min_intent:
                    count += 1

            # Order everything here from highest to lowest
            sorted_intents = sorted(list(filter(lambda pair: pair[1] >= min_intent, intents_count.items())),
                                    key=lambda pair: pair[1], reverse=True)

            # Sorted intents per lang appended to a common list
            sorted_intents_all[split].append(dict(sorted_intents))

    # cleaned_merged_intents, included_intents = create_equal_distribution_splits()

    for lang in languages:
        cleaned_merged_intents, included_intents = create_k_shot_splits(lang)
        # print(cleaned_merged_intents)


    # for split in datastreams.keys():
    #     for i in range(len(sorted_intents_all[split])):
    #         print("Split: ", split, " language:", languages[i], " sorted_intents_all:", sorted_intents_all[split][i],
    #               " total count:", sum([count for intent, count in sorted_intents_all[split][i].items()]))



    # print("intent:", intent, " merged_intents[intent]:", min(merged_intents[intent]))
    # merged_intents[intent] = min(merged_intents[intent])
    #
    # print("=========================================================================")
    # cleaned_merged_intents = {}
    # for intent, count in merged_intents.items():
    #     if count != 0:
    #         print(intent, count)
    #         cleaned_merged_intents.update({intent: count})
    #
    # print("cleaned_merged_intents:", cleaned_merged_intents)
    # print(merged_intents)