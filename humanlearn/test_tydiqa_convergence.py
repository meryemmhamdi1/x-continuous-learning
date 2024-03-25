from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
import importlib
import os
from sys import platform
import sys
import configparser
import torch

import torch.optim as optim
from tqdm import tqdm
from torch import LongTensor
import numpy as np
from transformers import BertForQuestionAnswering
import collections

if platform == "darwin":
    sys.path.append(
        "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Code/x-continuous-learning_new"
    )
    location = "LOCAL"
else:
    sys.path.append("/home1/mmhamdi/x-continuous-learning_new")
    location = "CARC"

from src.transformers_config import MODELS_dict
from src.data_utils import MultiPurposeDataset, AugmentedList
from src.utils import (
    transfer_batch_cuda,
    get_opt_scheduler,
    SquadResult,
    simple_f1,
    simple_em,
    compute_predictions_logits,
    squad_evaluate,
)

paths = configparser.ConfigParser()
paths.read("scripts/paths.ini")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Corpus with example sentences
class DataArgs:
    def __init__(self):
        self.data_root = (
            # "/Users/meryemmhamdi/USCGDrive/Fall23/SpacedRepetition/Datasets/"
        )
        self.task_name = "qa"
        self.data_name = "tydiqa"
        self.data_format = "json"
        self.data_root = os.path.join(
            str(paths.get(location, "DATA_ROOT")),
            self.task_name.upper(),
            self.data_name.upper(),
        )
        self.model_root = os.path.join(str(paths.get(location, "MODEL_ROOT")))
        self.setup_opt = "cll-er_kd"
        self.order_lst = "ru_id_te_sw"
        self.languages = ["ru", "id", "te", "sw"]
        self.order_class = 0
        self.order_lang = 0
        self.trans_model = "BertBaseMultilingualCased"
        self.seed = 42
        self.num_class_tasks = 10
        self.num_lang_tasks = 2
        self.multi_head_out = False
        self.max_mem_sz = 100
        self.er_strategy = "random"
        self.order_class = 0
        self.num_labels = 2
        self.batch_size = 4
        self.warmup_percent = 5
        self.adam_lr = 3e-05
        self.adam_eps = 1e-08
        self.grad_acc = 4
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.epsilon = 1e-06
        self.step_size = 7
        self.gamma = 0.1
        self.test_steps = 1
        self.num_class_tasks = 10
        self.num_lang_tasks = 2
        self.max_seq_length = 384
        self.max_query_length = 64
        self.max_answer_length = 30
        self.doc_stride = 128
        self.n_best_size = 20
        self.num_tasks = -1
        self.eff_num_classes_task = -1
        self.use_adapters = False
        self.do_lower_case = False
        self.verbose_logging = False
        self.version_2_with_negative = False
        self.null_score_diff_threshold = False
        self.multi_head_in = False
        self.opt_sched_type = "linear"
        self.epochs = 10


args = DataArgs()

trans_model = "BertBaseMultilingualCased"
model_name, tokenizer_alias, model_trans_alias, config_alias = MODELS_dict[trans_model]

if args.model_root:
    model_load_alias = os.path.join(args.model_root, model_name)
else:
    model_load_alias = model_name


config = config_alias.from_pretrained(
    model_load_alias,
    output_hidden_states=True,
    output_attentions=True,
)

tokenizer = tokenizer_alias.from_pretrained(
    model_load_alias, do_lower_case=True, do_basic_tokenize=False
)

model_trans = model_trans_alias.from_pretrained(model_load_alias, config=config)

# --Training loss: 2.3371658  Training f1: 0.49225771129054674
# --Testing on language:  ru  F1:  0.6930253277671322
# --Testing on language:  id  F1:  0.573873673957646
# --Testing on language:  te  F1:  0.34221747259524965
# --Testing on language:  sw  F1:  0.5094171151023791
# Epoch:  1
# --Training loss: 0.9362708  Training f1: 0.77621905112997
# --Testing on language:  ru  F1:  0.6867979119444356
# --Testing on language:  id  F1:  0.5534028910682401
# --Testing on language:  te  F1:  0.2669271809326095
# --Testing on language:  sw  F1:  0.42196688265062177

# DownstreamModel = importlib.import_module("src.basemodels.transQA")
# model = DownstreamModel.TransModel(trans_model=model_trans, args=args, device=device)
model = BertForQuestionAnswering.from_pretrained(model_name, config=config)
model.to(device)


def eval_model(features, examples):
    em_computed = []
    f1_computed = []

    all_exact = {}
    all_f1 = {}

    losses = []

    model.eval()

    example_index_to_features = collections.defaultdict(list)
    for feature in features:
        example_index_to_features[feature.example_index].append(feature)

    for example_index, example in enumerate(examples):
        features_eg = example_index_to_features[example_index]

        if len(features_eg) > 0:
            losses_feat = []
            em_feat = []
            f1_feat = []
            for feature in features_eg:
                sequence = LongTensor([feature.input_ids]).cuda()
                attn_mask = LongTensor([feature.attention_mask]).cuda()
                token_type = LongTensor([feature.token_type_ids]).cuda()
                start_positions = LongTensor([feature.start_position]).cuda()
                end_positions = LongTensor([feature.end_position]).cuda()

                if len(sequence) == 0:
                    continue

                eval_batch = {
                    "input_ids": sequence,
                    "attention_mask": attn_mask,
                    "token_type_ids": token_type,
                    "start_positions": start_positions,
                    "end_positions": end_positions,
                }

                with torch.no_grad():
                    outputs = model(**eval_batch)

                losses_feat.append(outputs[0].mean().squeeze().item())

                start_logits = outputs[1]
                end_logits = outputs[2]

                answer_start_index = start_logits[0].argmax()
                answer_end_index = end_logits[0].argmax()

                predict_answer_tokens = sequence[
                    0, answer_start_index : answer_end_index + 1
                ]
                gold_answer_tokens = sequence[
                    0, feature.start_position : feature.end_position + 1
                ]

                gold_toks = list(gold_answer_tokens.cpu().data.numpy())
                pred_toks = list(predict_answer_tokens.cpu().data.numpy())

                em = simple_em(
                    gold_toks,
                    pred_toks,
                )

                f1 = simple_f1(
                    gold_toks,
                    pred_toks,
                )

                em_feat.append(em)
                f1_feat.append(f1)

            avg_em = np.mean(em_feat)
            avg_f1 = np.mean(f1_feat)

            losses.append(np.mean(losses_feat))
            em_computed.append(avg_em)
            f1_computed.append(avg_f1)

            all_exact.update({example.unique_id: avg_em})
            all_f1.update({example.unique_id: avg_f1})

    return np.mean(f1_computed), np.mean(em_computed)


def eval_model_old(features, examples):
    all_results = []
    model.eval()

    for i, feature in tqdm(enumerate(features)):
        sequence = LongTensor([feature.input_ids]).cuda()
        attn_mask = LongTensor([feature.attention_mask]).cuda()
        token_type = LongTensor([feature.token_type_ids]).cuda()
        start_positions = LongTensor([feature.start_position]).cuda()
        end_positions = LongTensor([feature.end_position]).cuda()

        if len(sequence) == 0:
            continue

        eval_batch = {
            "input_ids": sequence,
            "attention_mask": attn_mask,
            "token_type_ids": token_type,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }
        with torch.no_grad():
            outputs = model(**eval_batch)

        start_logits = outputs[1]
        end_logits = outputs[2]

        all_results.append(
            SquadResult(feature.unique_id, start_logits[0], end_logits[0])
        )

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        None,
        None,
        None,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results, all_exact, all_f1 = squad_evaluate(examples, predictions)
    return results["f1"]


def eval_model_old_old(features, examples):
    all_results = []
    unique_ids = []

    model.eval()
    f1_computed = []
    em_computed = []

    # example_index_to_features = collections.defaultdict(list)
    # for feature in features:
    #     example_index_to_features[feature.example_index].append(feature)

    # for example_index, example in enumerate(examples):
    for i, feature in enumerate(features):
        sequence = LongTensor([feature.input_ids]).cuda()
        attn_mask = LongTensor([feature.attention_mask]).cuda()
        token_type = LongTensor([feature.token_type_ids]).cuda()
        start_positions = LongTensor([feature.start_position]).cuda()
        end_positions = LongTensor([feature.end_position]).cuda()

        unique_ids.append(feature.unique_id)

        if len(sequence) == 0:
            continue

        # eval_batch = {
        #     "input_ids": sequence,
        #     "input_masks": attn_mask,
        #     "token_type_ids": token_type,
        #     "start_positions": start_positions,
        #     "end_positions": end_positions,
        #     "train_idx": 0,
        # }

        eval_batch = {
            "input_ids": sequence,
            "attention_mask": attn_mask,
            "token_type_ids": token_type,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }

        with torch.no_grad():
            outputs = model(**eval_batch)

        start_logits = outputs[1]
        end_logits = outputs[2]

        # start_logits = outputs.logits["start_positions"]
        # end_logits = outputs.logits["end_positions"]

        all_results.append(
            SquadResult(feature.unique_id, start_logits[0], end_logits[0])
        )

        answer_start_index = start_logits[0].argmax()
        answer_end_index = end_logits[0].argmax()

        predict_answer_tokens = sequence[0, answer_start_index : answer_end_index + 1]
        gold_answer_tokens = sequence[
            0, feature.start_position : feature.end_position + 1
        ]

        gold_toks = list(gold_answer_tokens.cpu().data.numpy())
        pred_toks = list(predict_answer_tokens.cpu().data.numpy())

        f1 = simple_f1(
            gold_toks,
            pred_toks,
        )
        em = simple_em(
            gold_toks,
            pred_toks,
        )

        f1_computed.append(f1)
        em_computed.append(em)

    return np.mean(f1_computed)


tokenizer = MODELS_dict[args.trans_model][1].from_pretrained(
    MODELS_dict[args.trans_model][0], do_lower_case=True, do_basic_tokenize=False
)

dataset = MultiPurposeDataset(args, tokenizer)

no_decay = ["bias", "LayerNorm.weight"]

weight_decay = 0.0
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": weight_decay,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

optimizer = optim.AdamW(
    optimizer_grouped_parameters, lr=args.adam_lr, eps=args.adam_eps
)
opt_scheduler = get_opt_scheduler(optimizer, dataset, 0, args)

iterator = dataset.train_stream[0]


print("Before training evaluation >>>> ")
for lang in args.languages:
    test_data = dataset.test_stream[lang]

    _, examples, test_features = dataset.next_batch(
        dataset=dataset,
        batch_size=test_data["size"],
        data_split=test_data["examples"],
    )
    if len(test_features) > 0:
        mean_f1, mean_em = eval_model(test_features, examples)

    print(
        "--Testing on language: ",
        lang,
        " F1: ",
        mean_f1,
        " EM: ",
        mean_em,
    )

for epoch in range(10):
    print("Epoch: ", epoch)
    losses_epoch = []
    for i in tqdm(range(iterator["size"] // 8)):
        model.train()
        batch, examples, features = dataset.next_batch(
            dataset=dataset,
            batch_size=8,
            data_split=iterator["examples"],
        )
        new_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["input_masks"],
            "token_type_ids": batch["token_type_ids"],
            "start_positions": batch["start_positions"],
            "end_positions": batch["end_positions"],
        }

        new_batch = transfer_batch_cuda(new_batch, device)

        output = model(**new_batch)
        loss = output[0].mean()
        # loss = output.loss["overall"]
        losses_epoch.append(loss.cpu().data.numpy())
        loss.backward()
        optimizer.step()
        opt_scheduler.step()
        model.zero_grad()

    # # Evaluation on training data
    # _, examples, features = dataset.next_batch(
    #     dataset=dataset,
    #     batch_size=iterator["size"],
    #     data_split=iterator["examples"],
    # )
    # if len(test_features) > 0:
    #     mean_f1, mean_em = eval_model(features, examples)

    print(
        "Training end of epoch: ",
        epoch,
        " loss: ",
        np.mean(losses_epoch),
        # " F1: ",
        # mean_f1,
        # " EM: ",
        # mean_em,
    )

    for lang in args.languages:
        test_data = dataset.test_stream[lang]
        _, test_examples, test_features = dataset.next_batch(
            dataset=dataset,
            batch_size=test_data["size"],
            data_split=test_data["examples"],
        )
        if len(test_features) > 0:
            mean_f1, mean_em = eval_model(test_features, test_examples)

        print(
            "--Testing on language: ",
            lang,
            " F1: ",
            mean_f1,
            " EM: ",
            mean_em,
        )
