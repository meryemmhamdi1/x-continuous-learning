# Generic imports
import os
import random
import json
import sys
import importlib
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

# Torch imports
import torch
import torch.optim as optim
from torch import LongTensor
from torch.utils.tensorboard import SummaryWriter

# Project-related imports
sys.path.append(os.getcwd())
import src.logstats as logstats
from src.transformers_config import MODELS_dict
from src.data_utils import MultiPurposeDataset, AugmentedList
from src.utils import (
    epoch_time,
    categorical_accuracy,
    tydiqa_simple_evaluation,
    transfer_batch_cuda,
    logger,
    get_config_params,
    create_auxiliary_dirs,
    get_opt_scheduler,
)
from src.consts import SPLIT_NAMES
from parse_args import get_arguments

print("Parsing arguments ...")
args = get_arguments()
args = get_config_params(args)

print("Fixing the random seed and device...")
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Setting up the results directories...")
results_dir = os.path.join(
    args.out_dir,
    args.trans_model,
    args.data_name.upper(),
    "Ltn" if args.use_leitner else "Baseline",
)
if not args.rand_perf:
    if args.use_er_only:
        use_er_only_opt = "_TrainEROnly"
    else:
        use_er_only_opt = "_TrainERMain"

    if args.use_leitner:
        results_dir = os.path.join(
            results_dir,
            "LtnModel-" + args.ltn_model,  # LTN type either ltn or rbf
            "DemoteFirst" if args.demote_to_first_deck else "DemotePrevious",
            "LtnSampling-" + args.lt_sampling_mode,  # fifo, rand
            "FreqSample-" + str(args.sample_batch_epoch),  # every each batch or epoch
            "FreqUpdate-"
            + str(args.update_batch_epoch)
            + "-"
            + str(
                args.update_everything
            ),  # Update options each batch/epoch and everything/current scheduler
            "NumDecks-" + str(args.num_decks),  # Hyperparameter
        )

    if args.use_er:
        results_dir = os.path.join(
            results_dir,
            "ER_prop-" + str(args.er_strategy_prop) + use_er_only_opt,
            "_ERStrategy-" + args.er_strategy,
        )

    if args.use_er and args.use_leitner:
        results_dir = os.path.join(results_dir, "ERSched-" + args.er_lq_scheduler_type)

    results_dir = os.path.join(
        results_dir,
        "Mode-" + args.lt_queue_mode,  # mono, cont-mono, cont-multi etc
        args.order_lst,
    )

all_results_dir = dict()
all_results_dir = create_auxiliary_dirs(
    root_res_dir=results_dir,
    subdir_names=[
        "runs",
        "predictions",
        "lt_scheduler-decks",
        "lt_scheduler-idmovements",
        "checkpoints",
    ],
    all_results_dir=all_results_dir,
)

writer = SummaryWriter(all_results_dir["runs"])

print("Setting up the logger and saving the configuration ...")
app_log = logger(os.path.join(results_dir, args.log_file))
app_log.info("Saving logs to results_dir %s", results_dir)

stdoutOrigin = sys.stdout
sys.stdout = open(os.path.join(results_dir, args.log_file), "w")

app_log.info("Initializing stats file")
logstats.init(os.path.join(results_dir, args.stats_file))
app_log.info("Saving configuration and arguments")
config_path = os.path.join(results_dir, "config.json")
logstats.write_json(vars(args), config_path)
logstats.add_args("config", args)

app_log.info("Setup config/tokenizer/model ...")
model_name, tokenizer_alias, model_trans_alias, config_alias = MODELS_dict[
    args.trans_model
]

# the model root for the pre-trained model is already downloaded and included in a directory
if args.model_root:
    model_load_alias = os.path.join(args.model_root, model_name)
else:
    model_load_alias = model_name

app_log.info(model_load_alias)
config = config_alias.from_pretrained(
    model_load_alias,
    output_hidden_states=True,
    output_attentions=True,
)

tokenizer = tokenizer_alias.from_pretrained(
    model_load_alias, do_lower_case=True, do_basic_tokenize=False
)

model_trans = model_trans_alias.from_pretrained(model_load_alias, config=config)


def evaluate_model(examples=None, features=None, iterator=None, out_path=None):
    eval_outcomes = {}

    epoch_losses = []
    epoch_class = []
    epoch_tags = []

    sents_text = []

    classes_true = []
    classes_pred = []

    tags_true = []
    tags_pred = []

    if iterator:
        num_examples = iterator["size"]
    else:
        num_examples = len(examples)

    if args.task_name != "qa":
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(num_examples)):
                if iterator:
                    _, examples, features = dataset.next_batch(
                        dataset=dataset,
                        batch_size=1,
                        data_split=iterator["examples"],
                    )
                    example = examples[0]
                else:
                    example = examples[i]

                inputs = {
                    "input_ids": LongTensor([example.input_ids]),
                    "input_masks": LongTensor([example.input_mask]),
                    "token_type_ids": LongTensor([example.token_type_ids]),
                    "train_idx": 0,
                    "labels": LongTensor([example.label]),
                }
                if args.use_slots:
                    inputs.update({"slot_labels": LongTensor([example.slot_labels])})

                inputs = transfer_batch_cuda(inputs, device)

                output = model(**inputs)

                # Computing the loss
                loss = output.loss["overall"]
                epoch_losses.append(loss.item())

                # Computing the accuracy except for NER
                if args.task_name != "ner":
                    labels = inputs["labels"].to(torch.device("cpu"))
                    acc = categorical_accuracy(output.logits["class"], labels)
                    epoch_class.append(acc.item())
                    classes_true.append(labels.squeeze().item())
                    classes_pred.append(output.logits["class"].squeeze().max(0)[1])

                # Saving the input text
                sents_text.append(example.get_text())

                item_perf = 1
                if args.task_name != "ner":
                    item_perf = item_perf * acc

                if args.use_slots or args.task_name == "ner":
                    # Tag Golden Truth/Predictions
                    if args.task_name == "tod":
                        true_tag = example.slot_labels
                        pred_tag_label_alias = "slots_nokd"
                        labels_map = dataset.slot_types
                    else:
                        true_tag = example.label
                        pred_tag_label_alias = "tags"
                        labels_map = dataset.class_types

                    if args.task_name == "tod":
                        tag_logits = [
                            output.logits[pred_tag_label_alias][j, :length].data.numpy()
                            for j, length in enumerate([example.length])
                        ]
                    else:
                        tag_logits = [
                            output.logits[pred_tag_label_alias][j, :length]
                            .argmax(1)
                            .data.numpy()
                            for j, length in enumerate([example.length])
                        ]

                    pred_tag = list(tag_logits[0])

                    true_tag_l = [
                        labels_map[s] if 0 <= s < len(labels_map) else "X"
                        for s in true_tag
                    ]  # TODO why not do the same type of padding of X in TOD
                    pred_tag_l = [
                        labels_map[s] if 0 <= s < len(labels_map) else "X"
                        for s in pred_tag
                    ]

                    true_tag_no_x = []
                    pred_tag_no_x = []

                    for j, tag in enumerate(true_tag_l):
                        if tag != "X":
                            true_tag_no_x.append(true_tag_l[j])
                            pred_tag_no_x.append(pred_tag_l[j])

                    tags_true.append(true_tag_no_x)
                    tags_pred.append(pred_tag_no_x)

                    item_f1 = f1_score(true_tag_no_x, pred_tag_no_x, average="macro")
                    epoch_tags.append(item_f1)
                    item_perf = item_perf * item_f1

                eval_outcomes.update({example.unique_id: item_perf})
    else:  # "qa"
        if iterator:
            _, examples, features = dataset.next_batch(
                dataset=dataset,
                batch_size=iterator["size"],
                data_split=iterator["examples"],
            )

        (
            epoch_losses,
            avg_exact_perf,
            avg_f1_perf,
            all_exact,
            all_f1,
        ) = tydiqa_simple_evaluation(
            tokenizer,
            model,
            0,
            args,
            examples=examples,
            features=features,
            out_path=None,
        )

        eval_outcomes = all_f1
        epoch_class = [v for _, v in all_exact.items()]
        epoch_tags = [v for _, v in all_f1.items()]

    if out_path:
        with open(out_path, "w") as writer:
            for i in range(len(sents_text)):
                if i < 3:
                    app_log.info("Sent : %s", sents_text[i])
                    if args.task_name != "ner":
                        app_log.info(
                            " True Class: %s", dataset.class_types[classes_true[i]]
                        )
                        app_log.info(
                            " Prediction Class: %s",
                            dataset.class_types[classes_pred[i]],
                        )
                    if args.use_slots or args.task_name == "ner":
                        app_log.info(" True Tags: %s", " ".join(tags_true[i]))
                        app_log.info(" Prediction Tags: %s", " ".join(tags_pred[i]))

                text = sents_text[i]
                if args.task_name != "ner":
                    text = (
                        text
                        + "\t"
                        + dataset.class_types[classes_true[i]]
                        + "\t"
                        + dataset.class_types[classes_pred[i]]
                    )
                if args.use_slots or args.task_name == "ner":
                    text = (
                        text
                        + "\t"
                        + " ".join(tags_true[i])
                        + "\t"
                        + " ".join(tags_pred[i])
                    )
                writer.write(text + "\n")

    return epoch_losses, epoch_class, epoch_tags, eval_outcomes


def get_loss(batch):
    optimizer.zero_grad()  # clear gradients first
    torch.cuda.empty_cache()  # releases all unoccupied cached memory
    batch = transfer_batch_cuda(batch, device)
    batch["train_idx"] = 0
    if args.task_name != "qa":
        output = model(**batch)
        loss = output.loss["overall"]
        loss.backward()
        optimizer.step()
        if args.task_name != "tod":
            opt_scheduler.step()
    else:
        eff_batch_size = batch["input_ids"].shape[0]
        times_batch_size, left_batch_size = eff_batch_size // 4, eff_batch_size % 4
        print("eff_batch_size:", eff_batch_size, " left_batch_size:", left_batch_size)
        losses = []
        for eff_k in range(times_batch_size):
            print("--- eff_k*4:", eff_k * 4, " (eff_k+1)*4:", (eff_k + 1) * 4)
            eff_batch = {
                "input_ids": batch["input_ids"][eff_k * 4 : (eff_k + 1) * 4],
                "input_masks": batch["input_masks"][eff_k * 4 : (eff_k + 1) * 4],
                "token_type_ids": batch["token_type_ids"][eff_k * 4 : (eff_k + 1) * 4],
                "start_positions": batch["start_positions"][
                    eff_k * 4 : (eff_k + 1) * 4
                ],
                "end_positions": batch["end_positions"][eff_k * 4 : (eff_k + 1) * 4],
            }
            eff_batch["train_idx"] = 0
            output = model(**eff_batch)
            loss = output.loss["overall"]
            if not torch.isnan(loss):
                losses.append(loss)
                loss.backward()
                optimizer.step()
                if args.task_name != "tod":
                    opt_scheduler.step()
                optimizer.zero_grad()  # clear gradients first
                torch.cuda.empty_cache()  # releases all unoccupied cached memory

        if times_batch_size == 0:
            eff_k = 0
        else:
            eff_k += 1

        if left_batch_size > 0:
            print("--- LEFT eff_k*4:", eff_k * 4, " left_batch_size:", left_batch_size)
            left_batch = {
                "input_ids": batch["input_ids"][eff_k * 4 : left_batch_size],
                "input_masks": batch["input_masks"][eff_k * 4 : left_batch_size],
                "token_type_ids": batch["token_type_ids"][eff_k * 4 : left_batch_size],
                "start_positions": batch["start_positions"][
                    eff_k * 4 : left_batch_size
                ],
                "end_positions": batch["end_positions"][eff_k * 4 : left_batch_size],
            }
            left_batch["train_idx"] = 0
            output = model(**left_batch)
            loss = output.loss["overall"]
            if not torch.isnan(loss):
                losses.append(loss)
                loss.backward()
                optimizer.step()
                if args.task_name != "tod":
                    opt_scheduler.step()

                optimizer.zero_grad()  # clear gradients first
                torch.cuda.empty_cache()  # releases all unoccupied cached memory

        loss = torch.mean(torch.stack(losses))
    return loss


def train_model(iterator, i_task, epoch, lt_scheduler, er_lt_scheduler):
    def update_er_queues(er_lt_scheduler):
        # Update the ER Scheduler
        for task_memory_id in range(i_task):
            if args.er_strategy == "equal-lang":
                idx = task_memory_id
            else:
                idx = 0

            all_er_examples = [
                dataset.get_item_by_id(id_)
                for id_ in er_lt_scheduler[idx].all_items.keys()
            ]
            aug_er_examples = AugmentedList(
                all_er_examples, shuffle_between_epoch=False
            )

            _, er_examples, er_features = dataset.next_batch(
                dataset=dataset,
                batch_size=len(all_er_examples),
                data_split=aug_er_examples,
            )

            _, _, _, eval_output = evaluate_model(
                examples=er_examples, features=er_features
            )

            er_lt_scheduler[idx].place_items(eval_output)

    epoch_losses = []
    er_epoch_losses = []
    if args.use_leitner and args.er_lq_scheduler_type in ["er-main", "er-both"]:
        next_item_ids = lt_scheduler.next_items(epoch)
        scheduler_examples = AugmentedList(
            [dataset.get_item_by_id(id_) for id_ in next_item_ids],
            shuffle_between_epoch=False,
        )
        train_examples = scheduler_examples  #
        total_num = len(next_item_ids)  #
    else:
        train_examples = iterator["examples"]
        total_num = iterator["size"]

    num_iter = total_num // args.batch_size
    left_over_batch = total_num % args.batch_size

    if args.use_er:
        er_sample_freq = 10  # num_iter // 10 #
        if args.use_leitner and args.er_lq_scheduler_type != "er-main":
            if i_task > 0 and epoch == 0:
                if args.er_strategy != "equal-lang":
                    er_lt_scheduler[0].init_first_deck_er(
                        init_lt_scheduler=lt_scheduler
                    )
                else:
                    for i, lang in enumerate(args.languages[:i_task]):
                        er_lt_scheduler[i].init_first_deck_er(
                            init_lt_scheduler=lt_scheduler, lang=lang
                        )
        else:
            memory = iterator["memory"]  #
    for step_num in tqdm(range(num_iter + 1)):
        if step_num == num_iter:
            to_sample_nb = left_over_batch
        else:
            to_sample_nb = args.batch_size
        if to_sample_nb > 0:
            if not args.use_er_only or (
                (not args.use_er)
                or (args.use_er and i_task == 0)
                or (args.use_er and i_task > 0 and step_num % er_sample_freq != 0)
            ):
                # Pass over the main loss
                model.train()
                batch, batch_examples, batch_features = dataset.next_batch(
                    dataset=dataset, batch_size=to_sample_nb, data_split=train_examples
                )

                loss = get_loss(batch)
                epoch_losses.append(loss.item())

            if args.use_er and i_task > 0:
                if step_num % er_sample_freq == 0:
                    if not args.use_leitner or (
                        args.er_lq_scheduler_type == "er-main" and args.use_leitner
                    ):  # ER Baseline
                        model.train()
                        # Compute number of memory passes for baselines equivalence between random (with one total memory of K) and equal-lang memories (which goes separately over each language with 1/i_task * K for each language)
                        if args.er_strategy == "equal-lang":
                            num_memory_passes = 1
                        else:
                            num_memory_passes = i_task
                        for _ in range(num_memory_passes):
                            for task_memory_id, task_memory in enumerate(memory):
                                er_batch, _, _ = dataset.next_batch(
                                    dataset=dataset,
                                    batch_size=args.batch_size,
                                    data_split=task_memory,
                                )

                                er_loss = get_loss(er_batch)
                                er_epoch_losses.append(er_loss)

                    else:  # Leitner Queues ER or both
                        for task_memory_id in range(i_task):
                            model.train()
                            if args.er_strategy == "equal-lang":
                                er_lt_sched = er_lt_scheduler[task_memory_id]
                            else:
                                er_lt_sched = er_lt_scheduler[0]

                            er_ids = er_lt_sched.next_items(epoch)
                            if er_ids:
                                er_batch, _, _ = dataset.next_batch(
                                    dataset=dataset,
                                    batch_size=args.batch_size,
                                    identifiers=er_ids,
                                )
                                er_loss = get_loss(er_batch)
                                er_epoch_losses.append(er_loss)

            if (
                args.use_leitner
                and args.ltn_model == "ltn"
                and args.update_batch_epoch == "batch"
            ):
                # Update the Main Scheduler
                _, _, _, eval_output = evaluate_model(
                    examples=batch_examples, features=batch_features
                )

                lt_scheduler.place_items(eval_output)

                if (
                    args.use_er
                    and args.er_lq_scheduler_type != "er-main"
                    and i_task > 0
                ):
                    # Update the ER Scheduler
                    update_er_queues(er_lt_scheduler)

    # End of training: Update leitner queues if the policy is to update at the end of each epoch
    if (
        args.use_leitner
        and args.ltn_model == "ltn"
        and args.update_batch_epoch == "epoch"
    ):
        print("Evaluation and Adjusting position of items in MAIN Leitner Queues ...")
        if total_num != 0:
            if args.update_everything == "everything":
                _, all_examples, all_features = dataset.next_batch(
                    dataset=dataset,
                    batch_size=iterator["size"],
                    data_split=iterator["examples"],
                )
            else:  # updating only the scheduled examples that the Leitner Queues has visited here
                _, all_examples, all_features = dataset.next_batch(
                    dataset=dataset, batch_size=total_num, data_split=train_examples
                )

            print("Testing ...")
            _, _, _, eval_output = evaluate_model(
                examples=all_examples, features=all_features
            )  # OVER THE WHOLE SET OF EXAMPLES

            print("Placing items ...")
            lt_scheduler.place_items(eval_output)

        if args.use_er and args.er_lq_scheduler_type != "er-main" and i_task > 0:
            print("Update the ER Leitner Queues ...")
            update_er_queues(er_lt_scheduler)

    return (
        epoch_losses,
        er_epoch_losses,
        lt_scheduler,
        er_lt_scheduler,
        total_num,
    )


##  Dataset
app_log.info("Loading the Dataset ...")
dataset = MultiPurposeDataset(args, tokenizer)

## Base Model
app_log.info("Loading Base Model ...")
DownstreamModel = importlib.import_module(
    "src.basemodels.trans" + args.task_name.upper()
)

args.num_labels = len(dataset.class_types)
args.num_slots = len(dataset.slot_types)
args.num_tasks = -1
args.eff_num_classes_task = -1

model = DownstreamModel.TransModel(trans_model=model_trans, args=args, device=device)

model.to(device)

if args.rand_perf:
    train_loss_l, train_acc_f1_l, train_tags_l, _ = evaluate_model(
        iterator=dataset.train_stream[0],
        out_path=None,
    )

    app_log.info("RANDOM Train performance >>>")
    app_log.info(
        "--- train_loss: {} train_tag: {}".format(
            np.mean(train_loss_l), np.mean(train_tags_l)
        )
    )

    # Validation performance
    valid_loss_l, valid_acc_f1_l, valid_tags_l, _ = evaluate_model(
        iterator=dataset.dev_stream[0],
        out_path=None,
    )
    app_log.info("RANDOM Valid performance >>>")
    app_log.info(
        "--- loss: {} tag: {}".format(np.mean(valid_loss_l), np.mean(valid_tags_l))
    )
    test_accs = {lang: ([], []) for lang in args.languages}
    for lang in args.languages:
        test_loss_l, test_class_l, test_tags_l, _ = evaluate_model(
            iterator=dataset.test_stream[lang],
            out_path=os.path.join(
                all_results_dir["predictions"], "Test_random_" + lang
            ),
        )
        if args.task_name != "ner":
            test_accs[lang][0].append(np.mean(np.mean(test_class_l)))
        if args.use_slots or args.task_name == "ner":
            test_accs[lang][1].append(np.mean(np.mean(test_tags_l)))

    with open(os.path.join(results_dir, "random_perf_test.json"), "w") as output_file:
        json.dump(test_accs, output_file)
    exit(0)

## Optimizer/Scheduler
app_log.info("Optimizer/Scheduler ...")
if args.task_name in ["nli", "qa", "ner"]:
    optimizer = optim.AdamW(model.parameters(), lr=args.adam_lr, eps=args.adam_eps)
else:  # especially tod
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.adam_lr,
        eps=args.adam_eps,
        betas=(args.beta_1, args.beta_2),
    )

## Defining ER LTN Scheduler if applicable
if args.use_leitner:
    ## Leitner Scheduler
    app_log.info("Defining Main Leitner Leitner Scheduler ...")
    SpacedRepetitionModel = importlib.import_module("src.schedulers." + args.ltn_model)
    lt_scheduler = SpacedRepetitionModel.LeitnerQueue(args)

    app_log.info("Defining ER Leitner Leitner Scheduler(s) ...")
    if args.use_er:
        er_lt_scheduler = []
        if args.er_strategy == "equal-lang":
            for i in range(len(args.languages) - 1):
                er_lt_scheduler.append(SpacedRepetitionModel.LeitnerQueue(args))
        else:
            er_lt_scheduler.append(
                SpacedRepetitionModel.LeitnerQueue(args)
            )  # just one leitner queue for all languages

    else:
        er_lt_scheduler = None
else:
    lt_scheduler = None
    er_lt_scheduler = None

app_log.info("Training steps in the stream: %d", len(dataset.train_stream))
metrics = []
for i_train in range(len(dataset.train_stream)):
    train_data_len = dataset.streams["train"][i_train]["size"]
    app_log.info(
        "********* Training i_train: %d Number of instances:%d", i_train, train_data_len
    )
    opt_scheduler = get_opt_scheduler(optimizer, dataset, i_train, args)

    ## Leitner Queues
    if args.use_leitner and args.lt_queue_mode in ["mono", "cont-mono"]:
        # This will be reinitialized with empty deck[0] for each new hop (language) if lt_queue_mode used is mono or cont-mono
        lt_scheduler = SpacedRepetitionModel.LeitnerQueue(args)

    # Appending (or reinitializing) all the training examples from the current hop to deck[0]
    if args.use_leitner:
        lt_scheduler.init_first_deck(
            dataset=dataset,
            train_examples=dataset.train_stream[i_train]["examples"],
            nb_examples=train_data_len,
        )

    # Using validation performance to save the best model
    best_valid_perf = 0
    i_best_test = args.epochs - 1

    ep_metrics = {}
    for split_name in SPLIT_NAMES:
        if split_name in ["train", "valid"]:
            split_langs = [i_train]
        else:
            split_langs = args.languages

        ep_metrics[split_name] = {
            "class": {lang: [] for lang in split_langs},
            "tags": {lang: [] for lang in split_langs},
            "losses": {lang: [] for lang in split_langs},
        }

    loss_batch_l = {split_name: []}
    class_l_batch_l = {split_name: []}
    tags_batch_l = {split_name: []}
    eval_outcomes_l = {split_name: []}

    for epoch in range(args.epochs):
        start_time = time.time()
        (
            train_loss_l,
            train_er_loss_l,
            lt_scheduler,
            er_lt_scheduler,
            total_num,
        ) = train_model(
            dataset.train_stream[i_train],
            i_train,
            epoch,
            lt_scheduler,
            er_lt_scheduler,
        )

        for split_name in SPLIT_NAMES:
            for lang_split in ep_metrics[split_name]["losses"]:
                app_log.info("Evaluating {} ".format(split_name))
                if split_name in ["valid", "test"]:
                    out_path = os.path.join(
                        all_results_dir["predictions"],
                        split_name
                        + "_on-"
                        + str(lang_split)
                        + "_aftertrainon-"
                        + str(i_train)
                        + "_epoch-"
                        + str(epoch),
                    )
                else:
                    out_path = None

                (
                    loss_batch_l[split_name],
                    class_l_batch_l[split_name],
                    tags_batch_l[split_name],
                    eval_outcomes_l[split_name],
                ) = evaluate_model(
                    iterator=dataset.streams[split_name][lang_split], out_path=out_path
                )

                eval_loss = np.mean(loss_batch_l[split_name])
                eval_class = np.mean(class_l_batch_l[split_name])

                writer.add_scalar(
                    split_name + "_loss_" + str(i_train), eval_loss, epoch
                )
                writer.add_scalar(
                    split_name + "_class_" + str(i_train), eval_class, epoch
                )

                ep_metrics[split_name]["losses"][lang_split].append(eval_loss)
                ep_metrics[split_name]["class"][lang_split].append(eval_class)

                if args.use_slots or args.task_name == "ner":
                    eval_tag = np.mean(tags_batch_l[split_name])
                    ep_metrics[split_name]["tags"][lang_split].append(eval_tag)

                    writer.add_scalar(
                        split_name + "_tag_" + str(i_train), eval_tag, epoch
                    )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if args.task_name != "ner":
            curr_val_perf = ep_metrics["valid"]["class"][i_train][epoch]
            if args.use_slots:
                curr_val_perf = (
                    curr_val_perf * ep_metrics["valid"]["tags"][i_train][epoch]
                )
        else:
            curr_val_perf = ep_metrics["valid"]["tags"][i_train][epoch]

        if curr_val_perf > best_valid_perf:
            # New best valid perf => save the model
            torch.save(
                args,
                os.path.join(all_results_dir["checkpoints"], "best_train_args.bin"),
            )
            torch.save(
                model.state_dict(),
                os.path.join(all_results_dir["checkpoints"], "best_model.bin"),
            )
            torch.save(
                optimizer.state_dict(),
                os.path.join(all_results_dir["checkpoints"], "best_optimizer.pt"),
            )

            # Set new best valid results
            best_valid_perf = curr_val_perf
            i_best_test = epoch

        # Epoch Summary
        app_log.info(
            "Epoch: %d | Epoch Time: %dm %ds | total_num: %d",
            epoch,
            epoch_mins,
            epoch_secs,
            total_num,
        )
        for split_name in SPLIT_NAMES:
            for lang_split in ep_metrics[split_name]["losses"]:
                app_log.info(
                    "\t SPLIT: %s | Lang: %s | Loss: %.3f | Class: %.2f | Tag: %.2f",
                    split_name.upper(),
                    lang_split,
                    ep_metrics[split_name]["losses"][lang_split][epoch],
                    ep_metrics[split_name]["class"][lang_split][epoch] * 100,
                    ep_metrics[split_name]["tags"][lang_split][epoch] * 100,
                )

        if args.use_leitner:
            if args.ltn_model == "rbf":
                lt_scheduler.place_items(
                    eval_outcomes_l["valid"].values(),
                    loss_batch_l["valid"],
                    eval_outcomes_l["train"].values(),
                    loss_batch_l["train"],
                )  # TODO ADD TAGS TOO

            app_log.info(
                "[MAIN LEITNER QUEUES] Representation and Saving decks and id movements s..."
            )
            app_log.info("---Representation: %s", lt_scheduler.rep_sched())
            app_log.info(
                "---Counts of decks: {}".format(lt_scheduler.print_count_decks())
            )
            for descriptor in ["decks", "idmovements"]:
                with open(
                    os.path.join(
                        all_results_dir["lt_scheduler-" + descriptor],
                        "lt_scheduler_"
                        + descriptor
                        + "_trainon-"
                        + str(i_train)
                        + "_epoch-"
                        + str(epoch)
                        + ".json",
                    ),
                    "w",
                ) as output_file:
                    json.dump(lt_scheduler.get_by_descriptor(descriptor), output_file)

            app_log.info(
                "[ER LEITNER QUEUES] Representation and Saving decks and idmovements for  ..."
            )
            for k, er_lt in enumerate(er_lt_scheduler):
                app_log.info(
                    "ER_LTN_SCHEDULER REPRESENTATION: %i %s",
                    k,
                    er_lt.rep_sched(),
                )
                for descriptor in ["decks", "idmovements"]:
                    with open(
                        os.path.join(
                            all_results_dir["lt_scheduler-" + descriptor],
                            "er_lt_scheduler_"
                            + str(k)
                            + descriptor
                            + "_trainon-"
                            + str(i_train)
                            + "_epoch-"
                            + str(epoch)
                            + ".json",
                        ),
                        "w",
                    ) as output_file:
                        json.dump(er_lt.get_by_descriptor(descriptor), output_file)

    metrics.append(ep_metrics)

app_log.info("Saving Metrics in {} ...".format(results_dir))
with open(os.path.join(results_dir, "metrics.json"), "w") as output_file:
    json.dump(metrics, output_file)

sys.stdout.close()
sys.stdout = stdoutOrigin
