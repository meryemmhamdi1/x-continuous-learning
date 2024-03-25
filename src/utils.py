import os, pickle, logging
from tqdm import tqdm
import configparser
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import math
from sys import platform
import collections

# Torch imports
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR as SchedulerLR
from torch import LongTensor

# from utils_qa import *

from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    get_raw_scores,
    apply_no_ans_threshold,
    make_eval_dict,
    merge_eval,
    find_all_best_thresh,
)

from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import sys
import os

# sys.path.append("/home1/mmhamdi/x-continuous-learning/src")
# sys.path.append(os.getcwd())
# from src.data_processors.qa import squad_convert_examples_to_features


def multiply_two_lists(test_list1, test_list2):
    return list(map(lambda x, y: x * y, test_list1, test_list2))


def create_auxiliary_dirs(root_res_dir, subdir_names, all_results_dir):
    if not os.path.isdir(root_res_dir):
        os.makedirs(root_res_dir)

    for subdir_name in subdir_names:
        all_results_dir[subdir_name] = os.path.join(root_res_dir, subdir_name)
        if not os.path.isdir(all_results_dir[subdir_name]):
            os.makedirs(all_results_dir[subdir_name])

    return all_results_dir


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_variable_name(variable):
    globals_dict = globals()
    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable]


def set_optimizer(args, parameters):
    if args.task_name == "tod":
        optimizer = Adam(
            filter(lambda p: p.requires_grad, parameters),
            betas=(args.beta_1, args.beta_2),
            eps=args.adam_eps,
            lr=args.adam_lr,
        )

        scheduler = SchedulerLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        optimizer = Adam(
            filter(lambda p: p.requires_grad, parameters),
            eps=args.adam_eps,
            lr=args.adam_lr,
        )

        def get_scheduler(optimizer, warmup_steps):
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps
            )
            return scheduler

        total_steps = math.ceil(args.epochs * 1991 * 1.0 / args.batch_size)
        warmup_steps = int(total_steps * 0.2)
        scheduler = get_scheduler(optimizer, warmup_steps)

    return optimizer, scheduler


def get_opt_scheduler(optimizer, dataset, i_train, args):
    train_data_len = dataset.streams["train"][i_train]["size"]

    if args.opt_sched_type == "constant":
        total_steps = math.ceil(args.epochs * train_data_len * 1.0 / args.batch_size)
        warmup_steps = int(total_steps * args.warmup_percent)
        opt_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )
    else:
        # THE ONE USED NOW
        warmup_steps = args.warmup_percent * 100
        t_total = train_data_len // args.grad_acc * args.epochs
        opt_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
    return opt_scheduler


def read_saved_pickle(checkpoint_dir, task_i, obj="grads"):
    with open(
        os.path.join(checkpoint_dir, "pytorch_" + obj + "_" + str(task_i)), "rb"
    ) as file:
        read_obj = pickle.load(file)

    return read_obj


def name_in_list(list, name):
    for el in list:
        if el in name:
            return True
    return False


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


def format_store_grads(
    pp, grad_dims, cont_comp, checkpoint_dir=None, tid=-1, store=True
):
    """
    This stores parameter gradients of one task at a time.
    pp: parameters
    grads: gradients
    grad_dims: list with number of parameters per layers
    tid: task id
    """
    # store the gradients
    grads = torch.Tensor(sum(grad_dims))
    grads.fill_(0.0)
    cnt = 0
    for n, p in pp:
        if name_in_list(cont_comp, n):
            if p.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en].copy_(p.grad.data.view(-1))
            cnt += 1

    if store:
        with open(
            os.path.join(checkpoint_dir, "pytorch_grads_" + str(tid)), "wb"
        ) as file:
            pickle.dump(grads, file)

    return grads


def logger(log_file):
    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    # Setup File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Setup Stream Handler (i.e. console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.INFO)

    # Get our logger
    app_log = logging.getLogger("root")
    app_log.setLevel(logging.INFO)

    # Add both Handlers
    app_log.addHandler(file_handler)
    app_log.addHandler(stream_handler)
    return app_log


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def create_new_dir(root_path, dir_name):
    new_dir = os.path.join(root_path, dir_name)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    return new_dir


def transfer_batch_cuda(batch, device):
    if device != torch.device("cpu"):
        for name, value in batch.items():
            if name != "train_idx":
                batch[name] = value.cuda()
    return batch


def get_config_params(args):
    paths = configparser.ConfigParser()
    paths.read("scripts/paths.ini")

    if platform == "darwin":
        location = "LOCAL"
    else:
        location = "CARC"

    args.data_root = os.path.join(
        str(paths.get(location, "DATA_ROOT")),
        args.task_name.upper(),
        args.data_name.upper(),
    )
    args.trans_model = str(paths.get(location, "TRANS_MODEL"))
    args.model_root = str(paths.get(location, "MODEL_ROOT"))
    args.out_dir = str(paths.get(location, "OUT_DIR"))
    args.stats_file = str(paths.get(location, "STATS_FILE"))
    args.log_file = str(paths.get(location, "LOG_FILE"))

    params = configparser.ConfigParser()
    print(
        "scripts/" + args.task_name + "/hyperparameters/" + args.data_name + "/all.ini"
    )
    params.read(
        "scripts/" + args.task_name + "/hyperparameters/" + args.data_name + "/all.ini"
    )  # +args.param_tune_idx+'.ini' for hyperparameter tuning purposes

    args.epochs = int(params.get("HYPER", "epochs"))
    args.batch_size = int(params.get("HYPER", "batch_size"))
    args.warmup_percent = int(params.get("HYPER", "warmup_percent"))
    args.adam_lr = float(params.get("HYPER", "adam_lr"))
    args.adam_eps = float(params.get("HYPER", "adam_eps"))
    args.grad_acc = int(params.get("HYPER", "grad_acc"))
    args.beta_1 = float(params.get("HYPER", "beta_1"))
    args.beta_2 = float(params.get("HYPER", "beta_2"))
    args.epsilon = float(params.get("HYPER", "epsilon"))
    args.step_size = float(params.get("HYPER", "step_size"))
    args.gamma = float(params.get("HYPER", "gamma"))
    args.test_steps = int(params.get("HYPER", "test_steps"))
    args.num_class_tasks = int(
        params.get("HYPER", "num_class_tasks")
    )  # only in case of CILIA setup
    args.num_lang_tasks = int(
        params.get("HYPER", "num_lang_tasks")
    )  # only in case of CILIA setup

    args.opt_sched_type = str(params.get("HYPER", "opt_sched_type"))

    if args.task_name == "nli":
        args.max_seq_length = int(params.get("HYPER", "max_seq_length"))
        args.pad_token = int(params.get("HYPER", "pad_token"))
        args.pad_token_segment_id = int(params.get("HYPER", "pad_token_segment_id"))
        args.mask_padding_with_zero = bool(
            params.get("HYPER", "mask_padding_with_zero")
        )

    elif args.task_name == "qa":
        args.max_seq_length = int(params.get("HYPER", "max_seq_length"))
        args.max_query_length = int(params.get("HYPER", "max_query_length"))
        args.max_answer_length = int(params.get("HYPER", "max_answer_length"))
        args.doc_stride = int(params.get("HYPER", "doc_stride"))
        args.n_best_size = int(params.get("HYPER", "n_best_size"))

    elif args.task_name == "ner":
        args.max_seq_length = int(params.get("HYPER", "max_seq_length"))

    args.languages = args.order_lst.split("_")

    return args


class Results(object):
    def __init__(self, pool_out, hidden_states, attentions):
        self.loss = {}
        self.logits = {}
        self.pool_out = pool_out
        self.hidden_states = hidden_states
        self.attentions = attentions

    def add_loss(self, name, _loss):
        self.loss.update({name: _loss})

    def add_logits(self, name, _logits):
        self.logits.update({name: _logits})


def compute_diff(comp, comp_items, sum_layers, mean_layers, name, count):
    count_comp = 0
    for k, v1 in comp.named_parameters():
        k = name + "." + k
        count += 1
        count_bert += 1
        v2 = comp_items[k].data.numpy()
        v1 = v1.data.cpu().numpy()
        res = v1 - v2

        sum = np.sum(res)
        sum_layers[k] = sum
        sum_layers[name] += sum
        sum_layers["all"] += sum

        mean = np.mean(res)
        mean_layers[k] = mean
        mean_layers[name] += mean
        mean_layers["all"] += mean

    return sum_layers, mean_layers, count, count_comp


def compute_change(params_map):
    sum_layers, mean_layers = {}, {}
    for key in params_map.keys():
        sum_layers.update(
            {key + "." + k: 0.0 for k, _ in params_map[key][0].named_parameters()}
        )
        mean_layers.update(
            {key + "." + k: 0.0 for k, _ in params_map[key][0].named_parameters()}
        )

    for key in list(params_map.keys()) + ["all"]:
        sum_layers.update({key: 0.0})
        mean_layers.update({key: 0.0})

    count = 0
    count_comp = {key: 0 for key in params_map.keys()}
    for key in params_map.keys():
        comp_items = {
            key + "." + k: v for k, v in params_map[key][1].named_parameters()
        }
        sum_layers, mean_layers, count, count_comp[key] = compute_diff(
            params_map[key][0], comp_items, sum_layers, mean_layers, key, count
        )

    for key in sum_layers:
        print(key, sum_layers[key], mean_layers[key])

    return mean_layers, sum_layers


def set_out_dir(args):
    # out_dir -> setup_opt -> slots -> trans_model -> random_init/mono/lang_order -> class_order -> cont_learn_alg ->
    #  -> headed -> adapters -> freezing
    order_lang_dict = {0: "high2lowlang", 1: "low2highlang", 2: "randomlang"}

    order_class_dict = {0: "high2lowclass", 1: "low2highclass", 2: "randomclass"}

    new_out_dir = os.path.join(args.out_dir, args.data_name.upper())
    if args.use_leitner_queue:
        new_out_dir = os.path.join(
            new_out_dir, args.ltn_scheduler_type, args.lt_sampling_mode
        )

        if args.demote_to_first_deck:
            new_out_dir = os.path.join(new_out_dir, "DemoteFirstDeck")
        else:
            new_out_dir = os.path.join(new_out_dir, "DemotePreviousDeck")

        if args.use_cont_leitner_queue:
            new_out_dir = os.path.join(new_out_dir, "ContLeitnerQueue")
        else:
            new_out_dir = os.path.join(new_out_dir, "MultiLeitnerQueue")

        new_out_dir = os.path.join(new_out_dir, args.er_lq_scheduler_type)
    else:
        new_out_dir = os.path.join(new_out_dir, "BASELINE")

    results_dir = os.path.join(
        new_out_dir, args.setup_opt  # original output directory
    )  # setup option directory

    if args.task_name == "tod":
        results_dir = os.path.join(
            results_dir, (lambda x: "NLU" if x else "Intents_only")(args.use_slots)
        )  # slot usage

    results_dir = os.path.join(
        results_dir,
        args.trans_model,
        "lr-"
        + str(args.adam_lr)
        + "_eps-"
        + str(args.adam_eps)
        + "_beta-"
        + str(args.beta_1)
        + "-"
        + str(args.beta_2),
    )  # transformers model

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    if args.random_pred:
        results_dir = os.path.join(
            results_dir,
            "RANDOM",
            "SEED_" + str(args.seed),
            args.cil_stream_lang,
            "random_init",
        )

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        return results_dir

    if args.use_mono:
        if args.setup_opt == "cil":
            results_dir = os.path.join(
                results_dir,
                "MONO",
                "SEED_" + str(args.seed),
                args.cil_stream_lang,
                str(args.mono_index),
            )
        else:
            results_dir = os.path.join(
                results_dir, "MONO", "SEED_" + str(args.seed), args.languages[0]
            )

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        return results_dir

    if args.setup_opt not in ["multi"]:
        # the order of languages and classes and continuous learning algorithms are only specific to non multi setups
        order_lang = (
            args.order_lst
            if len(args.order_lst) > 0
            else order_lang_dict[args.order_lang]
        )

        cont_alg_option = args.cont_learn_alg
        if args.cont_learn_alg != "vanilla":
            if args.use_batches_reptile:
                print("HERE HERE HERE ")
                if args.use_batches_reptile:
                    print("_use_batches_reptile")
                else:
                    print("nothing used here!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            reptile_str = ""
            if args.use_reptile:
                if args.use_batches_reptile:
                    reptile_str += "_use-reptile-batches"
                else:
                    reptile_str += "_use-reptile"

            use_er_only = ""
            if args.use_er_only:
                use_er_only += "_use-er-only"

            cont_alg_option += (
                "_memsz-"
                + str(args.max_mem_sz)
                + "_type-"
                + str(args.storing_type)
                + "_sample-"
                + str(args.sampling_type)
                + "_k-"
                + str(args.sampling_k)
                + reptile_str
                + use_er_only
            )

        if args.cont_learn_alg == "ewc":
            if args.use_online:
                cont_alg_option += "_use-online_gamma-" + str(args.gamma_ewc)
            else:
                cont_alg_option += "_no-online"

        elif args.cont_learn_alg == "gem":
            if args.use_a_gem:
                cont_alg_option += "_use_averaged-" + str(args.a_gem_n)
            else:
                cont_alg_option += "_use_vanilla_gem-" + str(args.a_gem_n)

        if args.cont_learn_alg != "vanilla":
            cont_alg_option = os.path.join(cont_alg_option, "-".join(args.cont_comp))

        ## Multi-headed architecture
        if not args.multi_head_in and not args.multi_head_out:
            head_options = "single_head"
        else:
            if args.multi_head_in:

                def map_emb_enc_subtask(layer):
                    if "all" in layer:
                        return "all"
                    elif "embeddings" in layer:
                        return "embed"
                    elif "pooler" in layer:
                        return "pool"
                    else:
                        return "enc." + layer.split(".")[2]

                head_options = "multi_head_in"

                if args.multi_head_out:
                    head_options += "_out/"

                head_options += "-".join(
                    list(map(map_emb_enc_subtask, args.emb_enc_subtask_spec.split("_")))
                )

            else:
                if args.multi_head_out:
                    head_options = "multi_head_out"

        order_class = order_class_dict[args.order_class]

        if (
            "cil" in args.setup_opt
            or "cll-k-shots" in args.setup_opt
            or "cll-n-ways" in args.setup_opt
        ):
            order_class = os.path.join(order_class, args.cil_stream_lang)

        results_dir = os.path.join(
            results_dir,
            order_lang,  # language order
            order_class,  # class order
            cont_alg_option,  # continuous learning algorithm
            head_options,  # multi-headed option
            (lambda x: "adapters" if x else "no_adapters")(args.use_adapters),
        )

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        if args.use_adapters:
            results_dir = os.path.join(
                results_dir, args.adapter_type, args.adapter_layers
            )  # adapters option

            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

    # Freezing Options
    results_dir = os.path.join(
        results_dir,
        (lambda x: "freeze_trans" if x else "tune_all_trans")(args.freeze_trans),
        (lambda x: "freeze_linear" if x else "tune_all_linear")(args.freeze_linear),
        "SEED_" + str(args.seed),
    )

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    return results_dir


def simple_f1(gold_toks, pred_toks):
    if len(pred_toks) > 0 and len(gold_toks) > 0:
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)

        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1


def simple_em(gold_toks, pred_toks):
    return int(gold_toks == pred_toks)


def tydiqa_simple_evaluation(
    tokenizer, model, test_idx, args, examples=None, features=None, out_path=None
):
    f1_computed = []
    em_computed = []

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

            avg_loss = np.mean(losses_feat)
            avg_em = np.mean(em_feat)
            avg_f1 = np.mean(f1_feat)

            losses.append(avg_loss)
            em_computed.append(avg_em)
            f1_computed.append(avg_f1)

            all_exact.update({example.unique_id: avg_em})
            all_f1.update({example.unique_id: avg_f1})

        else:
            all_exact.update({example.unique_id: 0.0})
            all_f1.update({example.unique_id: 0.0})

    return losses, np.mean(em_computed), np.mean(f1_computed), all_exact, all_f1


def tydiqa_simple_evaluation_old(
    tokenizer, model, test_idx, args, examples=None, features=None, out_path=None
):
    all_results = []
    unique_ids = []
    losses = []

    model.eval()
    f1_computed = []
    em_computed = []
    for i, feature in tqdm(enumerate(features)):
        sequence = LongTensor([feature.input_ids]).cuda()
        attn_mask = LongTensor([feature.attention_mask]).cuda()
        token_type = LongTensor([feature.token_type_ids]).cuda()
        start_positions = LongTensor([feature.start_position]).cuda()
        end_positions = LongTensor([feature.end_position]).cuda()

        unique_ids.append(feature.unique_id)

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

        losses.append(outputs[0].mean().squeeze().item())

        start_logits = outputs[1]
        end_logits = outputs[2]

        all_results.append(
            SquadResult(feature.unique_id, start_logits[0], end_logits[0])
        )

        # answer_start_index = start_logits[0].argmax()
        # answer_end_index = end_logits[0].argmax()

        # predict_answer_tokens = sequence[0, answer_start_index : answer_end_index + 1]
        # gold_answer_tokens = sequence[
        #     0, feature.start_position : feature.end_position + 1
        # ]

        # f1 = simple_f1(
        #     list(gold_answer_tokens.cpu().data.numpy()),
        #     list(predict_answer_tokens.cpu().data.numpy()),
        # )
        # em = simple_em(
        #     list(gold_answer_tokens.cpu().data.numpy()),
        #     list(predict_answer_tokens.cpu().data.numpy()),
        # )

        # f1_computed.append(f1)
        # em_computed.append(em)

    # print(" mean F1:", np.mean(f1_computed))

    # output_prediction_file = os.path.join(out_path, "predictions_{}_{}.json".format(train_idx, test_idx))
    # output_nbest_file = os.path.join(out_path, "nbest_predictions_{}_{}.json".format(train_idx, test_idx))

    output_prediction_file = out_path
    output_nbest_file = None  # out_path + "_nbest"
    output_null_log_odds_file = None  # no version_with_negatives for now

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results, all_exact, all_f1 = squad_evaluate(examples, predictions)

    # per_item_acc = {}

    # if len(examples) > 0:
    #     id_counter = 0
    #     for _, v in all_f1.items():
    #         per_item_acc.update({unique_ids[id_counter]: v})
    #         id_counter += 1

    return losses, results["exact"], results["f1"], all_exact, all_f1


def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)

    correct = (max_preds.squeeze(1) == y).float()

    return correct.sum() / len(y)


# NLU/NLI Evaluation
def nlu_nli_evaluation(
    tokenizer,
    dataset,
    memory,
    cont_learn_alg,
    dataset_test,
    nb_examples,
    model,
    train_task,
    train_idx,
    test_task,
    test_idx,
    args,
    app_log,
    device,
    name,
    out_path=None,
    verbose=False,
    prior_mbert=None,
    prior_classes=None,
    prior_slots=None,
    prior_adapter=None,
):
    app_log.info("Evaluating on i_task: %d %s", test_idx, test_task)

    eval_outcomes = {}
    if prior_mbert or prior_classes or prior_slots or prior_adapter:
        model_dict = model.state_dict()

        if prior_mbert:
            app_log.info("Using prior_mbert")
            ### 1. wanted keys, values are in trans_model
            trans_model_dict = {"trans_model." + k: v for k, v in prior_mbert.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(trans_model_dict)

        if prior_classes:
            app_log.info("Using prior_classes")
            # TODO double check the naming with test_idx
            ### 1. wanted keys, values are in trans_model
            if "cil" in args.setup_opt:
                class_classifier_dict = {
                    "gclassifier." + str(test_idx) + "." + k: v
                    for k, v in prior_classes.items()
                }
            else:
                class_classifier_dict = {
                    "gclassifier." + k: v for k, v in prior_classes.items()
                }
            ### 2. overwrite entries in the existing state dict
            model_dict.update(class_classifier_dict)

        if prior_slots:
            app_log.info("Using prior_slots")
            ### 1. wanted keys, values are in trans_model
            slot_classifier_dict = {
                "slot_classifier." + k: v for k, v in prior_slots.items()
            }

            ### 2. overwrite entries in the existing state dict
            model_dict.update(slot_classifier_dict)

        if prior_adapter:
            adapter_norm_before_dict = {
                "adapter." + k: v for k, v in prior_adapter.items()
            }

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_norm_before_dict)

        ### 3. load the new state dict
        model.load_state_dict(model_dict)

    class_corrects = 0
    sents_text = []

    classes_true = []
    classes_pred = []
    losses = []

    if args.use_slots:
        slots_true = []
        slots_pred = []

        slots_true_all = []
        slots_pred_all = []

    for _ in tqdm(range(nb_examples)):
        batch_one, examples, _ = dataset.next_batch(1, dataset_test)

        batch_one = transfer_batch_cuda(batch_one, device)

        if train_idx > 0 and name != "dev":
            if args.cont_learn_alg == "mbpa":
                """Local adaptation of MbPA"""
                q = model.get_embeddings(
                    batch_one["input_ids"], batch_one["input_masks"]
                )[0]

                # eval_model = cont_learn_alg.forward(memory, q, train_idx, model) # Old this is up to train_idx taking into consideration all memory items in previously seen tasks
                if args.use_reptile:
                    if args.use_batches_reptile:
                        eval_model = cont_learn_alg.forward_reptile_many_batches(
                            memory, q, train_idx, model, dataset
                        )
                    else:
                        eval_model = cont_learn_alg.forward_reptile_one_batch(
                            memory, q, train_idx, model, dataset
                        )
                else:
                    eval_model = cont_learn_alg.forward(
                        memory, q, train_idx, model, dataset
                    )  # this is taking into consideration only the task we are testing from assuming we know that task.
            else:
                eval_model = model
        else:
            eval_model = model

        eval_model.eval()
        # TODO test this in particular
        # TODO do we change anything at all in the original model just to make sure?

        inputs = batch_one
        inputs["train_idx"] = test_idx

        with torch.no_grad():
            outputs = eval_model(**inputs)

        losses.append(outputs.loss["overall"].squeeze().item())
        # Intent Golden Truth/Predictions
        true_class = batch_one["labels"].squeeze().item()
        pred_class = outputs.logits["class"].squeeze().max(0)[1]

        class_corrects += int(pred_class == true_class)

        masked_text = " ".join(
            dataset.tokenizer.convert_ids_to_tokens(
                batch_one["input_ids"].squeeze().tolist()
            )
        )
        classes_true.append(true_class)
        classes_pred.append(pred_class.item())
        if args.task_name == "tod":
            sents_text.append(examples[0].utterance)
        else:  # "nli"
            sents_text.append(examples[0].text_a + " " + examples[0].text_b)

        if true_class == pred_class.item():
            class_outcome = 1
        else:
            class_outcome = 0

        if args.use_slots:
            # Slot Golden Truth/Predictions
            true_slot = batch_one["slot_labels"][0]

            slot_logits = [
                outputs.logits["slots_nokd"][j, :length].data.numpy()
                for j, length in enumerate(batch_one["lengths"])
            ]
            pred_slot = list(slot_logits[0])

            true_slot_l = [dataset.slot_types[s] for s in true_slot]
            pred_slot_l = [dataset.slot_types[s] for s in pred_slot]

            true_slot_no_x = []
            pred_slot_no_x = []

            for j, slot in enumerate(true_slot_l):
                if slot != "X":
                    true_slot_no_x.append(true_slot_l[j])
                    pred_slot_no_x.append(pred_slot_l[j])

            slots_true.append(true_slot_no_x)
            slots_pred.append(pred_slot_no_x)

            slots_true_all.extend(true_slot_no_x)
            slots_pred_all.extend(pred_slot_no_x)

            item_f1 = f1_score(true_slot_no_x, pred_slot_no_x, average="macro")
            eval_outcomes.update({examples[0].unique_id: class_outcome * item_f1})
        else:
            eval_outcomes.update({examples[0].unique_id: class_outcome})

    if out_path:
        with open(out_path, "w") as writer:
            for i in range(len(sents_text)):
                if False:  # i < 3:  # print first 3 predictions
                    print("sents_text[i]:", sents_text[i])
                    app_log.info("Sent : %s", sents_text[i])
                    app_log.info(
                        " True Class: %s", dataset.class_types[classes_true[i]]
                    )
                    app_log.info(
                        " Prediction Class: %s", dataset.class_types[classes_pred[i]]
                    )
                    if args.use_slots:
                        app_log.info(" True Slots: %s", " ".join(slots_true[i]))
                        app_log.info(" Slot Prediction: %s", " ".join(slots_pred[i]))

                text = (
                    sents_text[i]
                    + "\t"
                    + dataset.class_types[classes_true[i]]
                    + "\t"
                    + dataset.class_types[classes_pred[i]]
                )
                if args.use_slots:
                    text = (
                        text
                        + "\t"
                        + " ".join(slots_true[i])
                        + "\t"
                        + " ".join(slots_pred[i])
                    )
                writer.write(text + "\n")

    if verbose:
        app_log.info(test_idx)
        app_log.info(" -----------classes_true:")
        app_log.info(set(classes_true))
        app_log.info(" -----------classes_pred:")
        app_log.info(set(classes_pred))

    pre_metric_name = train_task + "_" + str(train_idx) + "_" + name
    post_metric_name = test_task + "_" + str(test_idx)

    metrics = {}
    per_item_acc = []
    avg_perf = 0.0
    metrics.update({pre_metric_name + "_class_acc_" + post_metric_name: 0.0})
    for metric_cat in [precision_score, recall_score, f1_score]:
        metric_str = get_variable_name(metric_cat)[0]
        metrics.update(
            {pre_metric_name + "_class_" + metric_str + "_" + post_metric_name: 0.0}
        )
        if args.use_slots:
            metrics.update(
                {pre_metric_name + "_slot_" + metric_str + "_" + post_metric_name: 0.0}
            )

    if nb_examples > 0:
        per_item_acc = [
            1 if classes_true[per_idx] == classes_pred[per_idx] else 0
            for per_idx in range(nb_examples)
        ]
        class_accuracy = float(class_corrects) / nb_examples
        metrics[pre_metric_name + "_class_acc_" + post_metric_name] = class_accuracy
        avg_perf = class_accuracy
        for metric_cat in [precision_score, recall_score, f1_score]:
            metric_str = get_variable_name(metric_cat)[0]
            val_metric = metric_cat(classes_true, classes_pred, average="macro")
            metrics[
                pre_metric_name + "_class_" + metric_str + "_" + post_metric_name
            ] = val_metric
            print("metric_str:", metric_str, val_metric)
            if args.use_slots:
                val_metric = metric_cat(slots_true_all, slots_pred_all, average="macro")
                metrics[
                    pre_metric_name + "_slot_" + metric_str + "_" + post_metric_name
                ] = val_metric
                if metric_str == "f1_score":
                    avg_perf = (avg_perf + val_metric) / 2

                print("SLOT metric_str:", metric_str, val_metric)

    return metrics, avg_perf, per_item_acc, losses, eval_outcomes


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(
        self,
        unique_id,
        start_logits,
        end_logits,
        start_top_index=None,
        end_top_index=None,
        cls_logits=None,
    ):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def squad_evaluate(
    examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0
):
    qas_id_to_has_answer = {
        example.qas_id: bool(example.answers) for example in examples
    }
    has_answer_qids = [
        qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer
    ]
    no_answer_qids = [
        qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer
    ]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1 = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(
        f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(
            exact_threshold, f1_threshold, qid_list=has_answer_qids
        )
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(
            exact_threshold, f1_threshold, qid_list=no_answer_qids
        )
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(
            evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer
        )

    return evaluation, exact, f1


# TYDIQA Evaluation
def tydiqa_evaluation(
    tokenizer,
    dataset,
    memory,
    cont_learn_alg,
    dataset_test,
    nb_examples,
    model,
    train_task,
    train_idx,
    test_task,
    test_idx,
    args,
    app_log,
    device,
    name,
    batch=None,
    out_path=None,
    verbose=False,
    prior_mbert=None,
    prior_classes=None,
    prior_slots=None,
    prior_adapter=None,
):
    if prior_mbert or prior_classes or prior_slots or prior_adapter:
        model_dict = model.state_dict()

        if prior_mbert:
            app_log.info("Using prior_mbert")
            ### 1. wanted keys, values are in trans_model
            trans_model_dict = {"trans_model." + k: v for k, v in prior_mbert.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(trans_model_dict)

        if prior_classes:
            app_log.info("Using prior_classes")
            # TODO double check the naming with test_idx
            ### 1. wanted keys, values are in trans_model
            if "cil" in args.setup_opt:
                class_classifier_dict = {
                    "gclassifier." + str(test_idx) + "." + k: v
                    for k, v in prior_classes.items()
                }
            else:
                class_classifier_dict = {
                    "gclassifier." + k: v for k, v in prior_classes.items()
                }
            ### 2. overwrite entries in the existing state dict
            model_dict.update(class_classifier_dict)

        if prior_adapter:
            adapter_norm_before_dict = {
                "adapter." + k: v for k, v in prior_adapter.items()
            }

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_norm_before_dict)

        ### 3. load the new state dict
        model.load_state_dict(model_dict)

    all_results = []
    all_examples = []
    all_features = []
    losses = []

    for idx in tqdm(range(nb_examples)):
        if not batch:
            batch_one, examples, features = dataset.next_batch(1, dataset_test)
        batch_one = transfer_batch_cuda(batch_one, device)

        if train_idx > 0 and name != "dev":
            if args.cont_learn_alg == "mbpa":
                """Local adaptation of MbPA"""
                q = model.get_embeddings(
                    batch_one["input_ids"], batch_one["input_masks"]
                )[0]

                # eval_model = cont_learn_alg.forward(memory, q, train_idx, model) # Old this is up to train_idx taking into consideration all memory items in previously seen tasks
                if args.use_reptile:
                    if args.use_batches_reptile:
                        eval_model = cont_learn_alg.forward_reptile_many_batches(
                            memory, q, train_idx, model, dataset
                        )
                    else:
                        eval_model = cont_learn_alg.forward_reptile_one_batch(
                            memory, q, train_idx, model, dataset
                        )
                else:
                    eval_model = cont_learn_alg.forward(
                        memory, q, train_idx, model, dataset
                    )  # this is taking into consideration only the task we are testing from assuming we know that task.
            else:
                eval_model = model
        else:
            eval_model = model

        eval_model.eval()
        # TODO test this in particular
        # TODO do we change anything at all in the original model just to make sure?

        inputs = batch_one
        inputs["train_idx"] = test_idx
        if len(inputs["input_ids"]) == 0:
            continue
        with torch.no_grad():
            outputs = eval_model(**inputs)

        losses.append(outputs.loss["overall"].squeeze().item())

        start_logits = outputs.logits["start_positions"]
        end_logits = outputs.logits["end_positions"]
        for idx_feat, feature in enumerate(features):
            all_results.append(
                SquadResult(
                    feature.unique_id, start_logits[idx_feat], end_logits[idx_feat]
                )
            )
        all_examples.extend(examples)
        all_features.extend(features)

    # output_prediction_file = os.path.join(out_path, "predictions_{}_{}.json".format(train_idx, test_idx))
    # output_nbest_file = os.path.join(out_path, "nbest_predictions_{}_{}.json".format(train_idx, test_idx))

    output_prediction_file = out_path
    output_nbest_file = None  # out_path + "_nbest"

    output_null_log_odds_file = None  # no version_with_negatives for now

    predictions = compute_predictions_logits(
        all_examples,
        all_features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    print(
        "UTILS len(examples):",
        len(all_examples),
        " len(predictions):",
        len(predictions),
    )

    # Compute the F1 and exact scores.
    results, all_exact, all_f1 = squad_evaluate(all_examples, predictions)
    print("results:", results)

    metrics = {}
    per_item_acc = []
    avg_perf = 0.0
    pre_metric_name = train_task + "_" + str(train_idx) + "_" + name
    post_metric_name = test_task + "_" + str(test_idx)

    for k, v in results.items():
        metrics.update({pre_metric_name + "_" + k + "_" + post_metric_name: v})
        avg_perf += v

    avg_perf = avg_perf / len(results)

    if nb_examples > 0:
        per_item_acc = [v for _, v in all_f1.items()]

    return metrics, avg_perf, per_item_acc, losses, all_f1


def evaluate_report(
    tokenizer,
    dataset,
    memory,
    cont_learn_alg,
    data_stream_examples,
    data_stream_size,
    model,
    train_task,  # lang or subtask
    train_idx,
    test_task,  # lang or subtask
    test_idx,
    num_steps,
    writer,
    args,
    app_log,
    device,
    name,
    out_path=None,
    verbose=False,
    prior_mbert=None,
    prior_classes=None,
    prior_slots=None,
    prior_adapter=None,
):
    if args.task_name in ["nli", "tod"]:
        eval_func = nlu_nli_evaluation
    else:  # SQUAD
        eval_func = tydiqa_evaluation

    metrics, avg_perf, per_item_acc, losses, eval_outcomes = eval_func(
        tokenizer,
        dataset,
        memory,
        cont_learn_alg,
        data_stream_examples,
        data_stream_size,
        model,
        train_task,
        train_idx,
        test_task,
        test_idx,
        args,
        app_log,
        device,
        name,
        out_path=out_path,
        verbose=verbose,
        prior_mbert=prior_mbert,
        prior_classes=prior_classes,
        prior_slots=prior_slots,
        prior_adapter=prior_adapter,
    )

    if num_steps > 0:
        for k, v in metrics.items():
            writer.add_scalar(k, v, num_steps)
            if args.task_name in ["nli", "tod"]:
                print(k, round(v * 100, 1))

    return metrics, avg_perf, per_item_acc, losses, eval_outcomes
