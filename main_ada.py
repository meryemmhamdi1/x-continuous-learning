# General
import os
import sys
import pickle
import argparse
import configparser
from tqdm import tqdm
from utils import logger, set_optimizer
from consts import INTENT_TYPES, SLOT_TYPES
from sklearn.metrics import f1_score, precision_score, recall_score

# Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter

# Base Model
from basemodels.crf import CRFLayer
from data_utils import NLUDataset  # alternatively from data_utils_ada import NLUDataset if using AdapterTrainer

# Transformers imports
from transformers import set_seed
import transformers.adapters.composition as ac  # for importing Stack, Parallel
from transformers import AutoTokenizer, AutoModelWithHeads, AdapterConfig


# TODO remove this and normalize with utils.py
def nlu_evaluation(dataset,
                   dataset_test,
                   nb_examples,
                   model,
                   crf_layer,
                   test_idx,
                   app_log,
                   out_path=None,
                   verbose=False):

    app_log.info("Evaluating on i_task: %d", test_idx)

    for k, v in model.named_parameters():
        if "bert.encoder.layer.0." in k and v.requires_grad:
            app_log.info("TESTING Parameter %s, required_grad: %r" % (k, v.requires_grad))

    model.eval()
    crf_layer.eval()

    app_log.info("----------------------------------------------------------------------------------------")

    intent_corrects = 0
    sents_text = []

    intents_true = []
    intents_pred = []

    slots_true = []
    slots_pred = []

    slots_true_all = []
    slots_pred_all = []

    for _ in range(nb_examples):
        (input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts), text \
            = dataset.next_batch(1, dataset_test)

        if DEVICE != torch.device("cpu"):
            input_ids = input_ids.cuda()
            lengths = lengths.cuda()
            input_masks = input_masks.cuda()
            token_type_ids = token_type_ids.cuda()
            intent_labels = intent_labels.cuda()
            slot_labels = slot_labels.cuda()

        if USE_SLOTS:
            with torch.no_grad():
                output1, output2 = model(input_ids=input_ids,
                                         attention_mask=input_masks,
                                         token_type_ids=token_type_ids)

            intent_logits = output1[0]
            slot_logits = crf_layer(output2[0])

            # Slot Golden Truth/Predictions
            true_slot = slot_labels[0]

            slot_logits = [slot_logits[j, :length].data.cpu().numpy() for j, length in enumerate(lengths)]
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

        else:
            with torch.no_grad():
                output1 = model(input_ids=input_ids,
                                attention_mask=input_masks,
                                token_type_ids=token_type_ids)

            intent_logits = output1[0]

        # Intent Golden Truth/Predictions
        true_intent = intent_labels.squeeze().item()
        pred_intent = intent_logits.squeeze().max(0)[1]

        intent_corrects += int(pred_intent == true_intent)

        masked_text = ' '.join(dataset.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()))
        intents_true.append(true_intent)
        intents_pred.append(pred_intent.item())
        sents_text.append(input_texts)

    if out_path:
        with open(out_path, "w") as writer:
            for i in range(len(sents_text)):
                if i < 3:  # print first 3 predictions
                    app_log.info("Sent : %s", sents_text[i][0])
                    app_log.info(" True Intent: ")
                    app_log.info(INTENT_TYPES[intents_true[i]])
                    app_log.info(" Intent Prediction :")
                    app_log.info(INTENT_TYPES[intents_pred[i]])
                    app_log.info(" True Slots: ")
                    app_log.info(" ".join(slots_true[i]))
                    app_log.info(" Slot Prediction:")
                    app_log.info(" ".join(slots_pred[i]))

                text = sents_text[i][0] + "\t" + INTENT_TYPES[intents_true[i]] + "\t" + INTENT_TYPES[intents_pred[i]] \
                    + "\t" + " ".join(slots_true[i]) + "\t" + " ".join(slots_pred[i])
                writer.write(text+"\n")

    if verbose:
        app_log.info(test_idx)
        app_log.info(" -----------intents_true:")
        app_log.info(set(intents_true))
        app_log.info(" -----------intents_pred:")
        app_log.info(set(intents_pred))

    if nb_examples > 0:
        intent_accuracy = float(intent_corrects) / nb_examples
        intent_prec = precision_score(intents_true, intents_pred, average="macro")
        intent_rec = recall_score(intents_true, intents_pred, average="macro")
        intent_f1 = f1_score(intents_true, intents_pred, average="macro")
        if USE_SLOTS:
            slot_prec = precision_score(slots_true_all, slots_pred_all, average="macro")
            slot_rec = recall_score(slots_true_all, slots_pred_all, average="macro")
            slot_f1 = f1_score(slots_true_all, slots_pred_all, average="macro")

            return intent_accuracy, intent_prec, intent_rec, intent_f1, slot_prec, slot_rec, slot_f1

        return intent_accuracy, intent_prec, intent_rec, intent_f1
    else:
        intent_accuracy = 0.0
        intent_prec = 0.0
        intent_rec = 0.0
        intent_f1 = 0.0
        if USE_SLOTS:
            slot_prec = 0.0
            slot_rec = 0.0
            slot_f1 = 0.0
            return intent_accuracy, intent_prec, intent_rec, intent_f1, slot_prec, slot_rec, slot_f1

        return intent_accuracy, intent_prec, intent_rec, intent_f1


# TODO: remove this and normalize with utils
def evaluate_report(dataset,
                    data_stream,
                    model,
                    crf_layer,
                    train_task,  # lang or subtask
                    train_idx,
                    test_task,  # lang or subtask
                    test_idx,
                    num_steps,
                    writer,
                    app_log,
                    device,
                    name,
                    out_path=None,
                    verbose=False):

    outputs = nlu_evaluation(dataset,
                             data_stream["examples"],
                             data_stream["size"],
                             model,
                             crf_layer,
                             test_idx,
                             app_log,
                             out_path=out_path,
                             verbose=verbose)

    output_text_format = "----size=%d, test_index=%d, and task=%s" % (data_stream["size"],
                                                                      test_idx,
                                                                      test_task)

    metrics = {}
    if not USE_SLOTS:
        intent_acc, intent_prec, intent_rec, intent_f1 = outputs
        avg_perf = intent_acc

    else:
        intent_acc, intent_prec, intent_rec, intent_f1, slot_prec, slot_rec, slot_f1 = outputs

        output_text_format += " SLOTS perf: (prec=%f, rec=%f, f1=%f) " % (round(slot_prec*100, 1),
                                                                          round(slot_rec*100, 1),
                                                                          round(slot_f1*100, 1))

        avg_perf = (intent_acc + slot_f1) / 2

        metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_slot_prec_'+test_task+'_'+str(test_idx): slot_prec})
        metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_slot_rec_'+test_task+'_'+str(test_idx): slot_rec})
        metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_slot_f1_'+test_task+'_'+str(test_idx): slot_f1})

    metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_intent_acc_'+test_task+'_'+str(test_idx): intent_acc})
    metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_intent_prec_'+test_task+'_'+str(test_idx): intent_prec})
    metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_intent_rec_'+test_task+'_'+str(test_idx): intent_rec})
    metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_intent_f1_'+test_task+'_'+str(test_idx): intent_f1})

    output_text_format += " INTENTS perf: (acc: %f, prec: %f, rec: %f, f1: %f)" % (round(intent_acc*100, 1),
                                                                                   round(intent_prec*100, 1),
                                                                                   round(intent_rec*100, 1),
                                                                                   round(intent_f1*100, 1))

    app_log.info(output_text_format)
    for k, v in metrics.items():
        writer.add_scalar(k, v, num_steps)

    return metrics, avg_perf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./main.py", description="Different options for running adapters")

    option_params = parser.add_argument_group("Options for freezing or not bert and adding or not task adapters")

    option_params.add_argument("--languages", help="List of languages to train on in the stream",
                               type=str, default="en_de_fr_hi_es_th")

    option_params.add_argument("--freeze_bert", help="Whether to freeze bert or not",
                               action="store_true")

    option_params.add_argument("--use_task_adapters", help="Whether to add task adapters or not",
                               action="store_true")

    option_params.add_argument("--use_mono", help="Whether to train monolingually",
                               action="store_true")

    option_params.add_argument("--train_from_scratch", help="Whether to train the adapter weights from scratch",
                               action="store_true")

    option_params.add_argument("--seed", help="Random seed",
                               type=int, default=42)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('scripts/hyperparam.ini')

    SEED = args.seed
    set_seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FROZEN_BERT = args.freeze_bert
    USE_TASK_ADAPTERS = args.use_task_adapters
    TRAIN_FROM_SCRATCH = args.train_from_scratch
    LANGUAGES = args.languages.split("_")
    LOCATION = "cluster"  # LOCATION = "local"
    MODEL_NAME = "BertBaseMultilingualCased"  # or "XLMRoberta_base"
    SETUP_OPT = "cll"
    SETUP_CILLIA = "intents"

    ORDER_CLASS = 0
    ORDER_LANG = 0
    ORDER_LST = args.languages
    USE_MONO = args.use_mono
    MULTI_HEAD_OUT = False
    USE_SLOTS = True
    VERBOSE = True

    DATA_FORMAT = "txt"
    SPLITS = ["train", "eval", "test"]

    # Hyperparameters
    BATCH_SIZE = int(config.get("HYPER", "BATCH_SIZE"))
    EPOCHS = int(config.get("HYPER", "EPOCHS"))
    ADAM_LR = float(config.get("HYPER", "ADAM_LR"))
    ADAM_EPS = float(config.get("HYPER", "ADAM_EPS"))
    BETA_1 = float(config.get("HYPER", "BETA_1"))
    BETA_2 = float(config.get("HYPER", "BETA_2"))
    EPSILON = float(config.get("HYPER", "EPSILON"))
    STEP_SIZE = float(config.get("HYPER", "STEP_SIZE"))
    GAMMA = float(config.get("HYPER", "GAMMA"))
    TEST_STEPS = int(config.get("HYPER", "TEST_STEPS"))
    NUM_INTENT_TASKS = int(config.get("HYPER", "NUM_INTENT_TASKS"))  # only in case of CILIA setup
    NUM_LANG_TASKS = int(config.get("HYPER", "NUM_LANG_TASKS"))  # only in case of CILIA setup

    print("DEVICE:", DEVICE, "FROZEN_BERT:", FROZEN_BERT, "USE_TASK_ADAPTERS:", USE_TASK_ADAPTERS,
          "LANGUAGES:", args.languages)

    if LOCATION == "local":
        DATA_ROOT = "/Users/d22admin/USCGDrive/Spring21/Research/XContLearn/Datasets/NLU/MTOP/"
        OUT_DIR = "/Users/d22admin/USCGDrive/Spring21/Research/XContLearn/Results/"
        MODEL_DIR = "bert-base-multilingual-cased"
    else:
        DATA_ROOT = "/project/jonmay_231/meryem/Datasets/MTOP/"
        OUT_DIR = "/home1/mmhamdi/Results/x-continuous-learn/"
        if MODEL_NAME == "BertBaseMultilingualCased":
            MODEL_DIR = "/project/jonmay_231/meryem/Models/mbert-with-heads"
        else:  # "XLMRoberta_base"
            MODEL_DIR = "/home1/mmhamdi/xlmr-base-with-heads"

    print("MODEL_DIR:", MODEL_DIR)

    """ 1. Setting the results directory """
    results_dir = os.path.join(OUT_DIR,  # original output directory
                               SETUP_OPT,  # setup option directory
                               "adapterHUB",
                               args.languages,  # language order
                               (lambda x: "NLU" if x else "Intents_only")(USE_SLOTS),  # slot usage
                               MODEL_NAME,
                               (lambda x: "FROZEN_BERT" if x else "TUNED_BERT")(FROZEN_BERT),
                               (lambda x: "USE_TASK_ADAPTERS" if x else "NO_TASK_ADAPTERS")(USE_TASK_ADAPTERS),
                               (lambda x: "TRAINING_ADAPTERS" if x else "LOADED_ADAPTERS")(TRAIN_FROM_SCRATCH))

    if USE_MONO:
        results_dir = os.path.join(results_dir, "USE_MONO_"+args.languages)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    writer = SummaryWriter(os.path.join(results_dir, 'runs'))
    app_log = logger(os.path.join(results_dir, "log.txt"))
    metrics_dir = os.path.join(results_dir, "metrics")
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir)

    app_log.info("Saving to results_dir %s", results_dir)

    stdoutOrigin = sys.stdout
    sys.stdout = open(os.path.join(results_dir, "log.txt"), "w")

    """ 2. Tokenizer and dataset loading """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    dataset = NLUDataset(DATA_ROOT,
                         SETUP_OPT,
                         SETUP_CILLIA,
                         MULTI_HEAD_OUT,
                         USE_MONO,
                         tokenizer,
                         DATA_FORMAT,
                         USE_SLOTS,
                         SEED,
                         LANGUAGES,
                         ORDER_CLASS,
                         ORDER_LANG,
                         ORDER_LST,
                         NUM_INTENT_TASKS,
                         NUM_LANG_TASKS,
                         intent_types=INTENT_TYPES,
                         slot_types=SLOT_TYPES)

    model = AutoModelWithHeads.from_pretrained(MODEL_DIR)
    crf_layer = CRFLayer(len(SLOT_TYPES), DEVICE)

    """ 3. Language Language Adapters """
    lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)

    if TRAIN_FROM_SCRATCH:
        for lang in LANGUAGES:
            model.add_adapter(lang, config=lang_adapter_config)
    else:
        for lang in LANGUAGES:
            if LOCATION == "local":
                model.load_adapter(lang+"/wiki@ukp", config=lang_adapter_config)
            else:
                model.load_adapter(MODEL_DIR+"/"+lang+"_ada", config=lang_adapter_config)

    """ 4. Prediction Heads """
    if USE_TASK_ADAPTERS:
        model.add_adapter("nlu_intent_head")

    model.add_classification_head("nlu_intent_head", num_labels=len(INTENT_TYPES), layers=1)
    #
    if USE_SLOTS:
        if USE_TASK_ADAPTERS:
            model.add_adapter("nlu_slot_head")
        model.add_tagging_head("nlu_slot_head", num_labels=len(SLOT_TYPES))
        if not USE_TASK_ADAPTERS:
            model.active_head = ["nlu_intent_head", "nlu_slot_head"]

    if DEVICE != torch.device("cpu"):
        model = model.cuda()

    metrics = {"train_"+lang: {"test_"+lang: {} for lang in LANGUAGES} for lang in LANGUAGES}

    args.beta_1 = BETA_1
    args.beta_2 = BETA_2
    args.adam_eps = ADAM_EPS
    args.adam_lr = ADAM_LR
    args.step_size = STEP_SIZE
    args.gamma = GAMMA

    optimizer, scheduler = set_optimizer(args, list(model.parameters()) + list(crf_layer.parameters()))
    model.zero_grad()
    crf_layer.zero_grad()

    train_stream = dataset.train_stream
    dev_stream = dataset.dev_stream
    test_stream = dataset.test_stream

    for train_idx, train_subtask_lang in enumerate(train_stream):
        train_examples = train_subtask_lang["examples"]
        train_lang = train_subtask_lang["lang"]
        num_iter = train_subtask_lang["size"]//BATCH_SIZE

        if USE_SLOTS:
            if USE_TASK_ADAPTERS:
                model.active_adapters = ac.Stack(train_lang, ac.Parallel("nlu_intent_head", "nlu_slot_head"))
                model.train_adapter([train_lang, "nlu_intent_head", "nlu_slot_head"])
            else:
                model.active_adapters = train_lang
                model.train_adapter([train_lang])
        else:
            if USE_TASK_ADAPTERS:
                model.active_adapters = ac.Stack(train_lang, "nlu_intent_head")
                model.train_adapter([train_lang, "nlu_intent_head"])
            else:
                model.active_adapters = train_lang
                model.train_adapter([train_lang])

        num_steps = 0
        dev_perf_best = 0.0
        best_model = None
        best_crf = None

        if not FROZEN_BERT:
            for k, v in model.named_parameters():
                if "bert." in k and "adapters" not in k:
                    v.requires_grad = True

        for k, v in model.named_parameters():
            if "bert.encoder.layer" not in k:
                app_log.info("TRAINING Parameter %s, required_grad: %r" % (k, v.requires_grad))
            if v.requires_grad and "bert.encoder.layer.0." in k:
                app_log.info("TRAINING Parameter %s, required_grad: %r" % (k, v.requires_grad))

        for epoch in tqdm(range(EPOCHS)):
            for step_iter in range(num_iter):
                num_steps += 1

                optimizer.zero_grad()
                model.train()
                crf_layer.train()

                batch, _ = dataset.next_batch(BATCH_SIZE, train_examples)

                input_ids, _, token_type_ids, input_masks, intent_labels, slot_labels, input_texts = batch

                if DEVICE != torch.device("cpu"):
                    input_ids = input_ids.cuda()
                    token_type_ids = token_type_ids.cuda()
                    input_masks = input_masks.cuda()
                    intent_labels = intent_labels.cuda()
                    slot_labels = slot_labels.cuda()

                if USE_SLOTS:
                    output1, output2 = model(input_ids=input_ids,
                                             attention_mask=input_masks,
                                             token_type_ids=token_type_ids)

                    print("output1:", output1)
                    print("output2:", output2)

                    intent_loss = torch.nn.CrossEntropyLoss()(output1[0], intent_labels)
                    slot_loss = crf_layer.loss(output2[0], slot_labels)
                    loss = intent_loss + slot_loss
                    app_log.info(" Training: (intent_loss: %f, slot_loss: %f, total loss: %f)"
                                 % (intent_loss, slot_loss, loss))
                else:
                    output1 = model(input_ids=input_ids,
                                    attention_mask=input_masks,
                                    token_type_ids=token_type_ids)

                    intent_loss = torch.nn.CrossEntropyLoss()(output1[0], intent_labels)
                    loss = intent_loss

                loss = loss.mean()
                loss.backward()
                optimizer.step()

            dev_out_path = os.path.join(results_dir,
                                        "Dev_perf-Epoch_" + str(epoch) + "-train_" + str(train_idx))

            # Check dev performance at the end of the epoch
            if dev_stream[train_idx]['size'] > 0:
                _, dev_perf = evaluate_report(dataset,
                                              dev_stream[train_idx],
                                              model,
                                              crf_layer,
                                              train_lang,
                                              train_idx,
                                              train_lang,
                                              train_idx,
                                              num_steps,
                                              writer,
                                              app_log=app_log,
                                              name="dev",
                                              out_path=dev_out_path)
            else:
                dev_perf = 0

            if dev_perf > dev_perf_best:
                dev_perf_best = dev_perf

                best_model = model
                best_crf = crf_layer

            if best_model is None:
                best_model = model

            if best_crf is None:
                best_crf = crf_layer

        app_log.info("------------------------------------ TESTING At the end of the training")
        metrics_sub = {"test_"+task: {} for task in test_stream}  # could be either per subtask or language
        for test_idx, test_subtask_lang in enumerate(test_stream):
            app_log.info("Testing on %s" % test_subtask_lang)
            if test_stream[test_subtask_lang]['size'] > 0:
                if USE_TASK_ADAPTERS:
                    model.active_adapters = ac.Stack(test_subtask_lang, ac.Parallel("nlu_intent_head", "nlu_slot_head"))
                else:
                    model.active_adapters = test_subtask_lang

                metrics_sub["test_"+test_subtask_lang], _ = evaluate_report(dataset,
                                                                            test_stream[test_subtask_lang],
                                                                            best_model,
                                                                            best_crf,
                                                                            train_lang,
                                                                            train_idx,
                                                                            test_subtask_lang,
                                                                            test_idx,
                                                                            num_steps,
                                                                            writer,
                                                                            app_log=app_log,
                                                                            name="test",
                                                                            out_path=os.path.join(results_dir,
                                                                                                  "End_test_perf-train_"
                                                                                                  + train_lang +"-test_"
                                                                                                  + test_subtask_lang),
                                                                            verbose=VERBOSE)

        metrics["train_"+train_lang] = metrics_sub

        with open(os.path.join(metrics_dir, "final_metrics_"+str(train_idx)+".pickle"), "wb") as output_file:
            pickle.dump(metrics, output_file)

    with open(os.path.join(results_dir, "all_metrics_"+str(EPOCHS)+"_epochs.pickle"), "wb") as file:
        pickle.dump(metrics, file)

    sys.stdout.close()
    sys.stdout = stdoutOrigin
