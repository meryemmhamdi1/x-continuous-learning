# From the project
from data_utils import *
from transformers_config import MODELS_dict
from basemodels.transNLUCRF import TransNLUCRF
from consts import intent_types, slot_types
from contlearnalg.EWC_grads import EWC
from contlearnalg.GEM import GEM
from utils import format_store_grads, name_in_list, logger

# Torch
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR as SchedulerLR

# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score

# Other Python Modules
import gc
import sys
import logstats
import os
import pickle
import numpy as np
import copy
import argparse
from copy import deepcopy
from tqdm import tqdm

# GPU/CPU
gpus_list = list(range(torch.cuda.device_count()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


def compute_change(mbert1, mbert2,
                   intent1, intent2,
                   slot1, slot2):

    sum_layers = {"trans_model."+k: 0.0 for k, v in mbert1.named_parameters()}
    sum_layers.update({"intent_classifier."+k: 0.0 for k, v in intent1.named_parameters()})
    sum_layers.update({"slot_classifier."+k: 0.0 for k, v in slot1.named_parameters()})
    sum_layers.update({"mbert": 0.0})
    sum_layers.update({"intent": 0.0})
    sum_layers.update({"slot": 0.0})
    sum_layers.update({"all": 0.0})

    mean_layers = {"trans_model."+k: 0.0 for k, v in mbert1.named_parameters()}
    mean_layers.update({"intent_classifier."+k: 0.0 for k, v in intent1.named_parameters()})
    mean_layers.update({"slot_classifier."+k: 0.0 for k, v in slot1.named_parameters()})
    mean_layers.update({"mbert": 0.0})
    mean_layers.update({"intent": 0.0})
    mean_layers.update({"slot": 0.0})
    mean_layers.update({"all": 0.0})

    mbert2_items = {"trans_model."+k: v for k, v in mbert2.named_parameters()}

    count = 0
    count_bert = 0
    for k, v1 in mbert1.named_parameters():
        k = "trans_model." + k
        count += 1
        count_bert += 1
        v2 = mbert2_items[k].data.numpy()
        v1 = v1.data.cpu().numpy()
        res = v1 - v2

        sum = np.sum(res)
        sum_layers[k] = sum
        sum_layers["mbert"] += sum
        sum_layers["all"] += sum

        mean = np.mean(res)
        mean_layers[k] = mean
        mean_layers["mbert"] += mean
        mean_layers["all"] += mean

    intent2_items = {"intent_classifier." + k: v for k, v in intent2.named_parameters()}
    for k, v1 in intent1.named_parameters():
        k = "intent_classifier." + k

        count += 1
        v2 = intent2_items[k].data.cpu().numpy()
        v1 = v1.data.cpu().numpy()
        res = v1 - v2

        sum = np.sum(res)
        sum_layers[k] = sum
        sum_layers["all"] += sum
        sum_layers["intent"] += sum

        mean = np.mean(res)
        mean_layers[k] = mean
        mean_layers["all"] += mean
        mean_layers["intent"] += mean

    slot2_items = {"slot_classifier." + k: v for k, v in slot2.named_parameters()}
    for k, v1 in slot1.named_parameters():
        k = "slot_classifier." + k

        count += 1
        v2 = slot2_items[k].data.cpu().numpy()
        v1 = v1.data.cpu().numpy()
        res = v1 - v2

        sum = np.sum(res)
        sum_layers[k] = sum
        sum_layers["all"] += sum
        sum_layers["slot"] += sum

        mean = np.mean(res)
        mean_layers[k] = mean
        mean_layers["all"] += mean
        mean_layers["slot"] += mean

    print("-------------All mean_all:", mean_layers["all"]/count, " sum_all:", sum_layers["all"],
          " mean_mbert:", mean_layers["mbert"]/count_bert, " sum_mbert:", sum_layers["mbert"],
          " mean_intent:", mean_layers["intent"]/2, " sum_intent:", sum_layers["intent"]/2,
          " mean_slot:", mean_layers["slot"]/2, " sum_slot:", sum_layers["slot"]/2)

    return mean_layers, sum_layers


def train(optimizer,
          model,
          grad_dims,
          cont_learn_alg,
          dataset,
          train_examples,
          train_examples_size,
          writer,
          epoch,
          i_task,
          num_steps,
          checkpoint_dir,
          sample_sizes=[]):

    optimizer.zero_grad()
    model.train()

    # Take batch by batch and move to cuda
    batch, _ = dataset.next_batch(args.batch_size, train_examples)

    input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts = batch

    input_ids = input_ids.cuda()
    lengths = lengths.cuda()
    token_type_ids = token_type_ids.cuda() # this is not needed at all
    input_masks = input_masks.cuda() # TODO compare when you use this versus when you don't
    intent_labels = intent_labels.cuda()
    slot_labels = slot_labels.cuda()

    if args.use_slots:
        logits_intents, logits_slots, intent_loss, slot_loss, loss = model(input_ids=input_ids,
                                                                           input_masks=input_masks,
                                                                           train_idx=i_task,
                                                                           lengths=lengths,
                                                                           intent_labels=intent_labels,
                                                                           slot_labels=slot_labels)

        writer.add_scalar('train_intent_loss_'+str(i_task), intent_loss.mean(), num_steps*epoch)
        writer.add_scalar('train_slot_loss_'+str(i_task), slot_loss.mean(), num_steps*epoch)
    else:
        logits_intents, intent_loss, loss = model(input_ids=input_ids,
                                                  input_masks=input_masks,
                                                  train_idx=i_task,
                                                  lengths=lengths,
                                                  intent_labels=intent_labels)
        writer.add_scalar('train_intent_loss_'+str(i_task), intent_loss.mean(), num_steps*epoch)

    if args.cont_learn_alg == "ewc":
        if i_task > 0:
            loss += (args.ewc_lambda / 2) * cont_learn_alg.penalty(i_task,
                                                                   model)

        loss = loss.mean()

        loss.backward()

        params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        saved_grads = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            saved_grads[n] = variable(p.data)

        for n, p in model.named_parameters():
            if p.grad is not None and p.requires_grad:
                saved_grads[n].data += p.grad.data ** 2 / train_examples_size[i_task]

        saved_grads = {n: p for n, p in saved_grads.items()}

    elif args.cont_learn_alg == "gem":

        loss = loss.mean()
        loss.backward()
        format_store_grads(pp=model.named_parameters(),
                           grad_dims=grad_dims,
                           cont_comp=args.cont_comp,
                           checkpoint_dir=checkpoint_dir,
                           tid=i_task,
                           store=True)  # storing for the current task
        if i_task > 0:
            cont_learn_alg.run(i_task,
                               sample_sizes,
                               model,
                               grad_dims)

        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        saved_grads = None

    else:
        loss = loss.mean()
        loss.backward()

        params = None
        saved_grads = None

    optimizer.step()

    if args.use_slots:
        return intent_loss, slot_loss, params, saved_grads, optimizer, model

    return intent_loss, params, saved_grads, optimizer, model


def nlu_evaluation(dataset,
                   dataset_test,
                   nb_examples,
                   model,
                   use_slots,
                   test_idx,
                   out_path=None,
                   verbose=False,
                   prior_mbert=None,
                   prior_intents=None,
                   prior_slots=None,
                   prior_adapter_norm_before=None,
                   prior_adapter_down_1=None,
                   prior_adapter_up_1=None,
                   prior_adapter_norm_after_1=None,
                   prior_adapter_feed_layer_1=None,
                   prior_adapter_feed_layer_2=None,
                   prior_adapter_down_2=None,
                   prior_adapter_up_2=None,
                   prior_adapter_norm_after_2=None):

    app_log.info("Evaluating on i_task: %d", test_idx)
    if prior_mbert or prior_intents or prior_slots or prior_adapter_norm_before or prior_adapter_down_1 \
            or prior_adapter_up_1 or prior_adapter_norm_after_1 or prior_adapter_feed_layer_1 or \
            prior_adapter_feed_layer_2 or prior_adapter_down_2 or prior_adapter_up_2 or \
            prior_adapter_norm_after_2:

        model_dict = model.state_dict()

        if prior_mbert:
            app_log.info("Using prior_mbert")
            ### 1. wanted keys, values are in trans_model
            trans_model_dict = {"trans_model."+k: v for k, v in prior_mbert.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(trans_model_dict)

        if prior_intents:
            app_log.info("Using prior_intents")
            # TODO double check the naming with test_idx
            ### 1. wanted keys, values are in trans_model
            if "cil" in args.setup_opt:
                intent_classifier_dict = {"intent_classifier."+str(test_idx)+"."+k: v for k, v in prior_intents.items()}
            else:
                intent_classifier_dict = {"intent_classifier."+k: v for k, v in prior_intents.items()}
            ### 2. overwrite entries in the existing state dict
            model_dict.update(intent_classifier_dict)

        if prior_slots:
            app_log.info("Using prior_slots")
            ### 1. wanted keys, values are in trans_model
            slot_classifier_dict = {"slot_classifier."+k: v for k, v in prior_slots.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(slot_classifier_dict)

        if prior_adapter_norm_before:
            adapter_norm_before_dict = {"adapter.adapter_norm_before."+k: v for k, v in prior_adapter_norm_before.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_norm_before_dict)

        if prior_adapter_down_1:
            adapter_down_1_dict = {"adapter.adapter_down_1.1."+k: v for k, v in prior_adapter_down_1.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_down_1_dict)

        if prior_adapter_up_1:
            adapter_up_1_dict = {"adapter.adapter_up_1."+k: v for k, v in prior_adapter_up_1.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_up_1_dict)

        if prior_adapter_norm_after_1:
            adapter_norm_after_after_1_dict = {"adapter.adapter_norm_after_1."+k: v for k, v in
                                               prior_adapter_norm_after_1.items()}
            model_dict.update(adapter_norm_after_after_1_dict)

        if prior_adapter_feed_layer_1:
            adapter_feed_layer_1_dict = {"adapter.feed_layer_1."+k: v for k, v in
                                         prior_adapter_feed_layer_1.items()}
            model_dict.update(adapter_feed_layer_1_dict)

        if prior_adapter_feed_layer_2:
            adapter_feed_layer_2_dict = {"adapter.feed_layer_2."+k: v for k, v in
                                         prior_adapter_feed_layer_2.items()}
            model_dict.update(adapter_feed_layer_2_dict)

        if prior_adapter_down_2:
            adapter_down_2_dict = {"adapter.adapter_down_2.0."+k: v for k, v in
                                   prior_adapter_down_2.items()}
            model_dict.update(adapter_down_2_dict)

        if prior_adapter_up_2:
            adapter_up_2_dict = {"adapter.adapter_up_2."+k: v for k, v in
                                 prior_adapter_up_2.items()}
            model_dict.update(adapter_up_2_dict)

        if prior_adapter_norm_after_2:
            adapter_norm_after_after_2_dict = {"adapter.adapter_norm_after_2."+k: v for k, v in
                                               prior_adapter_norm_after_2.items()}
            model_dict.update(adapter_norm_after_after_2_dict)


        ### 3. load the new state dict
        model.load_state_dict(model_dict)

    model.eval()

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

        input_ids = input_ids.cuda()
        lengths = lengths.cuda()
        input_masks = input_masks.cuda()
        intent_labels = intent_labels.cuda()
        slot_labels = slot_labels.cuda()

        if use_slots:
            intent_logits, slot_logits, intent_loss, slot_loss, loss = model(input_ids=input_ids,
                                                                             input_masks=input_masks,
                                                                             train_idx=test_idx,
                                                                             lengths=lengths,
                                                                             intent_labels=intent_labels,
                                                                             slot_labels=slot_labels)

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
            intent_logits, intent_loss, loss = model(input_ids=input_ids,
                                                     input_masks=input_masks,
                                                     train_idx=test_idx,
                                                     lengths=lengths,
                                                     intent_labels=intent_labels)



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
                    app_log.info(intent_types[intents_true[i]])
                    app_log.info(" Intent Prediction :")
                    app_log.info(intent_types[intents_pred[i]])
                    app_log.info(" True Slots: ")
                    app_log.info(" ".join(slots_true[i]))
                    app_log.info(" Slot Prediction:")
                    app_log.info(" ".join(slots_pred[i]))

                text = sents_text[i][0] + "\t" + intent_types[intents_true[i]] + "\t" + intent_types[intents_pred[i]] \
                         + "\t" + " ".join(slots_true[i]) + "\t" + " ".join(slots_pred[i])
                writer.write(text+"\n")

    if verbose:
        app_log.info(test_idx)
        app_log.info(" -----------intents_true:")
        app_log.info(set(intents_true))
        app_log.info(" -----------intents_pred:")
        app_log.info(set(intents_pred))

    intent_accuracy = float(intent_corrects) / nb_examples
    intent_prec = precision_score(intents_true, intents_pred, average="macro")
    intent_rec = recall_score(intents_true, intents_pred, average="macro")
    intent_f1 = f1_score(intents_true, intents_pred, average="macro")

    if use_slots:
        slot_prec = precision_score(slots_true_all, slots_pred_all, average="macro")
        slot_rec = recall_score(slots_true_all, slots_pred_all, average="macro")
        slot_f1 = f1_score(slots_true_all, slots_pred_all, average="macro")

        return intent_accuracy, intent_prec, intent_rec, intent_f1, slot_prec, slot_rec, slot_f1

    return intent_accuracy, intent_prec, intent_rec, intent_f1


def evaluate_report(dataset,
                    data_stream,
                    model,
                    train_task,  # lang or subtask
                    train_idx,
                    test_task,  # lang or subtask
                    test_idx,
                    num_steps,
                    writer,
                    name,
                    out_path=None,
                    verbose=False,
                    prior_mbert=None,
                    prior_intents=None,
                    prior_slots=None,
                    prior_adapter_norm_before=None,
                    prior_adapter_down_1=None,
                    prior_adapter_up_1=None,
                    prior_adapter_norm_after_1=None,
                    prior_adapter_feed_layer_1=None,
                    prior_adapter_feed_layer_2=None,
                    prior_adapter_down_2=None,
                    prior_adapter_up_2=None,
                    prior_adapter_norm_after_2=None):

    outputs = nlu_evaluation(dataset,
                             data_stream["examples"],
                             data_stream["size"],
                             model,
                             args.use_slots,
                             test_idx,
                             out_path=out_path,
                             verbose=verbose,
                             prior_mbert=prior_mbert,
                             prior_intents=prior_intents,
                             prior_slots=prior_slots,
                             prior_adapter_norm_before=prior_adapter_norm_before,
                             prior_adapter_down_1=prior_adapter_down_1,
                             prior_adapter_up_1=prior_adapter_up_1,
                             prior_adapter_norm_after_1=prior_adapter_norm_after_1,
                             prior_adapter_feed_layer_1=prior_adapter_feed_layer_1,
                             prior_adapter_feed_layer_2=prior_adapter_feed_layer_2,
                             prior_adapter_down_2=prior_adapter_down_2,
                             prior_adapter_up_2=prior_adapter_up_2,
                             prior_adapter_norm_after_2=prior_adapter_norm_after_2)

    output_text_format = "----size=%d, test_index=%d, and task=%s" % (data_stream["size"],
                                                                      test_idx,
                                                                      test_task)

    metrics = {}
    if not args.use_slots:
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


def set_optimizer(model):
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     betas=(args.beta_1, args.beta_2),
                     eps=args.adam_eps,
                     lr=args.adam_lr)

    scheduler = SchedulerLR(optimizer,
                            step_size=args.step_size,
                            gamma=args.gamma)

    model.zero_grad()

    return optimizer, scheduler


def set_out_dir():
    # out_dir -> setup_opt -> slots -> trans_model -> random_init/mono/lang_order -> class_order -> cont_learn_alg ->
    #  -> headed -> adapters -> freezing
    order_lang_dict = {0: "high2lowlang",
                       1: "low2highlang",
                       2: "randomlang"}

    order_class_dict = {0: "high2lowclass",
                        1: "low2highclass",
                        2: "randomclass"}

    results_dir = os.path.join(args.out_dir, # original output directory
                               args.setup_opt, # setup option directory
                               (lambda x: "NLU" if x else "Intents_only")(args.use_slots), # slot usage
                               args.trans_model) # transformers model

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    if args.random_pred:
        results_dir = os.path.join(results_dir,
                                   "random_init")

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        return results_dir

    if args.use_mono:
        results_dir = os.path.join(results_dir,
                                   "MONO",
                                   args.languages[0])

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        return results_dir

    if args.setup_opt not in ["multi"]:
        # the order of languages and classes and continuous learning algorithms are only specific to non multi setups
        order_lang = "_".join(args.order_str) if len(args.order_str) > 0 else order_lang_dict[args.order_lang]

        cont_alg_option = args.cont_learn_alg
        if args.cont_learn_alg != "vanilla":
            cont_alg_option += "_"+str(args.old_task_prop)

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
                    else:
                        return "enc."+layer.split(".")[2]

                head_options = "multi_head_in"

                if args.multi_head_out:
                    head_options += "_out/"

                head_options += "-".join(list(map(map_emb_enc_subtask, args.emb_enc_subtask_spec)))

            else:
                if args.multi_head_out:
                    head_options = "multi_head_out"

        order_class = order_class_dict[args.order_class]

        if "cil" in args.setup_opt:
            order_class = os.path.join(order_class, args.cil_stream_lang)

        results_dir = os.path.join(results_dir,
                                   order_lang,  # language order
                                   order_class, # class order
                                   cont_alg_option, # continuous learning algorithm
                                   head_options, # multi-headed option
                                   (lambda x: "adapters" if x else "no_adapters")(args.use_adapters),
                                   args.adapter_type,
                                   "-".join(args.adapter_layers))# adapters option

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)


    # Freezing Options
    results_dir = os.path.join(results_dir,
                               (lambda x: "freeze_trans" if x else "tune_all_trans")(args.freeze_trans),
                               (lambda x: "freeze_linear" if x else "tune_all_linear")(args.freeze_linear),
                               "SEED_"+str(args.seed))

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    return results_dir


def train_task_epochs(model,
                      optimizer,
                      grad_dims,
                      cont_learn_alg,
                      dataset,
                      train_examples,
                      dev_stream,
                      test_stream,
                      subtask_size,
                      sample_sizes,
                      num_iter,
                      train_idx, ## to be used for both the training language and subtask if cll this doesn't matter, if cil this helps
                      train_lang,
                      num_steps,
                      writer, # saving options
                      results_dir,
                      metrics_dir,
                      checkpoint_dir,
                      args_save_file,
                      model_save_file,
                      optim_save_file,
                      prior_mbert, # prior options
                      prior_intents,
                      prior_slots,
                      prior_adapter_norm_before,
                      prior_adapter_down_1,
                      prior_adapter_up_1,
                      prior_adapter_norm_after_1,
                      prior_adapter_feed_layer_1,
                      prior_adapter_feed_layer_2,
                      prior_adapter_down_2,
                      prior_adapter_up_2,
                      prior_adapter_norm_after_2):

    dev_perf_best = 0.0
    best_model = None
    for epoch in tqdm(range(args.epochs)):
        gc.collect()
        num_steps += 1
        for step_iter in range(num_iter):
            train_outputs = train(optimizer,
                                  model,
                                  grad_dims,
                                  cont_learn_alg,
                                  dataset,
                                  train_examples,
                                  subtask_size,
                                  writer,
                                  epoch,
                                  train_idx,
                                  step_iter,
                                  checkpoint_dir,
                                  sample_sizes=sample_sizes)

            if args.use_slots:
                intent_loss, slot_loss, params, saved_grads, optimizer, model = train_outputs
                if step_iter % args.test_steps == 0:
                    app_log.info('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(step_iter,
                                                                                       intent_loss.mean(),
                                                                                       slot_loss.mean()))
            else:
                intent_loss, params, saved_grads, optimizer, model = train_outputs
                if step_iter % args.test_steps == 0:
                    app_log.info('Iter {} | Intent Loss = {:.4f} '.format(step_iter,
                                                                   intent_loss.mean()))

        app_log.info(">>>>>>> Dev Performance >>>>>")
        dev_out_path = None
        if args.save_dev_pred:
            dev_out_path = os.path.join(results_dir,
                                        "Dev_perf-Epoch_" + str(epoch) + "-train_" + train_idx)

        if dev_stream[train_idx]['size'] > 0:
            _, dev_perf = evaluate_report(dataset,
                                          dev_stream[train_idx],
                                          model,
                                          train_lang,
                                          train_idx,
                                          train_lang,
                                          train_idx,
                                          num_steps,
                                          writer,
                                          name="dev",
                                          out_path=dev_out_path)
        else:
            dev_perf = 0

        if dev_perf > dev_perf_best:
            dev_perf_best = dev_perf

            if args.save_model:
                torch.save(args, args_save_file)
                torch.save(model.state_dict(), model_save_file)
                torch.save(optimizer.state_dict(), optim_save_file)

            best_model = model

        if best_model is None:
            best_model = model

        if args.save_test_every_epoch:
            metrics = {subtask_lang: {} for subtask_lang in test_stream}
            for test_idx, test_subtask_lang in enumerate(test_stream):
                if test_stream[test_subtask_lang]['size'] > 0:
                    metrics[test_subtask_lang], _ = evaluate_report(dataset,
                                                                    test_stream[test_subtask_lang],
                                                                    best_model,
                                                                    train_lang,
                                                                    train_idx,
                                                                    test_subtask_lang,
                                                                    test_idx,
                                                                    num_steps,
                                                                    writer,
                                                                    name="test",
                                                                    out_path=os.path.join(results_dir,
                                                                                          "Test_perf-Epoch_" + str(epoch)
                                                                                          + "-train_" + train_lang
                                                                                          + "-test_" + test_subtask_lang),
                                                                    verbose=args.verbose,
                                                                    prior_mbert=prior_mbert[test_idx],
                                                                    prior_intents=prior_intents[test_idx],
                                                                    prior_slots=prior_slots[test_idx],
                                                                    prior_adapter_norm_before=prior_adapter_norm_before[test_idx],
                                                                    prior_adapter_down_1=prior_adapter_down_1[test_idx],
                                                                    prior_adapter_up_1=prior_adapter_up_1[test_idx],
                                                                    prior_adapter_norm_after_1=prior_adapter_norm_after_1[test_idx],
                                                                    prior_adapter_feed_layer_1=prior_adapter_feed_layer_1[test_idx],
                                                                    prior_adapter_feed_layer_2=prior_adapter_feed_layer_2[test_idx],
                                                                    prior_adapter_down_2=prior_adapter_down_2[test_idx],
                                                                    prior_adapter_up_2=prior_adapter_up_2[test_idx],
                                                                    prior_adapter_norm_after_2=prior_adapter_norm_after_2[test_idx])

            with open(os.path.join(metrics_dir,
                                   "epoch_"+str(epoch)+"_metrics_"+str(train_idx)+".pickle"), "wb") \
                    as output_file:
                pickle.dump(metrics, output_file)

    return params, saved_grads, best_model, optimizer


def test_at_end_training(best_model,
                         dataset,
                         test_stream,
                         train_idx,
                         train_lang,
                         num_steps,
                         writer,
                         results_dir,
                         metrics_dir,
                         prior_mbert,
                         prior_intents,
                         prior_slots,
                         prior_adapter_norm_before,
                         prior_adapter_down_1,
                         prior_adapter_up_1,
                         prior_adapter_norm_after_1,
                         prior_adapter_feed_layer_1,
                         prior_adapter_feed_layer_2,
                         prior_adapter_down_2,
                         prior_adapter_up_2,
                         prior_adapter_norm_after_2):

    app_log.info("------------------------------------ TESTING At the end of the training")
    metrics = {task: {} for task in test_stream} # could be either per subtask or language
    for test_idx, test_subtask_lang in enumerate(test_stream):
        if test_stream[test_subtask_lang]['size'] > 0:
            metrics[test_subtask_lang], _ = evaluate_report(dataset,
                                                            test_stream[test_subtask_lang],
                                                            best_model,
                                                            train_lang,
                                                            train_idx,
                                                            test_subtask_lang,
                                                            test_idx,
                                                            num_steps,
                                                            writer,
                                                            name="test",
                                                            out_path=os.path.join(results_dir,
                                                                                  "End_test_perf-train_"+train_lang
                                                                                  + "-test_" + test_subtask_lang),
                                                            verbose=args.verbose,
                                                            prior_mbert=prior_mbert[test_idx],
                                                            prior_intents=prior_intents[test_idx],
                                                            prior_slots=prior_slots[test_idx],
                                                            prior_adapter_norm_before=prior_adapter_norm_before[test_idx],
                                                            prior_adapter_down_1=prior_adapter_down_1[test_idx],
                                                            prior_adapter_up_1=prior_adapter_up_1[test_idx],
                                                            prior_adapter_norm_after_1=prior_adapter_norm_after_1[test_idx],
                                                            prior_adapter_feed_layer_1=prior_adapter_feed_layer_1[test_idx],
                                                            prior_adapter_feed_layer_2=prior_adapter_feed_layer_2[test_idx],
                                                            prior_adapter_down_2=prior_adapter_down_2[test_idx],
                                                            prior_adapter_up_2=prior_adapter_up_2[test_idx],
                                                            prior_adapter_norm_after_2=prior_adapter_norm_after_2[test_idx])

    with open(os.path.join(metrics_dir, "final_metrics_"+str(train_idx)+".pickle"), "wb") as output_file:
        pickle.dump(metrics, output_file)


def run(results_dir, args, app_log):
    """
    The main of training over different streams and evaluating different approaches in terms of catastrophic forgetting
    and generalizability to new classes/languages
    :param args:
    :return:
    """
    writer = SummaryWriter(os.path.join(results_dir, 'runs'))

    metrics_dir = os.path.join(results_dir, "metrics")
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir)

    checkpoint_dir = os.path.join(results_dir, "checkpoint")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    args_save_file = os.path.join(checkpoint_dir, "training_args.bin")
    model_save_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    optim_save_file = os.path.join(checkpoint_dir, "optimizer.pt")

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

    dataset = Dataset(args.data_root,
                      args.setup_opt,
                      args.setup_cillia,
                      args.multi_head_out,
                      tokenizer,
                      args.data_format,
                      args.use_slots,
                      args.seed,
                      args.languages,
                      args.order_class,
                      args.order_lang,
                      args.order_str,
                      args.num_intent_tasks,
                      args.num_lang_tasks,
                      intent_types=intent_types,
                      slot_types=slot_types)

    eff_num_intent = len(dataset.intent_types)
    num_intents = len(dataset.intent_types)
    eff_num_slot = len(dataset.slot_types)

    if args.setup_opt in ["cll", "multi-incr-cll"]:
        train_stream = dataset.train_stream
        dev_stream = dataset.dev_stream
        test_stream = dataset.test_stream
    else:
        train_stream = dataset.train_stream[args.cil_stream_lang]
        dev_stream = dataset.dev_stream[args.cil_stream_lang]
        test_stream = dataset.test_stream[args.cil_stream_lang]

    """ eff_num_intent/eff_num_slot """
    if args.setup_opt == "cil-other":
        eff_num_intent += 1
        num_intents += 1

    if args.setup_opt == "multi":
        num_tasks = 1
        eff_num_intents_task = eff_num_intent
    elif args.setup_opt in ["multi-incr", "cll"]:
        if len(args.order_str) > 0:
            num_tasks = len(args.order_str)
        else:
            num_tasks = len(args.languages)
        eff_num_intents_task = eff_num_intent
    else:  # args.setup_opt in ["cil", "cil-other", "cil-ll"]:
        range_intents = range(0, len(dataset.intent_types), args.num_intent_tasks)
        num_tasks = len(list(range_intents))
        eff_num_intents_task = [len(range(i, i+args.num_intent_tasks))
                                if i+args.num_intent_tasks < len(dataset.intent_types)
                                else len(range(i, len(dataset.intent_types))) for i in range_intents]

    model = TransNLUCRF(args=args,
                        trans_model=model_trans,
                        num_tasks=num_tasks,
                        num_intents=num_intents,
                        eff_num_intents_task=eff_num_intents_task,
                        device=device,
                        num_slots=eff_num_slot)

    if torch.cuda.device_count() > 1:
        app_log.info("torch.cuda.device_count(): %d", torch.cuda.device_count())
        gpus_list = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=gpus_list, dim=0)

    model.cuda()

    if args.random_pred:
        metrics = {lang: {} for lang in dataset.test_stream}
        for test_idx, test_lang in enumerate(dataset.test_stream):
            metrics[test_lang], _ = evaluate_report(dataset,
                                                    dataset.test_stream[test_lang],
                                                    model,
                                                    test_lang,
                                                    test_idx,
                                                    test_lang,
                                                    test_idx,
                                                    0,
                                                    writer,
                                                    name="init",
                                                    out_path=os.path.join(results_dir, "initial_perf.txt"),
                                                    verbose=args.verbose)

        with open(os.path.join(metrics_dir, "initial_metrics.pickle"), "wb") as output_file:
            pickle.dump(metrics, output_file)

    optimizer, scheduler = set_optimizer(model)

    grad_dims = []
    for n, p in model.named_parameters():
        if name_in_list(args.cont_comp, n):
            grad_dims.append(p.data.numel())

    ## Continuous Learning Algorithms
    if args.cont_learn_alg == "ewc":
        cont_learn_alg = EWC(device,
                             checkpoint_dir,
                             args.use_online,
                             args.gamma_ewc,
                             args.cont_comp)

    elif args.cont_learn_alg == "gem":
        cont_learn_alg = GEM(dataset,
                             args.use_slots,
                             args.use_a_gem,
                             args.a_gem_n,
                             checkpoint_dir,
                             args.cont_comp)
    else:
        cont_learn_alg = None

    prior_mbert = [None for _ in train_stream]
    prior_intents = [None for _ in train_stream]
    prior_slots = [None for _ in train_stream]
    prior_adapter_norm_before = [None for _ in train_stream]

    prior_adapter_down_1 = [None for _ in train_stream]
    prior_adapter_up_1 = [None for _ in train_stream]

    prior_adapter_norm_after_1 = [None for _ in train_stream]

    prior_adapter_feed_layer_1 = [None for _ in train_stream]
    prior_adapter_feed_layer_2 = [None for _ in train_stream]

    prior_adapter_down_2 = [None for _ in train_stream]
    prior_adapter_up_2 = [None for _ in train_stream]

    prior_adapter_norm_after_2 = [None for _ in train_stream]

    if args.multi_head_in:
        if args.emb_enc_subtask_spec == ["all"]:
            prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root,
                                                                                            model_name)).
                            named_parameters()} for _ in train_stream]
        else:
            prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root,
                                                                                            model_name)).
                           named_parameters() if name_in_list(args.emb_enc_subtask_spec, k)}
                           for _ in train_stream]

    if args.use_adapters:
        adapter_size = model_trans.config.hidden_size
        prior_adapter_norm_before = [{k: v for k, v in nn.LayerNorm(adapter_size).named_parameters()}
                                     for _ in train_stream]

        prior_adapter_down_1 = [{k: v for k, v in nn.Linear(adapter_size, adapter_size//2).named_parameters()}
                                for _ in train_stream]

        prior_adapter_up_1 = [{k: v for k, v in nn.Linear(adapter_size//2, adapter_size).named_parameters()}
                            for _ in train_stream]

        if args.adapter_type == "houslby":
            prior_adapter_norm_after_1 = [{k: v for k, v in nn.LayerNorm(adapter_size).named_parameters()}
                                          for _ in train_stream]

            prior_adapter_feed_layer_1 = [{k: v for k, v in nn.Linear(adapter_size, adapter_size).named_parameters()}
                                          for _ in train_stream]

            prior_adapter_feed_layer_2 = [{k: v for k, v in nn.Linear(adapter_size, adapter_size).named_parameters()}
                                          for _ in train_stream]

            prior_adapter_down_2 = [{k: v for k, v in nn.Linear(adapter_size, adapter_size//2).named_parameters()}
                                    for _ in train_stream]

            prior_adapter_up_2 = [{k: v for k, v in nn.Linear(adapter_size//2, adapter_size).named_parameters()}
                                  for _ in train_stream]

            prior_adapter_norm_after_2 = [{k: v for k, v in nn.LayerNorm(adapter_size).named_parameters()}
                                          for _ in train_stream]

    if args.multi_head_out:
        # TODO change to accommodate different numbers of intents
        if "cil" in args.setup_opt:
            prior_intents = [{k: v for k, v in nn.Linear(model.trans_model.config.hidden_size, eff_num_intents_task[i]).
                             named_parameters()} for i in range(len(train_stream))]
        else:
            prior_intents = [{k: v for k, v in nn.Linear(model.trans_model.config.hidden_size, num_intents).
                             named_parameters()} for _ in range(len(train_stream))]

        prior_slots = [{k: v for k, v in nn.Linear(model.trans_model.config.hidden_size, eff_num_slot).
                        named_parameters()} for _ in train_stream]

    if args.setup_opt in ["cll", "multi-incr-cll", "cil", "cil-other", "multi-incr-cil"]:
        """ Continuous Learning Scenarios """

        sample_sizes = []
        best_saved_grads = None

        subtask_size = {i: 0 for i in range(len(train_stream))}

        best_model = None

        # Used for comparison only and to reinitialize trans_model but without using it as a reference
        original_mbert = model_trans_alias.from_pretrained(os.path.join(args.model_root, model_name))
        original_intent = copy.deepcopy(model.intent_classifier)
        original_slot = copy.deepcopy(model.slot_classifier)
        if args.use_adapters:
            original_ada_norm_before = copy.deepcopy(model.adapter.adapter_norm_before)
            original_ada_down_1 = copy.deepcopy(model.adapter.adapter_down_1[1])
            original_ada_up_1 = copy.deepcopy(model.adapter.adapter_up_1)

            if args.adapter_type == "houlsby":
                original_ada_norm_after_1 = copy.deepcopy(model.adapter.adapter_norm_after_1)
                original_feed_layer_1 = copy.deepcopy(model.adapter.feed_layer_1)
                original_feed_layer_2 = copy.deepcopy(model.adapter.feed_layer_2)

                original_ada_down_2 = copy.deepcopy(model.adapter.adapter_down_2[1])
                original_ada_up_2 = copy.deepcopy(model.adapter.adapter_up_2)
                original_ada_norm_after_2 = copy.deepcopy(model.adapter.adapter_norm_after_2)
        else:
            original_ada_norm_before = None
            original_ada_down_1 = None
            original_ada_up_1 = None

            original_ada_norm_after_1 = None
            original_feed_layer_1 = None
            original_feed_layer_2 = None

            original_ada_down_2 = None
            original_ada_up_2 = None
            original_ada_norm_after_2 = None

        mean_all_stream = []
        sum_all_stream = []

        for train_idx, train_subtask_lang in enumerate(train_stream):
            if train_subtask_lang["size"] == 0:
                app_log.warning("Skipped subtask/language: %d", train_idx)
                continue
            sample_sizes.append(int(train_subtask_lang["size"]*args.old_task_prop))
            num_steps = 0
            num_iter = train_subtask_lang["size"]//args.batch_size
            train_lang = train_subtask_lang["lang"]
            train_examples = train_subtask_lang["examples"]

            if args.freeze_first:
                start_freeze = 0
            else:
                start_freeze = 1

            if train_idx > start_freeze:
                for n, p in model.named_parameters():
                    if args.freeze_trans:
                        if p.requires_grad and 'trans_model' in n:
                            p.requires_grad = False

                    if args.freeze_linear:
                        if p.requires_grad and 'intent_classifier' in n:
                            p.requires_grad = False

                        if p.requires_grad and 'slot_classifier' in n:
                            p.requires_grad = False

            subtask_size[train_idx] = train_subtask_lang["size"]

            """ 1. Reinitialize linear layers for each new language """
            if args.multi_head_in or args.multi_head_out or args.use_adapters:
                model_dict = model.state_dict()

            if args.multi_head_in:
                if args.emb_enc_subtask_spec == ["all"]:  # could either be lang or subtask specific
                    trans_model_dict = {"trans_model."+k: v for k, v in original_mbert.named_parameters()}
                else:
                    trans_model_dict = {"trans_model."+k: v for k, v in original_mbert.named_parameters()
                                        if name_in_list(args.emb_enc_subtask_spec, k)}

                model_dict.update(trans_model_dict)

            if args.multi_head_out:
                if "cil" in args.setup_opt:
                    intent_classifier_dict = {"intent_classifier."+str(train_idx)+"."+k: v for k, v in original_intent[train_idx].named_parameters()}
                else:
                    intent_classifier_dict = {"intent_classifier."+k: v for k, v in original_intent.named_parameters()}
                model_dict.update(intent_classifier_dict)

                slot_classifier_dict = {"slot_classifier."+k: v for k, v in original_slot.named_parameters()}
                model_dict.update(slot_classifier_dict)

            if args.use_adapters:
                adapter_norm_before_dict = {"adapter.adapter_norm_before."+k: v for k, v in
                                            original_ada_norm_before.named_parameters()}
                model_dict.update(adapter_norm_before_dict)

                adapter_down_1_dict = {"adapter.adapter_down_1.1."+k: v for k, v in
                                       original_ada_down_1.named_parameters()}
                model_dict.update(adapter_down_1_dict)

                adapter_up_1_dict = {"adapter.adapter_up_1."+k: v for k, v in original_ada_up_1.named_parameters()}
                model_dict.update(adapter_up_1_dict)

                if args.adapter_type == "houlsby":

                    ##
                    adapter_norm_after_after_1_dict = {"adapter.adapter_norm_after_1."+k: v for k, v in
                                                       original_ada_norm_after_1.named_parameters()}
                    model_dict.update(adapter_norm_after_after_1_dict)

                    ##
                    adapter_feed_layer_1_dict = {"adapter.feed_layer_1."+k: v for k, v in
                                                 original_feed_layer_1.named_parameters()}
                    model_dict.update(adapter_feed_layer_1_dict)

                    ##
                    adapter_feed_layer_2_dict = {"adapter.feed_layer_2."+k: v for k, v in
                                                 original_feed_layer_2.named_parameters()}
                    model_dict.update(adapter_feed_layer_2_dict)

                    ##
                    adapter_down_2_dict = {"adapter.adapter_down_2.0."+k: v for k, v in
                                           original_ada_down_2.named_parameters()}
                    model_dict.update(adapter_down_2_dict)

                    ##
                    adapter_up_2_dict = {"adapter.adapter_up_2."+k: v for k, v in
                                           original_ada_up_2.named_parameters()}
                    model_dict.update(adapter_up_2_dict)

                    ##

                    adapter_norm_after_after_2_dict = {"adapter.adapter_norm_after_2."+k: v for k, v in
                                                       original_ada_norm_after_2.named_parameters()}
                    model_dict.update(adapter_norm_after_after_2_dict)

            if args.multi_head_in or args.multi_head_out or args.use_adapters:
                model.load_state_dict(model_dict)

            params, saved_grads, best_model, optimizer = train_task_epochs(model,
                                                                           optimizer,
                                                                           grad_dims,
                                                                           cont_learn_alg,
                                                                           dataset,
                                                                           train_examples,
                                                                           dev_stream,
                                                                           test_stream,
                                                                           subtask_size,
                                                                           sample_sizes,
                                                                           num_iter,
                                                                           train_idx,
                                                                           train_lang, # training name of the task # TODO think about it
                                                                           num_steps,
                                                                           writer,  # saving options
                                                                           results_dir,
                                                                           metrics_dir,
                                                                           checkpoint_dir,
                                                                           args_save_file,
                                                                           model_save_file,
                                                                           optim_save_file,
                                                                           prior_mbert,  # prior options
                                                                           prior_intents,
                                                                           prior_slots,
                                                                           prior_adapter_norm_before,
                                                                           prior_adapter_down_1,
                                                                           prior_adapter_up_1,
                                                                           prior_adapter_norm_after_1,
                                                                           prior_adapter_feed_layer_1,
                                                                           prior_adapter_feed_layer_2,
                                                                           prior_adapter_down_2,
                                                                           prior_adapter_up_2,
                                                                           prior_adapter_norm_after_2)

            """ 2. Saving the trained weights of MBERT in each language to be used later on in testing stage"""

            if args.multi_head_in:
                mbert_task = copy.deepcopy(model.trans_model)
                prior_mbert[train_idx] = {k: v for k, v in mbert_task.named_parameters()
                                          if name_in_list(args.emb_enc_subtask_spec, k)}

            if args.multi_head_out:
                # TODO DOUBLE CHECK model.intent_classifier[train_idx]
                if "cil" in args.setup_opt:
                    intents_task = copy.deepcopy(model.intent_classifier[train_idx])
                else:
                    intents_task = copy.deepcopy(model.intent_classifier)
                prior_intents[train_idx] = {k: v for k, v in intents_task.named_parameters()}

                slots_task = copy.deepcopy(model.slot_classifier)
                prior_slots[train_idx] = {k: v for k, v in slots_task.named_parameters()}

            if args.use_adapters:
                adapter_norm_before = copy.deepcopy(model.adapter.adapter_norm_before)
                prior_adapter_norm_before[train_idx] = {k: v for k, v in adapter_norm_before.named_parameters()}

                adapter_down_1 = copy.deepcopy(model.adapter.adapter_down_1[1])
                prior_adapter_down_1[train_idx] = {k: v for k, v in adapter_down_1.named_parameters()}

                adapter_up = copy.deepcopy(model.adapter.adapter_up_1)
                prior_adapter_up_1[train_idx] = {k: v for k, v in adapter_up.named_parameters()}

                if args.adapter_type == "houlsby":
                    ##
                    adapter_norm_after_1 = copy.deepcopy(model.adapter.adapter_norm_after_1)
                    prior_adapter_norm_after_1[train_idx] = {k: v for k, v in adapter_norm_after_1.named_parameters()}

                    ##
                    adapter_feed_layer_1 = copy.deepcopy(model.adapter.feed_layer_1)
                    prior_adapter_feed_layer_1[train_idx] = {k: v for k, v in adapter_feed_layer_1.named_parameters()}

                    ##
                    adapter_feed_layer_2 = copy.deepcopy(model.adapter.feed_layer_2)
                    prior_adapter_feed_layer_2[train_idx] = {k: v for k, v in adapter_feed_layer_2.named_parameters()}

                    ##
                    adapter_down_2 = copy.deepcopy(model.adapter.adapter_down_2[0])
                    prior_adapter_down_2[train_idx] = {k: v for k, v in adapter_down_2.named_parameters()}

                    ##
                    adapter_up_2 = copy.deepcopy(model.adapter.adapter_up_2)
                    prior_adapter_up_2[train_idx] = {k: v for k, v in adapter_up_2.named_parameters()}

                    ##
                    adapter_norm_after_2 = copy.deepcopy(model.adapter.adapter_norm_after_2)
                    prior_adapter_norm_after_2[train_idx] = {k: v for k, v in adapter_norm_after_2.named_parameters()}

            if args.save_change_params:
                app_log.info("After training on language: %d", train_idx)
                mean_all, sum_all = compute_change(mbert_task, original_mbert,
                                                   intents_task, original_intent,
                                                   slots_task, original_slot)
                mean_all_stream.append(mean_all)
                sum_all_stream.append(sum_all)

            app_log.info("Saving the best model at the end of the training stream for that language ....")
            params_save_file = os.path.join(checkpoint_dir, "pytorch_params_"+str(train_idx))
            grads_save_file = os.path.join(checkpoint_dir, "pytorch_grads_"+str(train_idx))

            if args.cont_learn_alg != "gem":
                if params:
                    with open(params_save_file, "wb") as file:
                        pickle.dump(params, file)

                if saved_grads:
                    with open(grads_save_file, "wb") as file:
                        pickle.dump(saved_grads, file)

            test_at_end_training(best_model,
                                 dataset,
                                 test_stream,
                                 train_idx,
                                 train_lang,
                                 num_steps,
                                 writer,
                                 results_dir,
                                 metrics_dir,
                                 prior_mbert,
                                 prior_intents,
                                 prior_slots,
                                 prior_adapter_norm_before,
                                 prior_adapter_down_1,
                                 prior_adapter_up_1,
                                 prior_adapter_norm_after_1,
                                 prior_adapter_feed_layer_1,
                                 prior_adapter_feed_layer_2,
                                 prior_adapter_down_2,
                                 prior_adapter_up_2,
                                 prior_adapter_norm_after_2)

        with open(os.path.join(metrics_dir, "mean_all_stream_mbert.pickle"), "wb") as output_file:
            pickle.dump(mean_all_stream, output_file)

    elif args.setup_opt == "cil-ll":
        """
        Setup 4: Cross-CIL-LL: "Cross-lingual combinations of languages/intents"
        - Stream consisting of different combinations 
        """
        _lambda = 20
        sample_sizes = []
        for i, subtask in enumerate(dataset.train_stream):
            num_steps = 0
            # Iterating over the stream of languages and intents
            sample_sizes.append(subtask["size"] // 10)

            if subtask["size"] == 0:
                app_log.warning("Skipped task: %d", i)
                continue

            for epoch in tqdm(range(args.epochs)):
                gc.collect()
                num_steps += 1
                num_iter = subtask["size"] // args.batch_size
                lang = subtask["lang"]
                stream = subtask["examples"]

                for j in range(num_iter):
                    if args.use_slots:
                        intent_loss, slot_loss = train(args,
                                                       optimizer,
                                                       model,
                                                       dataset,
                                                       stream,
                                                       writer,
                                                       epoch,
                                                       i,
                                                       j,
                                                       old_dataset=dataset.train_stream, # all dataset to get old stream
                                                       sample_sizes=sample_sizes,
                                                       _lambda=_lambda)
                    else:
                        intent_loss = train(args,
                                            optimizer,
                                            model,
                                            dataset,
                                            stream,
                                            writer,
                                            epoch,
                                            i,
                                            j,
                                            old_dataset=dataset.train_stream, # all dataset to get old stream
                                            sample_sizes=sample_sizes,
                                            _lambda=_lambda)
                    if args.use_slots:
                        app_log.info('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                           intent_loss.mean(),
                                                                                           slot_loss.mean()))
                    else:
                        app_log.info('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

                    if j % args.test_steps == 0:
                        for lang in dataset.test_stream[i]:
                            for k in range(0, i+1):
                                if dataset.test_stream[k][lang]["size"] > 0:
                                    evaluate_report(dataset.test_stream[k][lang],
                                                    k,
                                                    lang,
                                                    lang,
                                                    args,
                                                    dataset,
                                                    model,
                                                    writer,
                                                    i,
                                                    num_steps,
                                                    out_path=os.path.join(results_dir,
                                                                          "Test_perf-Epoch_"+epoch+"-train_"+str(i)+"-test_"+str(k)),
                                                    verbose=args.verbose)
                                else:
                                    app_log.warning("Skipped task: %d", k, " in lang: %s", lang)

            for lang in dataset.test_stream[i]:
                    for k in range(0, i+1):
                        if dataset.test_stream[k][lang]["size"] > 0:
                            metrics = evaluate_report(dataset.test_stream[k][lang],
                                                      k,
                                                      lang,
                                                      lang,
                                                      args,
                                                      dataset,
                                                      model,
                                                      writer,
                                                      i,
                                                      num_steps,
                                                      out_path=os.path.join(results_dir,
                                                                            "End-Test_perf-train_"+str(i)+"-test_"+str(k)),
                                                      verbose=args.verbose)
                        else:
                            app_log.warning("Skipped task: %d", k, " in lang: %s", lang)

            with open(os.path.join(metrics_dir, "final_metrics_"+str(i)+".pickle"), "wb") as output_file:
                pickle.dump(metrics, output_file)

    elif args.setup_opt == "multi":
        """
        Setup 5: Multi-task/Joint Learning: train on all languages and intent classes at the same time 
        """

        # There is only one task here no subtasks
        train_examples = dataset.train_stream["examples"]
        dev_stream = dataset.dev_stream
        test_stream = dataset.test_stream
        subtask_size = {0: dataset.train_stream["size"]}
        num_iter = dataset.train_stream["size"]//args.batch_size
        train_idx = 0  # only one task here
        sample_sizes = [int(dataset.train_stream["size"]*args.old_task_prop)] # not needed anyways because no continuous learning here
        train_lang = "-".join(args.languages)  # all languages
        num_steps = 0

        params, saved_grads, best_model, optimizer = train_task_epochs(model,
                                                                       optimizer,
                                                                       grad_dims,
                                                                       cont_learn_alg,
                                                                       dataset,
                                                                       train_examples,
                                                                       dev_stream,
                                                                       test_stream,
                                                                       subtask_size,
                                                                       sample_sizes,
                                                                       num_iter,
                                                                       train_idx,
                                                                       train_lang,
                                                                       num_steps,
                                                                       writer,  # saving options
                                                                       results_dir,
                                                                       metrics_dir,
                                                                       checkpoint_dir,
                                                                       args_save_file,
                                                                       model_save_file,
                                                                       optim_save_file,
                                                                       prior_mbert,  # prior options
                                                                       prior_intents,
                                                                       prior_slots,
                                                                       prior_adapter_norm_before,
                                                                       prior_adapter_down_1,
                                                                       prior_adapter_up_1,
                                                                       prior_adapter_norm_after_1,
                                                                       prior_adapter_feed_layer_1,
                                                                       prior_adapter_feed_layer_2,
                                                                       prior_adapter_down_2,
                                                                       prior_adapter_up_2,
                                                                       prior_adapter_norm_after_2)

        test_at_end_training(best_model,
                             dataset,
                             test_stream,
                             train_idx,
                             train_lang,
                             num_steps,
                             writer,
                             results_dir,
                             metrics_dir,
                             prior_mbert,
                             prior_intents,
                             prior_slots,
                             prior_adapter_norm_before,
                             prior_adapter_down_1,
                             prior_adapter_up_1,
                             prior_adapter_norm_after_1,
                             prior_adapter_feed_layer_1,
                             prior_adapter_feed_layer_2,
                             prior_adapter_down_2,
                             prior_adapter_up_2,
                             prior_adapter_norm_after_2)


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./main.py", description="Different options/arguments for running continuous"
                                                              "learning algorithms.")

    ## PATHS
    path_params = parser.add_argument_group("Path Parameters")
    path_params.add_argument("--data_root", help="Root directory of the dataset.",
                             type=str, default="")

    path_params.add_argument("--model_root", help="Path to the root directory hosting the trans model, if offline.",
                             type=str, default="")

    path_params.add_argument("--out_dir", help="The root directory of the results for this project.",
                             type=str, default="")

    path_params.add_argument("--stats_file", help="Filename of the stats file.",  # TODO CHECK WHAT THIS DOES EXACTLY
                             type=str, default="stats.txt")

    path_params.add_argument("--log_file", help="Filename of the log file.",  # TODO DO PROPER CHECKPOINTING
                             type=str, default="log.txt")

    ## SETUP OPTIONS
    setup_params = parser.add_argument_group("Setup Scenarios Parameters")
    setup_params.add_argument("--setup_opt", help="The different setup scenarios to pick from:"
                                                  "* cil:        Cross-CIL with fixed LL. "
                                                  "* cil-other:  Incremental version of cil where previous intents' "
                                                  "              subtasks are added in addition to other labels for"
                                                  "              subsequent intents'subtasks."
                                                  "* cll:        Cross-LL with fixed CIL."
                                                  "* cil-ll:     Cross CIL and CLL mixed."
                                                  "* multi-incr-cil: Weaker version of Multi-Task Learning, where we"
                                                  "                  gradually fine-tune on the accumulation of "
                                                  "                  different subtasks."
                                                  "* multi-incr-cll: Weaker version of Multilingual Learning, where we"
                                                  "                  gradually fine-tune on the accumulation of "
                                                  "                  different languages."
                                                  "* multi:      Multi-tasking one model on all tasks and languages.",

                              choices=["cil", "cil-other", "multi-incr-cil",
                                       "cll", "multi-incr-cll",
                                       "cil-ll",
                                       "multi"],
                              type=str, default="cll")

    setup_params.add_argument("--cil_stream_lang", help="Which lang to work on for the CIL setup if it is picked.",
                              default="en")

    setup_params.add_argument("--order_class", help="Different ways of ordering the classes:"
                                                    "* 0: high2lowclass: decreasing order (from high to low-resource)."
                                                    "* 1: low2highclass: increasing order (from low to high-resource)."
                                                    "* 2: randomclass: random order.",
                              type=int, default=0)

    setup_params.add_argument("--order_lang", help="Different ways of ordering the languages:"
                                                   "* 0: high2lowlang: decreasing order (from high to low-resource)."
                                                   "* 1: low2highlang: increasing order (from low to high-resource)."
                                                   "* 2: randomlang: random order.",
                              type=int, default=0)

    setup_params.add_argument("--order_str", help="Specific order for subtasks and languages: list of languages "
                                                  "or subtasks.",
                              nargs='+', default=[],
                              choices=[])

    setup_params.add_argument("--setup_cillia", help="Different ways of ordering mixture of both cll and cil:"
                                                     "* intents: traversing subtasks horizontally over all intent "
                                                     "           classes first then to languages."
                                                     "* langs: traversing subtasks vertically over all languages first"
                                                     "         then to classes.",
                              type=str, default="intents")

    setup_params.add_argument('--random_pred', help="Whether to predict directly the random initialization of the model"
                                                    "when tested directly on the languages without any fine-tuning.",
                              action="store_true")


    ## DATASET OPTIONS
    dataset_params = parser.add_argument_group("Dataset Options")
    dataset_params.add_argument("--data_format", help="Whether it is tsv (MTOD), json, or txt (MTOP).",
                                type=str, default="txt")

    dataset_params.add_argument("--languages", help="Train languages list.",
                                nargs="+", default=["de", "en", "es", "fr", "hi", "th"])

    dataset_params.add_argument("--num_intent_tasks", help="The number of intent per task.",
                                type=int, default=10)

    dataset_params.add_argument("--num_lang_tasks", help="The number of lang per task.",
                                type=int, default=2)

    ## Checkpointing/logging options
    checkpointing_params = parser.add_argument_group("Checkpointing/logging Parameters")
    checkpointing_params.add_argument("--verbose", help="If true, return golden labels and predictions to console.",
                                      action="store_true")

    checkpointing_params.add_argument("--save_dev_pred", help="If true, save the dev predictions.",
                                      action="store_true")

    checkpointing_params.add_argument("--save_test_every_epoch", help="If true, save test at the end of each epoch.",
                                      action="store_true")

    checkpointing_params.add_argument("--save_change_params", help="If true, save test at the end of each epoch.",
                                      action="store_true")

    checkpointing_params.add_argument("--no_debug", help="If true, save training and testing logs to disk.",
                                      action="store_true")

    checkpointing_params.add_argument("--save_model", help="Whether to save the model after training.",
                                      action="store_true")

    ## BASE MODEL TRAINING OPTIONS
    base_model_params = parser.add_argument_group("Base Model Parameters")
    base_model_params.add_argument("--trans_model", help="Name of the Transformer encoder model.",
                                   type=str, default="BertBaseMultilingualCased",
                                   choices=["BertBaseMultilingualCased", "BertLarge", "BertBaseCased",
                                            "Xlnet_base", "Xlnet_large", "XLM", "DistilBert_base",
                                            "DistilBert_large", "Roberta_base", "Roberta_large",
                                            "XLMRoberta_base", "XLMRoberta_large", "ALBERT-base-v1",
                                            "ALBERT-large-v1", "ALBERT-xlarge-v1", "ALBERT-xxlarge-v1",
                                            "ALBERT-base-v2", "ALBERT-large-v2", "ALBERT-xlarge-v2",
                                            "ALBERT-xxlarge-v2"])

    base_model_params.add_argument("--use_slots", help="If true, optimize for slot filling loss too.",
                                   action="store_true")

    base_model_params.add_argument("--use_mono", help="Whether to train monolingually.",
                                   action="store_true")

    base_model_params.add_argument("--epochs", help="The total number of epochs.",
                                   type=int, default=10)

    base_model_params.add_argument("--dev_steps", help="The total number of epochs to evaluate the model on the dev.",
                                   type=int, default=200)  # TODO DEV IS EVALUATED ON ONLY AFTER EACH EPOCH

    base_model_params.add_argument("--test_steps", help="The total number of epochs to evaluate the model on the test.",
                                   type=int, default=200)  # TODO THIS IS NOT USED CONSISTENTLY

    base_model_params.add_argument("--batch_size", help="The total number of epochs for the model to evaluate.",
                                   type=int, default=32)

    base_model_params.add_argument("--adam_lr", help="The learning rate for Adam Optimizer.",
                                   type=float, default=1e-03)

    base_model_params.add_argument("--adam_eps", help="Epsilon for the Adam Optimizer.",
                                   type=float, default=1e-08)

    base_model_params.add_argument("--beta_1", help="Beta_1 for the Adam Optimizer.",
                                   type=float, default=0.9)

    base_model_params.add_argument("--beta_2", help="Beta_2 for the Adam Optimizer.",
                                   type=float, default=0.99)

    base_model_params.add_argument("--step_size", help="The step size for the scheduler.",
                                   type=int, default=7)

    base_model_params.add_argument("--gamma", help="Gamma for the scheduler.",
                                   type=float, default=0.1)

    base_model_params.add_argument("--seed", help="Random Seed.",
                                   type=int, default=42)

    ### FREEZING OPTIONS
    freezing_params = parser.add_argument_group("Freezing Options")
    freezing_params.add_argument("--freeze_trans", help="Whether to freeze all layers in Transformer encoder/embed.",
                                 action="store_true")

    freezing_params.add_argument("--freeze_first", help="Whether to freeze from the first subtask/language.",
                                 action="store_true")

    freezing_params.add_argument("--freeze_linear", help="Whether to freeze all task-specific layers.",
                                 action="store_true")


    ## MODEL EXPANSION OPTIONS
    model_expansion_params = parser.add_argument_group("Model Expansion Options")
    model_expansion_params.add_argument("--multi_head_in", help="Whether to use multiple heads "
                                                                "that would imply multiple subtask/language-specific "
                                                                "heads at the input level.",
                                        action="store_true")

    model_expansion_params.add_argument("--emb_enc_subtask_spec", help="Which layer in the embeddings or the encoder "
                                                                       "to tune for each subtask/language"
                                                                       " independently.",
                                        choices=["embeddings",
                                                 "encoder.layer.0.",
                                                 "encoder.layer.1.",
                                                 "encoder.layer.2.",
                                                 "encoder.layer.3.",
                                                 "encoder.layer.4.",
                                                 "encoder.layer.5.",
                                                 "encoder.layer.6.",
                                                 "encoder.layer.7.",
                                                 "encoder.layer.8.",
                                                 "encoder.layer.9.",
                                                 "encoder.layer.10.",
                                                 "encoder.layer.11.",
                                                 "pooler",
                                                 "all"],
                                        nargs="+", default=["embeddings"])

    model_expansion_params.add_argument("--multi_head_out", help="Whether to use multiple heads in the outputs that "
                                                                 "would imply the use of different task-specific "
                                                                 "layers.",
                                        action="store_true")

    model_expansion_params.add_argument("--use_adapters", help="whether to use adapters.",
                                        action="store_true")

    model_expansion_params.add_argument("--adapter_type", help="Which adapter to use.",
                                        type=str, default="MADX", choices=["Houlsby", "MADX"])

    model_expansion_params.add_argument("--adapter_layers", help="List of layers to which adapters are applied.",
                                        nargs="+", default="0 1 2 3 4 5")

    ## MODEL EXPANSION OPTIONS
    cont_learn_params = parser.add_argument_group("Continuous Learning Options")
    cont_learn_params.add_argument("--cont_learn_alg", help="vanilla fine-tuning or some continuous learning algorithm:"
                                                            "(ewc, gem, mbpa, metambpa, etc) or vanilla if no specific"
                                                            "continuous learning algorithm is used.",
                                   choices=["vanilla", "ewc", "gem"],  # TODO to be covered next "mbpa", "metambpa", "icarl", "xdg", "si", "lwf", "gr", "rtf", "er"
                                   type=str, default="vanilla")

    cont_learn_params.add_argument("--cont_comp", help="Which component(s) in the model to focus on while learning "
                                                       "during regularization or replay",
                                   nargs="+", default=["trans intent slot"])

    ### for ewc
    cont_learn_params.add_argument("--old_task_prop", help="The percentage of old tasks used in regularization "
                                                           "or replay.",
                                   type=float, default=1.0)

    cont_learn_params.add_argument("--ewc_lambda", help="If ewc: lambda for regularization in ewc.",
                                   type=int, default=20)

    cont_learn_params.add_argument("--use_online", help="If ewc: Whether to use the online version of EWC or not.",
                                   action='store_true')

    cont_learn_params.add_argument("--gamma_ewc", help="If ewc: The percentage of decay.",
                                   type=int, default=0.01)

    ### for gem
    cont_learn_params.add_argument("--use_a_gem", help="If gem: whether to use averaged gem.",
                                   action="store_true")

    cont_learn_params.add_argument("--a_gem_n", help="If gem: The number of examples in the averaged memory.",
                                   type=int, default=100)

    args = parser.parse_args()
    set_seed()

    results_dir = set_out_dir()

    app_log = logger(os.path.join(results_dir, args.log_file))
    app_log.info("Saving to results_dir %s", results_dir)

    if args.no_debug:
        stdoutOrigin = sys.stdout

        sys.stdout = open(os.path.join(results_dir, args.log_file), "w")
        logstats.init(os.path.join(results_dir, args.stats_file))
        config_path = os.path.join(results_dir, 'config.json')
        logstats.add_args('config', args)
        logstats.write_json(vars(args), config_path)

    run(results_dir, args, app_log)

    if args.no_debug:
        sys.stdout.close()
        sys.stdout = stdoutOrigin

