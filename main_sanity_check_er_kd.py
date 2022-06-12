# From the project
from data_utils import *
from parser_args import *
from transformers_config import MODELS_dict
from basemodels.transNLUCRF import TransNLUCRF
from consts import INTENT_TYPES, SLOT_TYPES
from contlearnalg.EWC_grads import EWC
# from contlearnalg.GEM import GEM
from contlearnalg.MbPA import MBPA
from contlearnalg.ER import ER
from contlearnalg.KD import KD
from utils import variable, format_store_grads, name_in_list, logger, import_from, evaluate_report, set_optimizer, \
    get_config_params
import os, psutil
from schedulers.StaticLeitnerSystem import *
from schedulers.MeanRecallApproximation import *

# Torch
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from transformers import set_seed

# Other Python Modules
import gc
import sys
import logstats
import importlib
import pickle
import numpy as np
import argparse
import copy
from copy import deepcopy
from tqdm import tqdm

# GPU/CPU
gpus_list = list(range(torch.cuda.device_count()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def test_one_batch(batch,
                   model,
                   train_idx,
                   test_idx,
                   memory,
                   cont_learn_alg,
                   dataset,
                   use_slots,
                   name="train",
                   prior_mbert=None,
                   prior_intents=None,
                   prior_slots=None,
                   prior_adapter=None):

    app_log.info("Evaluating on Training batch ...")

    if prior_mbert or prior_intents or prior_slots or prior_adapter:

        model_dict = model.state_dict()

        if prior_mbert:
            app_log.info("Using prior_mbert")
            ### 1. wanted keys, values are in trans_model
            trans_model_dict = {"trans_model." + k: v for k, v in prior_mbert.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(trans_model_dict)

        if prior_intents:
            app_log.info("Using prior_intents")
            ### 1. wanted keys, values are in trans_model
            if "cil" in args.setup_opt:
                intent_classifier_dict = {"intent_classifier." + str(test_idx) + "." + k: v for k, v in
                                          prior_intents.items()}
            else:
                intent_classifier_dict = {"intent_classifier." + k: v for k, v in prior_intents.items()}
            ### 2. overwrite entries in the existing state dict
            model_dict.update(intent_classifier_dict)

        if prior_slots:
            app_log.info("Using prior_slots")
            ### 1. wanted keys, values are in trans_model
            slot_classifier_dict = {"slot_classifier." + k: v for k, v in prior_slots.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(slot_classifier_dict)

        if prior_adapter:
            adapter_norm_before_dict = {"adapter." + k: v for k, v in prior_adapter.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_norm_before_dict)

        ### 3. load the new state dict
        model.load_state_dict(model_dict)

    intent_corrects = 0
    sents_text = []

    intents_true = []
    intents_pred = []

    slots_true = []
    slots_pred = []

    slots_true_all = []
    slots_pred_all = []


    input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts, input_identifiers = batch

    if device != torch.device("cpu"):
        input_ids = input_ids.cuda()
        lengths = lengths.cuda()
        input_masks = input_masks.cuda()
        intent_labels = intent_labels.cuda()
        slot_labels = slot_labels.cuda()

    if train_idx > 0 and name == "test":
        if args.cont_learn_alg == "mbpa":
            """ Local adaptation of MbPA """

            q = model.get_embeddings(input_ids, input_masks)[0]

            if args.use_reptile:
                if args.use_batches_reptile:
                    eval_model = cont_learn_alg.forward_reptile_many_batches(memory, q, train_idx, model, dataset)
                else:
                    eval_model = cont_learn_alg.forward_reptile_one_batch(memory, q, train_idx, model, dataset)
            else:
                eval_model = cont_learn_alg.forward(memory, q, train_idx, model,
                                                    dataset)  # this is taking into consideration only the task we are testing from assuming we know that task.
        else:
            eval_model = model
    else:
        eval_model = model

    eval_model.eval()
    if use_slots:
        with torch.no_grad():
            intent_logits, slot_logits, _, intent_loss, slot_loss, loss, pooled_output \
                = eval_model(input_ids=input_ids,
                             input_masks=input_masks,
                             train_idx=test_idx,
                             lengths=lengths,
                             intent_labels=intent_labels,
                             slot_labels=slot_labels)

        # Slot Golden Truth/Predictions
        true_slot = slot_labels[0]

        slot_logits = [slot_logits[j, :length].data.numpy() for j, length in enumerate(lengths)]
        pred_slot = list(slot_logits[0])

        true_slot_l = [dataset.slot_types[s] for s in true_slot]
        pred_slot_l = [dataset.slot_types[s] for s in pred_slot]

        true_slot_no_x = []
        pred_slot_no_x = []

        for j, slot in enumerate(true_slot_l):
            if slot != "X":
                if j < len(pred_slot_l):
                    true_slot_no_x.append(true_slot_l[j])
                    pred_slot_no_x.append(pred_slot_l[j])

        slots_true.append(true_slot_no_x)
        slots_pred.append(pred_slot_no_x)

        slots_true_all.extend(true_slot_no_x)
        slots_pred_all.extend(pred_slot_no_x)

    else:
        with torch.no_grad():
            intent_logits, intent_loss, loss, pooled_output = eval_model(input_ids=input_ids,
                                                                         input_masks=input_masks,
                                                                         train_idx=test_idx,
                                                                         lengths=lengths,
                                                                         intent_labels=intent_labels)

    # Intent Golden Truth/Predictions
    true_intents = intent_labels.tolist()
    pred_intents = intent_logits.max(1)[1]

    intent_outcomes = []
    for i in range(len(true_intents)):
        if true_intents[i] == pred_intents[i]:
            intent_outcomes.append(1)
        else:
            intent_outcomes.append(0)

    assert len(intent_outcomes) == len(input_identifiers)

    eval_outcomes = {input_identifiers[i]:intent_outcomes[i] for i in range(len(input_identifiers))}

    return eval_outcomes

def test_train_batches(dataset_train,
                       nb_examples,
                       model,
                       train_idx,
                       test_idx,
                       memory,
                       cont_learn_alg,
                       dataset,
                       use_slots,
                       name="train",
                       prior_mbert=None,
                       prior_intents=None,
                       prior_slots=None,
                       prior_adapter=None):

    app_log.info("Evaluating on Training batch ...")

    if prior_mbert or prior_intents or prior_slots or prior_adapter:

        model_dict = model.state_dict()

        if prior_mbert:
            app_log.info("Using prior_mbert")
            ### 1. wanted keys, values are in trans_model
            trans_model_dict = {"trans_model." + k: v for k, v in prior_mbert.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(trans_model_dict)

        if prior_intents:
            app_log.info("Using prior_intents")
            ### 1. wanted keys, values are in trans_model
            if "cil" in args.setup_opt:
                intent_classifier_dict = {"intent_classifier." + str(test_idx) + "." + k: v for k, v in
                                          prior_intents.items()}
            else:
                intent_classifier_dict = {"intent_classifier." + k: v for k, v in prior_intents.items()}
            ### 2. overwrite entries in the existing state dict
            model_dict.update(intent_classifier_dict)

        if prior_slots:
            app_log.info("Using prior_slots")
            ### 1. wanted keys, values are in trans_model
            slot_classifier_dict = {"slot_classifier." + k: v for k, v in prior_slots.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(slot_classifier_dict)

        if prior_adapter:
            adapter_norm_before_dict = {"adapter." + k: v for k, v in prior_adapter.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_norm_before_dict)

        ### 3. load the new state dict
        model.load_state_dict(model_dict)

    intent_corrects = 0
    sents_text = []

    intents_true = []
    intents_pred = []

    slots_true = []
    slots_pred = []

    slots_true_all = []
    slots_pred_all = []

    intent_outcomes = []
    input_identifiers_list = []

    eval_outcomes = {}
    for _ in tqdm(range(nb_examples)):
        batch_one, text \
            = dataset.next_batch(1, dataset_train)

        input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts, input_identifiers \
            = batch_one

        if device != torch.device("cpu"):
            input_ids = input_ids.cuda()
            lengths = lengths.cuda()
            input_masks = input_masks.cuda()
            intent_labels = intent_labels.cuda()
            slot_labels = slot_labels.cuda()

        if train_idx > 0 and name == "test":
            if args.cont_learn_alg == "mbpa":
                """ Local adaptation of MbPA """

                q = model.get_embeddings(input_ids, input_masks)[0]

                if args.use_reptile:
                    if args.use_batches_reptile:
                        eval_model = cont_learn_alg.forward_reptile_many_batches(memory, q, train_idx, model, dataset)
                    else:
                        eval_model = cont_learn_alg.forward_reptile_one_batch(memory, q, train_idx, model, dataset)
                else:
                    eval_model = cont_learn_alg.forward(memory, q, train_idx, model,
                                                        dataset)  # this is taking into consideration only the task we are testing from assuming we know that task.
            else:
                eval_model = model
        else:
            eval_model = model

        eval_model.eval()
        if use_slots:
            with torch.no_grad():
                intent_logits, slot_logits, _, intent_loss, slot_loss, loss, pooled_output \
                    = eval_model(input_ids=input_ids,
                                 input_masks=input_masks,
                                 train_idx=test_idx,
                                 lengths=lengths,
                                 intent_labels=intent_labels,
                                 slot_labels=slot_labels)

            # Slot Golden Truth/Predictions
            true_slot = slot_labels[0]

            slot_logits = [slot_logits[j, :length].data.numpy() for j, length in enumerate(lengths)]
            pred_slot = list(slot_logits[0])

            true_slot_l = [dataset.slot_types[s] for s in true_slot]
            pred_slot_l = [dataset.slot_types[s] for s in pred_slot]

            true_slot_no_x = []
            pred_slot_no_x = []

            for j, slot in enumerate(true_slot_l):
                if slot != "X":
                    if j < len(pred_slot_l):
                        true_slot_no_x.append(true_slot_l[j])
                        pred_slot_no_x.append(pred_slot_l[j])

            slots_true.append(true_slot_no_x)
            slots_pred.append(pred_slot_no_x)

            slots_true_all.extend(true_slot_no_x)
            slots_pred_all.extend(pred_slot_no_x)
        else:
            with torch.no_grad():
                intent_logits, intent_loss, loss, pooled_output = eval_model(input_ids=input_ids,
                                                                             input_masks=input_masks,
                                                                             train_idx=test_idx,
                                                                             lengths=lengths,
                                                                             intent_labels=intent_labels)

        # Intent Golden Truth/Predictions
        true_intent = intent_labels.squeeze().item()
        pred_intent = intent_logits.squeeze().max(0)[1]

        if true_intent == pred_intent:
            intent_outcome = 1
        else:
            intent_outcome = 0

        eval_outcomes.update({input_identifiers[0]: intent_outcome})

    return eval_outcomes


def train(batch_left_size,
          optimizer,
          intent_embeddings,
          model,
          grad_dims,
          cont_learn_alg,
          dataset,
          train_examples,
          memory,
          train_examples_size,
          writer,
          epoch,
          i_task,
          num_steps,
          checkpoint_dir,
          model_trans,
          num_tasks,
          num_intents,
          eff_num_intents_task,
          eff_num_slot,
          ltn_scheduler,
          eval_sched_freq,  # How frequently should we evaluate and populate the scheduler
          sample_sizes=[]):

    optimizer.zero_grad()
    model.train()

    # Take batch by batch and move to cuda
    batch, _ = dataset.next_batch(batch_left_size, train_examples)

    input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts, input_identifiers = batch

    # print("input_identifiers:", input_identifiers)
    if device != torch.device("cpu"):
        input_ids = input_ids.cuda()
        lengths = lengths.cuda()
        token_type_ids = token_type_ids.cuda()  # this is not needed at all
        input_masks = input_masks.cuda()  # TODO compare when you use this versus when you don't
        intent_labels = intent_labels.cuda()
        slot_labels = slot_labels.cuda()

    params = None
    saved_grads = None

    if args.use_slots:
        logits_intents, logits_slots, logits_slots_, intent_loss, slot_loss, loss, pooled_output \
            = model(input_ids=input_ids,
                    input_masks=input_masks,
                    train_idx=i_task,
                    lengths=lengths,
                    intent_labels=intent_labels,
                    slot_labels=slot_labels)

        writer.add_scalar('train_intent_loss_'+str(i_task), intent_loss.detach().mean(), num_steps*epoch)
        writer.add_scalar('train_slot_loss_'+str(i_task), slot_loss.detach().mean(), num_steps*epoch)
    else:
        logits_intents, intent_loss, loss, pooled_output = model(input_ids=input_ids,
                                                                 input_masks=input_masks,
                                                                 train_idx=i_task,
                                                                 lengths=lengths,
                                                                 intent_labels=intent_labels)

        writer.add_scalar('train_intent_loss_'+str(i_task), intent_loss.detach().mean(), num_steps*epoch)

    for i in range(batch_left_size):
        intent = intent_labels[i].squeeze().item()
        intent_embeddings[intent].append(pooled_output[i])

    if args.cont_learn_alg == "ewc":
        if i_task > 0:
            loss += (args.ewc_lambda / 2) * cont_learn_alg.penalty(i_task,
                                                                   model)

        loss = loss.mean()
        loss.backward()
        optimizer.step()

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
        optimizer.step()

        format_store_grads(pp=model.named_parameters(),
                           grad_dims=grad_dims,
                           cont_comp=args.cont_comp,
                           checkpoint_dir=checkpoint_dir,
                           tid=i_task,
                           store=True)  # storing for the current task

        print("grad_dims: ", grad_dims)
        if i_task > 0:
            cont_learn_alg.run(i_task,
                               sample_sizes,
                               model,
                               grad_dims)

        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        saved_grads = None

    elif args.cont_learn_alg == "er":
        # Backward pass over the MAIN training

        loss = loss.mean()
        loss.backward()
        optimizer.step()
        gc.collect()

        del logits_intents
        if args.use_slots:
            del logits_slots
            del logits_slots_
        torch.cuda.empty_cache()  # TODO try this emptying of cache after backward

        if i_task > 0:
            # memory training is interleaved with the main training so that the process doesn't overfit to the memory
            if num_steps % 100 == 0:
                for task_memory_id, task_memory in enumerate(memory):
                    # sample a batch from the memory of size sampling_k to go over
                    batch, _ = dataset.next_batch(args.sampling_k, task_memory)
                    optimizer.zero_grad()

                    input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts, \
                        input_identifiers = batch

                    if device != torch.device("cpu"):
                        input_ids = input_ids.cuda()
                        lengths = lengths.cuda()
                        token_type_ids = token_type_ids.cuda()  # this is not needed at all
                        input_masks = input_masks.cuda()  # TODO compare when you use this versus when you don't
                        intent_labels = intent_labels.cuda()
                        slot_labels = slot_labels.cuda()

                    if args.use_slots:
                        logits_intents, logits_slots, logits_slots_, er_intent_loss, er_slot_loss, er_loss, pooled_output \
                            = model(input_ids=input_ids,
                                    input_masks=input_masks,
                                    train_idx=i_task,
                                    lengths=lengths,
                                    intent_labels=intent_labels,
                                    slot_labels=slot_labels)

                        writer.add_scalar('er_intent_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                          er_intent_loss.detach().mean(), num_steps*epoch)
                        writer.add_scalar('er_slot_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                          er_slot_loss.detach().mean(), num_steps*epoch)
                    else:
                        logits_intents, er_intent_loss, er_loss, pooled_output = model(input_ids=input_ids,
                                                                                       input_masks=input_masks,
                                                                                       train_idx=i_task,
                                                                                       lengths=lengths,
                                                                                       intent_labels=intent_labels)

                        writer.add_scalar('er_intent_loss_'+str(i_task), er_intent_loss.detach().mean(), num_steps*epoch)

                    er_loss = er_loss.mean()
                    er_loss.backward()
                    optimizer.step()
                    gc.collect()

        params = None
        saved_grads = None

    elif "kd_old" in args.cont_learn_alg:  # Old KD
        if i_task > 0:
            # memory training is interleaved with the main training so that the process doesn't overfit to the memory
            if num_steps % 100 == 0:
                total_loss = loss.mean()
                for task_memory_id, task_memory in enumerate(memory):
                    # sample a batch from the memory of size sampling_k to go over
                    batch, _ = dataset.next_batch(args.sampling_k//len(memory), task_memory)
                    optimizer.zero_grad()

                    input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts, \
                        input_identifiers = batch

                    if device != torch.device("cpu"):
                        input_ids = input_ids.cuda()
                        lengths = lengths.cuda()
                        token_type_ids = token_type_ids.cuda()  # this is not needed at all
                        input_masks = input_masks.cuda()  # TODO compare when you use this versus when you don't
                        intent_labels = intent_labels.cuda()
                        slot_labels = slot_labels.cuda()

                    if args.use_slots:
                        kd_logits_intents, kd_logits_slots, kd_logits_slots_, kd_intent_loss, kd_slot_loss, kd_er_loss, \
                        kd_pooled_output = model(input_ids=input_ids,
                                                 input_masks=input_masks,
                                                 train_idx=i_task,
                                                 lengths=lengths,
                                                 intent_labels=intent_labels,
                                                 slot_labels=slot_labels)

                        writer.add_scalar('kd_slot_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                          kd_slot_loss.detach().mean(), num_steps*epoch)
                    else:
                        kd_logits_intents, kd_intent_loss, kd_er_loss, kd_pooled_output = model(input_ids=input_ids,
                                                                                                input_masks=input_masks,
                                                                                                train_idx=i_task,
                                                                                                lengths=lengths,
                                                                                                intent_labels=intent_labels)

                    writer.add_scalar('kd_intent_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                      kd_intent_loss.detach().mean(), num_steps*epoch)

                    writer.add_scalar('er_kd_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                      kd_er_loss.detach().mean(), num_steps*epoch)

                    min_num_sent = min(args.sampling_k//len(memory), args.batch_size)
                    if args.cont_learn_alg == "kd-logits":
                        kd_intent_loss = F.mse_loss(kd_logits_intents[:min_num_sent].to(device).float(),
                                                    logits_intents[:min_num_sent].to(device).float())

                        # kd_intent_loss = torch.nn.KLDivLoss(size_average=False)(main_intent_logits, memory_intent_logits)
                        min_seq_len = min(kd_logits_slots_.shape[1], logits_slots_.shape[1])

                        kd_slot_loss = F.mse_loss(kd_logits_slots_[:min_num_sent, :min_seq_len, :].to(device).float(),
                                                  logits_slots_[:min_num_sent, :min_seq_len, :].to(device).float())

                        # kd_slot_loss = torch.nn.KLDivLoss(size_average=False)(main_slot_logits, memory_slot_logits)
                        kd_loss = kd_intent_loss + kd_slot_loss

                    else:  # rep
                        kd_loss = F.mse_loss(kd_pooled_output[:min_num_sent].to(device).float(),
                                             pooled_output[:min_num_sent].to(device).float())

                    total_loss += 0.01 * kd_loss.mean()  # kd_loss.mean() # kd_er_loss.mean()
                    # print(num_steps, ' kd_loss_', i_task, '-mem_', str(task_memory_id), kd_loss.detach().mean(),
                    #       " current loss:", loss.detach().mean())
                    writer.add_scalar('kd_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                      kd_loss.detach().mean(), num_steps*epoch)

                # print("--------------- total_loss:", total_loss)
                writer.add_scalar('total_kd_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                  total_loss.detach(), num_steps*epoch)
                total_loss.backward()
                optimizer.step()

                if len(memory) == 0:
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
            else:
                loss = loss.mean()
                loss.backward()
                optimizer.step()
        else:
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        gc.collect()
        del logits_intents
        del pooled_output
        if args.use_slots:
            del logits_slots
            del logits_slots_
        torch.cuda.empty_cache()  # TODO try this emptying of cache after backward
    elif "kd" in args.cont_learn_alg:
        if i_task > 0:
            # memory training is interleaved with the main training so that the process doesn't overfit to the memory
            if num_steps % 100 == 0:
                total_loss = loss.mean()
                for task_memory_id, task_memory in enumerate(memory):
                    # Sample a batch from the memory of size sampling_k to go over
                    batch, _ = dataset.next_batch(args.sampling_k//len(memory), task_memory)
                    optimizer.zero_grad()

                    input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts,\
                        input_identifiers = batch

                    if device != torch.device("cpu"):
                        input_ids = input_ids.cuda()
                        lengths = lengths.cuda()
                        token_type_ids = token_type_ids.cuda()  # this is not needed at all
                        input_masks = input_masks.cuda()  # TODO compare when you use this versus when you don't
                        intent_labels = intent_labels.cuda()
                        slot_labels = slot_labels.cuda()

                    # Forward Pass on the batch using the current model
                    if args.use_slots:
                        kd_logits_intents, kd_logits_slots, kd_logits_slots_, kd_intent_loss, kd_slot_loss, kd_er_loss,\
                          kd_pooled_output = model(input_ids=input_ids,
                                                   input_masks=input_masks,
                                                   train_idx=i_task,
                                                   lengths=lengths,
                                                   intent_labels=intent_labels,
                                                   slot_labels=slot_labels)

                        writer.add_scalar('kd_slot_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                          kd_slot_loss.detach().mean(), num_steps*epoch)
                    else:
                        kd_logits_intents, kd_intent_loss, kd_er_loss, kd_pooled_output \
                            = model(input_ids=input_ids,
                                    input_masks=input_masks,
                                    train_idx=i_task,
                                    lengths=lengths,
                                    intent_labels=intent_labels)

                    writer.add_scalar('kd_intent_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                      kd_intent_loss.detach().mean(), num_steps*epoch)

                    writer.add_scalar('kd_er_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                      kd_er_loss.detach().mean(), num_steps*epoch)

                    # Forward Pass on the batch using the current model without gradients
                    # prev_model = TransNLUCRF(args=args,
                    #                          trans_model=model_trans,
                    #                          num_tasks=num_tasks,
                    #                          num_intents=num_intents,
                    #                          eff_num_intents_task=eff_num_intents_task,
                    #                          device=device,
                    #                          num_slots=eff_num_slot)

                    prev_model = torch.load(os.path.join(checkpoint_dir, "pytorch_params_"+str(task_memory_id)))


                    # prev_model = prev_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "pytorch_params_"+str(task_memory_id))))

                    # prev_model_dict = prev_model.state_dict()
                    # with open(os.path.join(checkpoint_dir, "pytorch_params_"+str(task_memory_id)), "rb") as file:
                    #     previous_weights = pickle.load(file)
                    # prev_model_dict.update(previous_weights)
                    # prev_model.load_state_dict(prev_model_dict)

                    with torch.no_grad():
                        if args.use_slots:
                            prev_logits_intents, prev_logits_slots, prev_logits_slots_, prev_intent_loss, \
                                prev_slot_loss, prev_er_loss, prev_pooled_output \
                                = prev_model(input_ids=input_ids,
                                             input_masks=input_masks,
                                             train_idx=i_task,
                                             lengths=lengths,
                                             intent_labels=intent_labels,
                                             slot_labels=slot_labels)
                        else:
                            prev_logits_intents, prev_intent_loss, prev_er_loss, prev_pooled_output \
                                = prev_model(input_ids=input_ids,
                                             input_masks=input_masks,
                                             train_idx=i_task,
                                             lengths=lengths,
                                             intent_labels=intent_labels)

                    if args.cont_learn_alg == "kd-logits":
                        # print("prev_logits_intents:", prev_logits_intents)
                        # prev_logits_intents = kd_logits_intents
                        # prev_logits_slots_ = kd_logits_slots_
                        kd_intent_loss = F.mse_loss(kd_logits_intents.to(device).float(),
                                                    prev_logits_intents.to(device).float())

                        kd_slot_loss = F.mse_loss(kd_logits_slots_.to(device).float(),
                                                  prev_logits_slots_.to(device).float())

                        # kd_slot_loss = torch.nn.KLDivLoss(size_average=False)(main_slot_logits, memory_slot_logits)
                        kd_loss = kd_intent_loss + kd_slot_loss

                    else:  # rep
                        # prev_pooled_output = kd_pooled_output
                        # print("prev_pooled_output:", prev_pooled_output.detach())
                        kd_loss = F.mse_loss(kd_pooled_output.to(device).float(),
                                             prev_pooled_output.detach().to(device).float())

                    # print("total_loss:", total_loss)
                    # print("kd_loss.mean():", kd_loss.mean())
                    total_loss += 0.01 * kd_loss.mean()  # kd_loss.mean() # kd_er_loss.mean()
                    # print(num_steps, ' kd_loss_', i_task, '-mem_', str(task_memory_id), kd_loss.detach().mean(),
                    #       " current loss:", loss.detach().mean())
                    writer.add_scalar('kd_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                      kd_loss.detach().mean(), num_steps*epoch)

                # print("--------------- total_loss:", total_loss)
                writer.add_scalar('total_kd_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                  total_loss.detach(), num_steps*epoch)
                # with torch.autograd.set_detect_anomaly(True):
                total_loss.backward()
                optimizer.step()

                if len(memory) == 0:
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
            else:
                loss = loss.mean()
                loss.backward()
                optimizer.step()
        else:
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        gc.collect()
        del logits_intents
        del pooled_output
        if args.use_slots:
            del logits_slots
            del logits_slots_
        torch.cuda.empty_cache()  # TODO try this emptying of cache after backward

    # elif args.cont_learn_alg == "reptile-er":
    #     # Hierarchical adaptation of Experience Replay
    #
    #     ## 1. Draw batches from the memory buffer to
    else:
        loss = loss.mean()
        loss.backward() #retain_graph=True
        optimizer.step()
        gc.collect()
        del logits_intents
        del pooled_output
        if args.use_slots:
            del logits_slots
            del logits_slots_
        torch.cuda.empty_cache()  # TODO try this emptying of cache after backward

        params = None
        saved_grads = None

    if args.use_leitner_queue:
        """ Eval on train here and populate/update the Leitner Queue accordingly for each element in the batch """
        if args.evaluate_one_batch:
            """ Eval only on one batch """
            eval_output = test_one_batch(batch,
                                         model,
                                         i_task,
                                         i_task,
                                         memory,
                                         cont_learn_alg,
                                         dataset,
                                         args.use_slots)

        else:
            """ Eval on the whole training data """
            eval_output = test_train_batches(train_examples,
                                             train_examples_size[i_task],
                                             model,
                                             i_task,
                                             i_task,
                                             memory,
                                             cont_learn_alg,
                                             dataset,
                                             args.use_slots)

        ##  eval to get the outcome for each item
        ## Generate a timestamp which is uniform for each item in the batch
        ltn_scheduler.place_items(eval_output)

        app_log.info(ltn_scheduler.rep_sched())


    if args.use_slots:
        return intent_loss, slot_loss, params, saved_grads, optimizer, model, intent_embeddings, ltn_scheduler

    return intent_loss, params, saved_grads, optimizer, model, intent_embeddings, ltn_scheduler


def set_out_dir():
    # out_dir -> setup_opt -> slots -> trans_model -> random_init/mono/lang_order -> class_order -> cont_learn_alg ->
    #  -> headed -> adapters -> freezing
    order_lang_dict = {0: "high2lowlang",
                       1: "low2highlang",
                       2: "randomlang"}

    order_class_dict = {0: "high2lowclass",
                        1: "low2highclass",
                        2: "randomclass"}

    if args.use_leitner_queue:
        new_out_dir = args.out_dir
        if args.demote_to_first_deck:
            new_out_dir = os.path.join(new_out_dir,
                                       "DemoteFirstDeck")
        else:
            new_out_dir = os.path.join(new_out_dir,
                                       "DemotePreviousDeck")
    else:
        new_out_dir = os.path.join(args.out_dir,
                                   "BASELINE")

    results_dir = os.path.join(new_out_dir,  # original output directory
                               args.setup_opt,  # setup option directory
                               (lambda x: "NLU" if x else "Intents_only")(args.use_slots),  # slot usage
                               args.trans_model,
                               "lr-"+str(args.adam_lr)+"_eps-"+str(args.adam_eps)+"_beta-"+str(args.beta_1)+"-"+str(args.beta_2))  # transformers model

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    if args.random_pred:
        results_dir = os.path.join(results_dir,
                                   "RANDOM",
                                   "SEED_" + str(args.seed),
                                   args.cil_stream_lang,
                                   "random_init")

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        return results_dir

    if args.use_mono:
        if args.setup_opt == "cil":
            results_dir = os.path.join(results_dir,
                                       "MONO",
                                       "SEED_" + str(args.seed),
                                       args.cil_stream_lang,
                                       str(args.mono_index))
        else:
            results_dir = os.path.join(results_dir,
                                       "MONO",
                                       "SEED_"+str(args.seed),
                                       args.languages[0])

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        return results_dir

    if args.setup_opt not in ["multi"]:
        # the order of languages and classes and continuous learning algorithms are only specific to non multi setups
        order_lang = args.order_lst if len(args.order_lst) > 0 else order_lang_dict[args.order_lang]

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

            cont_alg_option += "_memsz-"+str(args.max_mem_sz)+"_type-"+str(args.storing_type)\
                               +"_sample-"+str(args.sampling_type)+"_k-"+str(args.sampling_k)+ \
                               reptile_str

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
                        return "enc."+layer.split(".")[2]

                head_options = "multi_head_in"

                if args.multi_head_out:
                    head_options += "_out/"

                head_options += "-".join(list(map(map_emb_enc_subtask, args.emb_enc_subtask_spec.split("_"))))

            else:
                if args.multi_head_out:
                    head_options = "multi_head_out"

        order_class = order_class_dict[args.order_class]

        if "cil" in args.setup_opt or "cll-k-shots" in args.setup_opt or "cll-n-ways" in args.setup_opt:
            order_class = os.path.join(order_class, args.cil_stream_lang)

        results_dir = os.path.join(results_dir,
                                   order_lang,  # language order
                                   order_class, # class order
                                   cont_alg_option, # continuous learning algorithm
                                   head_options, # multi-headed option
                                   (lambda x: "adapters" if x else "no_adapters")(args.use_adapters))

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        if args.use_adapters:
            results_dir = os.path.join(results_dir,
                                       args.adapter_type,
                                       args.adapter_layers)# adapters option

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
                      lt_scheduler,
                      intent_embeddings,
                      optimizer,
                      grad_dims,
                      cont_learn_alg,
                      dataset,
                      train_examples,
                      memory,
                      dev_stream,
                      test_stream,
                      subtask_size,
                      sample_sizes,
                      num_iter,
                      train_idx,  # to be used for both the training language and subtask if cll this doesn't matter, if cil this helps
                      train_lang,
                      num_steps,
                      writer,  # saving options
                      predictions_dir,
                      metrics_dir,
                      checkpoint_dir, #
                      model_trans,
                      num_tasks,
                      num_intents,
                      eff_num_intents_task,
                      eff_num_slot, #
                      args_save_file,
                      model_save_file,
                      optim_save_file,
                      prior_mbert, # prior options
                      prior_intents,
                      prior_slots,
                      prior_adapter):

    dev_perf_best = 0.0
    best_model = None
    params = None
    saved_grads = None
    features_examples = []

    # Store the items as a dictionary where the key is the identifier

    for epoch in tqdm(range(args.epochs)):
        gc.collect()
        num_steps += 1

        if args.use_leitner_queue:
            next_item_ids = lt_scheduler.next_items(epoch)
            num_iter = len(next_item_ids)//args.batch_size # depending on the output of next_items over batch size
            shuffle = True
            if epoch == 0:
                shuffle = False
            scheduler_examples = AugmentedList([dataset.get_item_by_id(id_) for id_ in next_item_ids],
                                               shuffle_between_epoch=shuffle)
        else:
            scheduler_examples = train_examples

        for step_iter in tqdm(range(num_iter)):
            train_outputs = train(args.batch_size,
                                  optimizer,
                                  intent_embeddings,
                                  model,
                                  grad_dims,
                                  cont_learn_alg,
                                  dataset,
                                  scheduler_examples,
                                  memory,
                                  subtask_size,
                                  writer,
                                  epoch,
                                  train_idx,
                                  step_iter,
                                  checkpoint_dir,
                                  model_trans,
                                  num_tasks,
                                  num_intents,
                                  eff_num_intents_task,
                                  eff_num_slot,
                                  lt_scheduler,
                                  eval_sched_freq=-1,
                                  sample_sizes=sample_sizes)
            if args.use_slots:
                intent_loss, slot_loss, params, saved_grads, optimizer, model, intent_embeddings, lt_scheduler = train_outputs
                if step_iter % args.test_steps == 0:
                    app_log.info('Epoch {} | Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(epoch,
                                                                                                         step_iter,
                                                                                                         intent_loss.mean(),
                                                                                                         slot_loss.mean()))
            else:
                intent_loss, params, saved_grads, optimizer, model, intent_embeddings, lt_scheduler = train_outputs
                if step_iter % args.test_steps == 0:
                    app_log.info('Epoch {} | Iter {} | Intent Loss = {:.4f} '.format(epoch,
                                                                                     step_iter,
                                                                                     intent_loss.mean()))

        if args.use_leitner_queue:
            left_over_size = len(next_item_ids) % args.batch_size
        else:
            left_over_size = subtask_size[train_idx] % args.batch_size

        if left_over_size > 0:
            train_outputs = train(left_over_size,
                                  optimizer,
                                  intent_embeddings,
                                  model,
                                  grad_dims,
                                  cont_learn_alg,
                                  dataset,
                                  scheduler_examples,
                                  memory,
                                  subtask_size,
                                  writer,
                                  epoch,
                                  train_idx,
                                  step_iter,
                                  checkpoint_dir,
                                  model_trans,
                                  num_tasks,
                                  num_intents,
                                  eff_num_intents_task,
                                  eff_num_slot,
                                  lt_scheduler,
                                  eval_sched_freq=1,
                                  sample_sizes=sample_sizes)
            if args.use_slots:
                intent_loss, slot_loss, params, saved_grads, optimizer, model, intent_embeddings, lt_scheduler = train_outputs
                if step_iter % args.test_steps == 0:
                    app_log.info('Epoch {} | Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(epoch,
                                                                                                         step_iter,
                                                                                                         intent_loss.mean(),
                                                                                                         slot_loss.mean()))
            else:
                intent_loss, params, saved_grads, optimizer, model, intent_embeddings, lt_scheduler = train_outputs
                if step_iter % args.test_steps == 0:
                    app_log.info('Epoch {} | Iter {} | Intent Loss = {:.4f} '.format(epoch,
                                                                                     step_iter,
                                                                                     intent_loss.mean()))


        if args.use_leitner_queue:
            flag_continue = False
            for deck in range(0, lt_scheduler.num_decks):
                if len(lt_scheduler.decks[deck]) != 0:
                    flag_continue = True

            if not flag_continue:
                print("Converged after epochs with all items in the last deck")
                break

        app_log.info(">>>>>>> Dev Performance >>>>>")
        dev_out_path = None
        if args.save_dev_pred:
            dev_out_path = os.path.join(predictions_dir,
                                        "Dev_perf-Epoch_" + str(epoch) + "-train_" + str(train_idx))

        if dev_stream[train_idx]['size'] > 0:
            _, dev_perf = evaluate_report(dataset,
                                          memory,
                                          cont_learn_alg,
                                          dev_stream[train_idx],
                                          model,
                                          train_lang,
                                          train_idx,
                                          train_lang,
                                          train_idx,
                                          num_steps,
                                          writer,
                                          args,
                                          app_log,
                                          device,
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
                                                                    memory,
                                                                    cont_learn_alg,
                                                                    test_stream[test_subtask_lang],
                                                                    best_model,
                                                                    train_lang,
                                                                    train_idx,
                                                                    test_subtask_lang,
                                                                    test_idx,
                                                                    num_steps,
                                                                    writer,
                                                                    args,
                                                                    app_log,
                                                                    device,
                                                                    name="test",
                                                                    out_path=os.path.join(predictions_dir,
                                                                                          "Test_perf-Epoch_" + str(epoch)
                                                                                          + "-train_" + train_lang
                                                                                          + "-test_" + test_subtask_lang),
                                                                    verbose=args.verbose,
                                                                    prior_mbert=prior_mbert[conversion_lang[test_subtask_lang]],
                                                                    prior_intents=prior_intents[conversion_lang[test_subtask_lang]],
                                                                    prior_slots=prior_slots[conversion_lang[test_subtask_lang]],
                                                                    prior_adapter=prior_adapter[conversion_lang[test_subtask_lang]])

            with open(os.path.join(metrics_dir,
                                   "epoch_"+str(epoch)+"_metrics_"+str(train_idx)+".pickle"), "wb") \
                    as output_file:
                pickle.dump(metrics, output_file)

    return params, saved_grads, best_model, optimizer, intent_embeddings


def test_at_end_training(best_model,
                         dataset,
                         test_stream,
                         memory,
                         cont_learn_alg,
                         train_idx,
                         train_lang,
                         num_steps,
                         writer,
                         predictions_dir,
                         metrics_dir,
                         prior_mbert,
                         prior_intents,
                         prior_slots,
                         prior_adapter):

    app_log.info("------------------------------------ TESTING At the end of the training")
    metrics = {task: {} for task in test_stream} # could be either per subtask or language
    for test_idx, test_subtask_lang in enumerate(test_stream):
        if test_stream[test_subtask_lang]['size'] > 0:
            metrics[test_subtask_lang], _ = evaluate_report(dataset,
                                                            memory,
                                                            cont_learn_alg,
                                                            test_stream[test_subtask_lang],
                                                            best_model,
                                                            train_lang,
                                                            train_idx,
                                                            test_subtask_lang,
                                                            test_idx,
                                                            num_steps,
                                                            writer,
                                                            args,
                                                            app_log,
                                                            device,
                                                            name="test",
                                                            out_path=os.path.join(predictions_dir,
                                                                                  "End_test_perf-train_"+train_lang
                                                                                  + "-test_" + test_subtask_lang),
                                                            verbose=args.verbose,
                                                            prior_mbert=prior_mbert[conversion_lang[test_subtask_lang]],
                                                            prior_intents=prior_intents[conversion_lang[test_subtask_lang]],
                                                            prior_slots=prior_slots[conversion_lang[test_subtask_lang]],
                                                            prior_adapter=prior_adapter[conversion_lang[test_subtask_lang]])

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

    prediction_dir = os.path.join(results_dir, "predictions")
    if not os.path.isdir(prediction_dir):
        os.makedirs(prediction_dir)

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

    dataset = NLUDataset(args.data_root,
                         args.setup_opt,
                         args.setup_cillia,
                         args.multi_head_out,
                         args.use_mono,
                         tokenizer,
                         args.data_format,
                         args.use_slots,
                         args.seed,
                         args.languages,
                         args.order_class,
                         args.order_lang,
                         args.order_lst,
                         args.num_intent_tasks,
                         args.num_lang_tasks,
                         args.max_mem_sz,
                         intent_types=INTENT_TYPES,
                         slot_types=SLOT_TYPES)

    eff_num_intent = len(dataset.intent_types)
    num_intents = len(dataset.intent_types)
    eff_num_slot = len(dataset.slot_types)

    if args.setup_opt in ["cll", "multi-incr-cll", "cll-er_kd", "multi"]:
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
    elif args.setup_opt in ["multi-incr-cll", "cll", "cll-er_kd"]:
        if len(args.order_lst) > 0:
            num_tasks = len(args.order_lst.split("_"))
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

    if device != torch.device("cpu"):
        model.cuda()

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
    elif args.cont_learn_alg in ["er"]:
        cont_learn_alg = ER(args,
                            device,
                            writer)
    elif "kd" in args.cont_learn_alg:
        cont_learn_alg = KD(args,
                            device,
                            writer,
                            kd_mode=args.cont_learn_alg.split("-")[1])

    elif args.cont_learn_alg == "mbpa":
        cont_learn_alg = MBPA(args,
                              device,
                              model_trans,
                              num_tasks,
                              num_intents,
                              eff_num_intents_task,
                              eff_num_slot)
    else:
        cont_learn_alg = None

    if args.random_pred:
        metrics = {lang: {} for lang in dataset.test_stream}
        for lang in test_stream:
            print("HERE;", lang)
        memory = None
        for test_idx, test_subtask_lang in enumerate(test_stream):
            metrics[test_subtask_lang], _ = evaluate_report(dataset,
                                                            memory,
                                                            cont_learn_alg,
                                                            test_stream[test_subtask_lang],
                                                            model,
                                                            args.cil_stream_lang,
                                                            0,
                                                            test_subtask_lang,
                                                            test_idx,
                                                            0,
                                                            writer,
                                                            args,
                                                            app_log,
                                                            device,
                                                            name="init",
                                                            out_path=os.path.join(prediction_dir,
                                                                                  "initial_perf_" + test_subtask_lang + ".txt"),
                                                            verbose=args.verbose)

        with open(os.path.join(metrics_dir, "initial_metrics.pickle"), "wb") as output_file:
            pickle.dump(metrics, output_file)

        exit(0)

    optimizer, scheduler = set_optimizer(args, list(model.parameters()))
    model.zero_grad()

    grad_dims = []
    for n, p in model.named_parameters():
        if name_in_list(args.cont_comp, n):
            grad_dims.append(p.data.numel())

    prior_mbert = [None for _ in train_stream]
    prior_intents = [None for _ in train_stream]
    prior_slots = [None for _ in train_stream]
    prior_adapter = [None for _ in train_stream]

    if args.multi_head_in:
        if args.emb_enc_subtask_spec.split("_") == ["all"]:
            print("-------WE ARE HERE in ALLLLLL")
            prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root,
                                                                                            model_name)).
                            named_parameters()} for _ in train_stream]

            print("prior_mbert[0]:", prior_mbert[0].keys())
        else:
            # args.emb_spec_subtask_spec = "encoder.layer.0._encoder.layer.1._encoder.layer.2._encoder.layer.3._encoder.layer.4._encoder.layer.5._encoder.layer.6._encoder.layer.7._encoder.layer.8."
            print("-------WE ARE HERE in COOOMPONEEEENTS args.emb_spec_subtask_spec: ", args.emb_enc_subtask_spec)
            prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root,
                                                                                            model_name)).
                           named_parameters() if name_in_list(args.emb_enc_subtask_spec.split("_"), k)}
                           for _ in train_stream]



            for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root, model_name)).named_parameters():
                print("k:", k, " name_in_list(args.emb_enc_subtask_spec, k):", name_in_list(args.emb_enc_subtask_spec.split("_"), k))
            print("prior_mbert[0]:", prior_mbert[0].keys())


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

    if args.use_adapters:
        Adapter = import_from('contlearnalg.adapter', args.adapter_type)
        prior_adapter = [{k: v for k, v in Adapter(model_trans.config.hidden_size).named_parameters()}
                         for _ in train_stream]

    if args.setup_opt in ["cll", "multi-incr-cll", "cil", "cil-other", "multi-incr-cil", "cll-er_kd", "cll-equal", "cll-equal-er_kd", "cll-k-shots", "cll-n-ways"]:
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
            original_adapter = copy.deepcopy(model.adapter) # TODO CHANGE MANY ADAPTERS
        else:
            original_adapter = None

        mean_all_stream = []
        sum_all_stream = []
        intent_embeddings = {intent: [] for intent in range(len(INTENT_TYPES))}

        for train_idx, train_subtask_lang in enumerate(train_stream):
            print("train_idx:", train_idx, " train_subtask_lang:", train_subtask_lang)
            if train_subtask_lang["size"] == 0:
                app_log.warning("Skipped subtask/language: %d", train_idx)
                continue
            sample_sizes.append(int(train_subtask_lang["size"]*args.old_task_prop))
            num_steps = 0
            num_iter = train_subtask_lang["size"]//args.batch_size # TODO add samples that are not covered at the end of the batch
            train_lang = train_subtask_lang["lang"]
            train_examples = train_subtask_lang["examples"]

            if args.use_leitner_queue:
                lt_scheduler = LeitnerQueue(num_decks=5,
                                            dataset=dataset,
                                            train_examples=train_examples,
                                            nb_examples=train_subtask_lang["size"],
                                            demote_to_first=args.demote_to_first_deck)
            else:
                lt_scheduler = None

            if args.setup_opt in ["cll-er_kd", "cll-equal-er_kd"]:
                memory = train_subtask_lang["memory"]
            else:
                memory = None
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

            if args.use_pretrained_adapters:
                for k, v in model.named_parameters():
                    if "trans_model" in k:
                        v.requires_grad = False

                    if ".adapters."+train_lang+"." in k:
                        v.requires_grad = True

            subtask_size[train_idx] = train_subtask_lang["size"]

            """ 1. Reinitialize linear layers for each new language """
            if args.multi_head_in or args.multi_head_out or args.use_adapters:
                model_dict = model.state_dict()

            if args.multi_head_in:
                if args.emb_enc_subtask_spec.split("_") == ["all"]:  # could either be lang or subtask specific
                    trans_model_dict = {"trans_model."+k: v for k, v in original_mbert.named_parameters()}
                else:
                    trans_model_dict = {"trans_model."+k: v for k, v in original_mbert.named_parameters()
                                        if name_in_list(args.emb_enc_subtask_spec.split("_"), k)}

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
                adapter_dict = {"adapter."+k: v for k, v in
                                original_adapter.named_parameters()}
                model_dict.update(adapter_dict)

            if args.multi_head_in or args.multi_head_out or args.use_adapters:
                model.load_state_dict(model_dict)

            params, saved_grads, best_model, optimizer, intent_embeddings = train_task_epochs(model,
                                                                                              lt_scheduler,
                                                                                              intent_embeddings,
                                                                                              optimizer,
                                                                                              grad_dims,
                                                                                              cont_learn_alg,
                                                                                              dataset,
                                                                                              train_examples,
                                                                                              memory,
                                                                                              dev_stream,
                                                                                              test_stream,
                                                                                              subtask_size,
                                                                                              sample_sizes,
                                                                                              num_iter,
                                                                                              train_idx,
                                                                                              train_lang, # training name of the task
                                                                                              num_steps,
                                                                                              writer,  # saving options
                                                                                              results_dir,
                                                                                              metrics_dir,
                                                                                              checkpoint_dir, #
                                                                                              model_trans,
                                                                                              num_tasks,
                                                                                              num_intents,
                                                                                              eff_num_intents_task,
                                                                                              eff_num_slot, #
                                                                                              args_save_file,
                                                                                              model_save_file,
                                                                                              optim_save_file,
                                                                                              prior_mbert,  # prior options
                                                                                              prior_intents,
                                                                                              prior_slots,
                                                                                              prior_adapter)

            """ 2. Saving the trained weights of MBERT in each language to be used later on in testing stage"""

            if args.multi_head_in:
                mbert_task = copy.deepcopy(best_model.trans_model)
                prior_mbert[train_idx] = {k: v for k, v in mbert_task.named_parameters()
                                          if name_in_list(args.emb_enc_subtask_spec.split("_"), k)}

                if train_idx == 0:
                    mbert_task_test = copy.deepcopy(model.trans_model)
                    mean_all, sum_all = compute_change(mbert_task_test, original_mbert,
                                                       original_intent, original_intent,
                                                       original_slot, original_slot)

                    print("JUST DOUBLE CHECKING ON THE BUG OF MULTI-HEAD-IN mean_all comparing mbert with the original ", mean_all, sum_all)

            if args.multi_head_out:
                # TODO DOUBLE CHECK model.intent_classifier[train_idx]
                if "cil" in args.setup_opt:
                    intents_task = copy.deepcopy(best_model.intent_classifier[train_idx])
                else:
                    intents_task = copy.deepcopy(best_model.intent_classifier)
                prior_intents[train_idx] = {k: v for k, v in intents_task.named_parameters()}

                slots_task = copy.deepcopy(best_model.slot_classifier)
                prior_slots[train_idx] = {k: v for k, v in slots_task.named_parameters()}

            if args.use_adapters:
                adapter = copy.deepcopy(best_model.adapter)
                prior_adapter[train_idx] = {k: v for k, v in adapter.named_parameters()}

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

            # params = {n: p for n, p in best_model.named_parameters()}
            # with open(params_save_file, "wb") as file:
            #     pickle.dump(params, file)

            if "kd" in args.cont_learn_alg:
                torch.save(best_model, params_save_file)

            test_at_end_training(best_model,
                                 dataset,
                                 test_stream,
                                 memory,
                                 cont_learn_alg,
                                 train_idx,
                                 train_lang,
                                 num_steps,
                                 writer,
                                 prediction_dir,
                                 metrics_dir,
                                 prior_mbert,
                                 prior_intents,
                                 prior_slots,
                                 prior_adapter)

        with open(os.path.join(metrics_dir, "mean_all_stream_mbert.pickle"), "wb") as output_file:
            pickle.dump(mean_all_stream, output_file)

    elif args.setup_opt in ["multi", "multi-equal"]:
        """
        Setup 5: Multi-task/Joint Learning: train on all languages and intent classes at the same time
        """

        # There is only one task here no subtasks
        prior_mbert = [None, None, None, None, None, None]
        prior_intents = [None, None, None, None, None, None]
        prior_slots = [None, None, None, None, None, None]
        prior_adapter = [None, None, None, None, None, None]
        train_examples = dataset.train_stream["examples"]
        dev_stream = [dataset.dev_stream]
        test_stream = dataset.test_stream
        subtask_size = {0: dataset.train_stream["size"]}
        num_iter = dataset.train_stream["size"]//args.batch_size
        train_idx = 0  # only one task here
        sample_sizes = [int(dataset.train_stream["size"]*args.old_task_prop)] # not needed anyways because no continuous learning here
        train_lang = "-".join(args.languages)  # all languages
        num_steps = 0

        if args.use_leitner_queue:
            lt_scheduler = LeitnerQueue(num_decks=5,
                                        dataset=dataset,
                                        train_examples=train_examples,
                                        nb_examples=dataset.train_stream["size"],
                                        demote_to_first=args.demote_to_first_deck)
        else:
            lt_scheduler = None

        intent_embeddings = {intent: [] for intent in range(len(INTENT_TYPES))}
        memory = None

        params, saved_grads, best_model, optimizer, intent_embeddings = train_task_epochs(model,
                                                                                          lt_scheduler,
                                                                                          intent_embeddings,
                                                                                          optimizer,
                                                                                          grad_dims,
                                                                                          cont_learn_alg,
                                                                                          dataset,
                                                                                          train_examples,
                                                                                          memory,
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
                                                                                          model_trans,
                                                                                          num_tasks,
                                                                                          num_intents,
                                                                                          eff_num_intents_task,
                                                                                          eff_num_slot,
                                                                                          args_save_file,
                                                                                          model_save_file,
                                                                                          optim_save_file,
                                                                                          prior_mbert,  # prior options
                                                                                          prior_intents,
                                                                                          prior_slots,
                                                                                          prior_adapter)

        test_at_end_training(best_model,
                             dataset,
                             test_stream,
                             memory,
                             cont_learn_alg,
                             train_idx,
                             train_lang,
                             num_steps,
                             writer,
                             prediction_dir,
                             metrics_dir,
                             prior_mbert,
                             prior_intents,
                             prior_slots,
                             prior_adapter)


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
    add_spaced_repetition_setup(parser)
    args = parser.parse_args()

    args = get_config_params(args)

    conversion_lang = {order_lang: orderlang_id for orderlang_id, order_lang in enumerate(args.order_lst.split("_"))}

    set_seed(args.seed)
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

    print("args.use_processor_sharing:", args.use_processor_sharing)
    print("args.evaluate_one_batch:", args.evaluate_one_batch)
    print("args.eval_sched_freq:", args.eval_sched_freq)
    print("args.warm_start_epochs:", args.warm_start_epochs)
    print("args.use_leitner_queue:", args.use_leitner_queue)

    run(results_dir, args, app_log)

    if args.no_debug:
        sys.stdout.close()
        sys.stdout = stdoutOrigin
