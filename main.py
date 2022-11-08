from parser_args import *

# Dataset
from data_utils import *

# Downstream Models
from transformers_config import MODELS_dict
from basemodels.transNLUCRF import TransNLUCRF
from basemodels.transNLI import TransNLI
from basemodels.transQA import TransQA 

# Continual Models
from contlearnalg.EWC_grads import EWC
# from contlearnalg.GEM import GEM
from contlearnalg.MbPA import MBPA
from contlearnalg.ER import ER
from contlearnalg.KD import KD
from utils import variable, format_store_grads, name_in_list, logger, import_from, evaluate_report, set_optimizer, \
    get_config_params, create_new_dir, transfer_batch_cuda

# Schedulers
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
import os
import psutil

# GPU/CPU
gpus_list = list(range(torch.cuda.device_count()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        sum_layers.update({key+"."+k: 0.0 for k, _ in params_map[key][0].named_parameters()})
        mean_layers.update({key+"."+k: 0.0 for k, _ in params_map[key][0].named_parameters()})

    for key in list(params_map.keys())+["all"]:
        sum_layers.update({key: 0.0})
        mean_layers.update({key: 0.0})

    count = 0
    count_comp = {key: 0 for key in params_map.keys()}
    for key in params_map.keys():
        comp_items = {key+"."+k: v for k, v in params_map[key][1].named_parameters()}
        sum_layers, mean_layers, count, count_comp[key] = compute_diff(params_map[key][0], comp_items, sum_layers, mean_layers, key, count)

    for key in sum_layers:
        print(key, sum_layers[key], mean_layers[key])

    return mean_layers, sum_layers

def test_one_batch(batch,
                   examples,
                   task_name,
                   model,
                   train_idx,
                   test_idx,
                   memory,
                   cont_learn_alg,
                   dataset,
                   name="train",
                   prior_mbert=None,
                   prior_classes=None,
                   prior_slots=None,
                   prior_adapter=None):

    app_log.info("Evaluating on Training batch ...")

    # TODO something about this
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
            ### 1. wanted keys, values are in trans_model
            if "cil" in args.setup_opt:
                gclassifier_dict = {"gclassifier." + str(test_idx) + "." + k: v for k, v in
                                          prior_classes.items()}
            else:
                gclassifier_dict = {"gclassifier." + k: v for k, v in prior_classes.items()}
            ### 2. overwrite entries in the existing state dict
            model_dict.update(gclassifier_dict)

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

    class_corrects = 0
    sents_text = []

    class_true = []
    class_pred = []

    batch = transfer_batch_cuda(batch, device)

    # inputs = batch
    # inputs["train_idx"] = i_task

    # outputs = model(**inputs)

    # all_losses = outputs.loss
    # loss = all_losses["overall"]
    # pooled_output = outputs.pool_out

    if train_idx > 0 and name == "test":
        if args.cont_learn_alg == "mbpa":
            """ Local adaptation of MbPA """

            q = model.get_embeddings(batch["input_ids"], batch["input_masks"])[0]

            if args.use_reptile:
                if args.use_batches_reptile:
                    eval_model = cont_learn_alg.forward_reptile_many_batches(memory, q, train_idx, model, dataset)
                else:
                    eval_model = cont_learn_alg.forward_reptile_one_batch(memory, q, train_idx, model, dataset)
            else:
                eval_model = cont_learn_alg.forward(memory, 
                                                    q, 
                                                    train_idx, 
                                                    model,
                                                    dataset)  # this is taking into consideration only the task we are testing from assuming we know that task.
        else:
            eval_model = model
    else:
        eval_model = model

    eval_model.eval()

    inputs = batch
    inputs["train_idx"] = test_idx

    with torch.no_grad():
        outputs = eval_model(**inputs)

    if task_name in ["tod", "nli"]:
        # Golden Truth/Predictions Valid for intents and classes
        true_classes = batch["labels"].tolist()
        pred_classes = outputs.logits["class"].max(1)[1]
        input_identifiers = [example.unique_id for example in examples]

        class_outcomes = []
        for i in range(len(true_classes)):
            if true_classes[i] == pred_classes[i]:
                class_outcomes.append(1)
            else:
                class_outcomes.append(0)

        assert len(class_outcomes) == len(input_identifiers)

        eval_outcomes = {input_identifiers[i]:class_outcomes[i] for i in range(len(input_identifiers))}

    else: # qa
        for name in ["start_positions", "end_positions"]:
            true_classes = batch[name].tolist()
            pred_classes = outputs.logits[name].max(1)[1]
            input_identifiers = [example.unique_id for example in examples]

            class_outcomes = []
            for i in range(len(true_classes)):
                if true_classes[i] == pred_classes[i]:
                    class_outcomes.append(1)
                else:
                    class_outcomes.append(0)

            assert len(class_outcomes) == len(input_identifiers)

            eval_outcomes = {input_identifiers[i]:class_outcomes[i] for i in range(len(input_identifiers))}

            all_results = []
            predictions = compute_predictions_log_probs(
                            examples,
                            features,
                            all_results,
                            args.n_best_size,
                            args.max_answer_length,
                            output_prediction_file,
                            output_nbest_file,
                            output_null_log_odds_file,
                            start_n_top,
                            end_n_top,
                            args.version_2_with_negative,
                            tokenizer,
                            args.verbose_logging,
                        )

    # for name, _logits in outputs.logits.items():
    #     if name == "slots_nocrf":
    #         slots_true = []
    #         slots_pred = []

    #         slots_true_all = []
    #         slots_pred_all = []

    #         true_slot = batch["slot_labels"][0]

    #         slot_logits = [_logits[j, :length].data.numpy() for j, length in enumerate(batch["lengths"])]
    #         pred_slot = list(_logits[0])

    #         true_slot_l = [dataset.slot_types[s] for s in true_slot]
    #         pred_slot_l = [dataset.slot_types[s] for s in pred_slot]

    #         true_slot_no_x = []
    #         pred_slot_no_x = []

    #         for j, slot in enumerate(true_slot_l):
    #             if slot != "X":
    #                 if j < len(pred_slot_l):
    #                     true_slot_no_x.append(true_slot_l[j])
    #                     pred_slot_no_x.append(pred_slot_l[j])

    #         slots_true.append(true_slot_no_x)
    #         slots_pred.append(pred_slot_no_x)

    #         slots_true_all.extend(true_slot_no_x)
    #         slots_pred_all.extend(pred_slot_no_x)

    return eval_outcomes

def test_train_batches(dataset_train,
                       nb_examples,
                       model,
                       train_idx,
                       test_idx,
                       memory,
                       cont_learn_alg,
                       dataset,
                       task_name,
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
                intent_classifier_dict = {"gclassifier." + str(test_idx) + "." + k: v for k, v in
                                          prior_intents.items()}
            else:
                intent_classifier_dict = {"gclassifier." + k: v for k, v in prior_intents.items()}
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

    eval_outcomes = {}
    for _ in tqdm(range(nb_examples)):
        batch_one, examples \
            = dataset.next_batch(1, dataset_train)

        batch_one = transfer_batch_cuda(batch_one, device)

        if train_idx > 0 and name == "test":
            if args.cont_learn_alg == "mbpa":
                """ Local adaptation of MbPA """

                q = model.get_embeddings(batch_one["input_ids"], batch_one["input_masks"])[0]

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

        inputs = batch_one
        inputs["train_idx"] = test_idx

        with torch.no_grad():
            outputs = eval_model(**inputs)

        if task_name in ["tod", "nli"]:

            # Golden Truth/Predictions Valid for intents and classes
            true_class = batch_one["labels"].squeeze().item()
            pred_class = outputs.logits["class"].squeeze().max(0)[1]
            input_identifiers = [example.unique_id for example in examples]

            if true_class == pred_class:
                class_outcome = 1
            else:
                class_outcome = 0

            eval_outcomes.update({input_identifiers[0]: class_outcome})


        else: # qa
            for name in ["start_positions", "end_positions"]:
                true_classes = batch[name].tolist()
                pred_classes = outputs.logits[name].max(1)[1]
                input_identifiers = [example.unique_id for example in examples]

                class_outcomes = []
                for i in range(len(true_classes)):
                    if true_classes[i] == pred_classes[i]:
                        class_outcomes.append(1)
                    else:
                        class_outcomes.append(0)

                assert len(class_outcomes) == len(input_identifiers)

                eval_outcomes = {input_identifiers[i]:class_outcomes[i] for i in range(len(input_identifiers))}

                all_results = []
                predictions = compute_predictions_log_probs(
                                examples,
                                features,
                                all_results,
                                args.n_best_size,
                                args.max_answer_length,
                                output_prediction_file,
                                output_nbest_file,
                                output_null_log_odds_file,
                                start_n_top,
                                end_n_top,
                                args.version_2_with_negative,
                                tokenizer,
                                args.verbose_logging,
                            )


    return eval_outcomes


def train(batch_left_size,
          optimizer,
          classes_embeddings,
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
          ltn_scheduler,
          sample_sizes=[]):

    optimizer.zero_grad()
    model.train()

    
    # Take batch by batch and move to cuda

    # NLI => BATCH =>           (input_ids, input_masks, token_type_ids, labels),  examples
    #     => INPUT TO MODEL =>   input_ids, input_masks, token_type_ids, train_idx, labels=None
    #     => OUTPUT => logits, loss, pooled_output || loss, pooled_output
    # TOD => BATCH =>          input_ids, input_masks, lengths, intent_labels, slot_labels
    #     => INPUT TO MODEL => input_ids, input_masks, train_idx, lengths, intent_labels=None, slot_labels=None
    #     => OUTPUT => logits_intent, logits_slots, _, intent_loss, slot_loss, loss, pooled_output
    # TYDIQA => BATCH =>         (input_ids, attention_masks, token_type_ids, start_positions, end_positions, cls_index, p_mask, langs), features
    #        => INPUT TO MODEL => input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, start_positions, end_positions, output_attentions, output_hidden_states, return_dict, train_idx 
    #        => OUTPUT => total_loss, start_logits, end_logits,  lm_output.hidden_states, lm_output.attentions

    batch, examples = dataset.next_batch(batch_left_size, train_examples)
    batch = transfer_batch_cuda(batch, device)

    params = None
    saved_grads = None

    inputs = batch
    inputs["train_idx"] = i_task

    outputs = model(**inputs)

    all_losses = outputs.loss
    loss = all_losses["overall"]
    pooled_output = outputs.pool_out

    if "labels" not in inputs:
        labels = inputs["start_positions"]
    else:
        labels = inputs["labels"] # intents or nli 

    for loss_name, loss_value in outputs.loss.items():
        writer.add_scalar(loss_name+'train_loss_'+str(i_task), loss_value.detach().mean(), num_steps*epoch)

    for i in range(batch_left_size):
        label = labels[i].squeeze().item()
        classes_embeddings[label].append(pooled_output[i])

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

        del outputs
        torch.cuda.empty_cache()  # TODO try this emptying of cache after backward

        if i_task > 0:
            # memory training is interleaved with the main training so that the process doesn't overfit to the memory
            if num_steps % 100 == 0:
                for task_memory_id, task_memory in enumerate(memory):
                    # sample a batch from the memory of size sampling_k to go over
                    batch, _ = dataset.next_batch(args.sampling_k, task_memory)
                    optimizer.zero_grad()

                    batch = transfer_batch_cuda(batch, device)

                    inputs = batch
                    inputs["train_idx"] = i_task

                    er_outputs = model(**inputs)
                    er_loss = er_outputs.loss["overall"]
 
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

                    batch = transfer_batch_cuda(batch, device)

                    inputs = batch
                    inputs["train_idx"] = i_task

                    kd_outputs = model(**inputs)
                    kd_loss = kd_outputs.loss["overall"]
                    kd_pooled_output = kd_outputs["pooled_output"]

                    min_num_sent = min(args.sampling_k//len(memory), args.batch_size)
                    if args.cont_learn_alg == "kd-logits":
                        kd_loss = 0
                        for kd_logits_name, kd_logits_value in kd_outputs.logits.items():
                            # TODO normalize the shapes of logits and kd_logits 
                            if "nokd" not in kd_logits_name:
                                kd_loss += F.mse_loss(kd_logits_value[:min_num_sent].to(device).float(),
                                                    outputs.logits[kd_logits_name][:min_num_sent].to(device).float())

                            # kd_intent_loss = torch.nn.KLDivLoss(size_average=False)(main_intent_logits, memory_intent_logits)
                            # min_seq_len = min(kd_logits_slots_.shape[1], logits_slots_.shape[1])

                            # kd_slot_loss = F.mse_loss(kd_logits_slots_[:min_num_sent, :min_seq_len, :].to(device).float(),
                            #                         logits_slots_[:min_num_sent, :min_seq_len, :].to(device).float())

                            # kd_slot_loss = torch.nn.KLDivLoss(size_average=False)(main_slot_logits, memory_slot_logits)

                    else:  # rep
                        kd_loss = F.mse_loss(kd_pooled_output[:min_num_sent].to(device).float(),
                                             pooled_output[:min_num_sent].to(device).float())

                    total_loss += 0.01 * kd_loss.mean()  # kd_loss.mean() # kd_er_loss.mean()
                    writer.add_scalar('kd_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                      kd_loss.detach().mean(), num_steps*epoch)

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
        del pooled_output
        for k, v in outputs.logits.items():
            del v
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

                    batch = transfer_batch_cuda(batch, device)

                    inputs = batch
                    inputs["train_idx"] = i_task

                    kd_outputs = model(**inputs)
                    kd_loss = kd_outputs["loss"]
                    kd_pooled_output = kd_outputs["pooled_output"]

                    prev_model = torch.load(os.path.join(checkpoint_dir, "pytorch_params_"+str(task_memory_id)))

                    with torch.no_grad():
                        prev_outputs = prev_model(**inputs)
                        # TODO prev outputs

                    if args.cont_learn_alg == "kd-logits":
                        kd_loss = 0
                        for kd_logits_name, kd_logits_value in kd_outputs.logits.items():
                             if "nokd" not in kd_logits_name:
                                kd_loss += F.mse_loss(kd_logits_value.to(device).float(),
                                                    prev_outputs.logits[kd_logits_name].to(device).float())

                    else:  # rep
                        kd_loss = F.mse_loss(kd_pooled_output.to(device).float(),
                                             prev_outputs.pool_out.detach().to(device).float())

                    total_loss += 0.01 * kd_loss.mean()  # kd_loss.mean() # kd_er_loss.mean()
                    writer.add_scalar('kd_loss_'+str(i_task)+'-mem_'+str(task_memory_id),
                                      kd_loss.detach().mean(), num_steps*epoch)

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
        del pooled_output
        for k, v in outputs.logits.items():
            del v
        torch.cuda.empty_cache()  # TODO try this emptying of cache after backward

    else:
        loss = loss.mean()
        loss.backward() #retain_graph=True
        optimizer.step()
        gc.collect()
        del pooled_output
        for k, v in outputs.logits.items():
            del v
        torch.cuda.empty_cache()  # TODO try this emptying of cache after backward

        params = None
        saved_grads = None

    if args.use_leitner_queue:
        """ Eval on train here and populate/update the Leitner Queue accordingly for each element in the batch """
        if args.evaluate_one_batch:
            """ Eval only on one batch """
            eval_output = test_one_batch(batch,
                                         examples,
                                         args.task_name,
                                         model,
                                         i_task,
                                         i_task,
                                         memory,
                                         cont_learn_alg,
                                         dataset)

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
                                             args.task_name)

        ##  eval to get the outcome for each item
        ## Generate a timestamp which is uniform for each item in the batch
        ltn_scheduler.place_items(eval_output)

        app_log.info(ltn_scheduler.rep_sched())


    return all_losses, params, saved_grads, optimizer, model, classes_embeddings, ltn_scheduler


def set_out_dir():
    # out_dir -> setup_opt -> slots -> trans_model -> random_init/mono/lang_order -> class_order -> cont_learn_alg ->
    #  -> headed -> adapters -> freezing
    order_lang_dict = {0: "high2lowlang",
                       1: "low2highlang",
                       2: "randomlang"}

    order_class_dict = {0: "high2lowclass",
                        1: "low2highclass",
                        2: "randomclass"}

    new_out_dir = os.path.join(args.out_dir,
                               args.data_name.upper())
    if args.use_leitner_queue:
        if args.demote_to_first_deck:
            new_out_dir = os.path.join(new_out_dir,
                                       "DemoteFirstDeck")
        else:
            new_out_dir = os.path.join(new_out_dir,
                                       "DemotePreviousDeck")
    else:
        new_out_dir = os.path.join(new_out_dir,
                                   "BASELINE")

    results_dir = os.path.join(new_out_dir,  # original output directory
                               args.setup_opt)  # setup option directory

    if args.task_name == "tod":
        results_dir = os.path.join(results_dir,
                                   (lambda x: "NLU" if x else "Intents_only")(args.use_slots)) # slot usage
    
    results_dir = os.path.join(results_dir,
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
                      classes_embeddings,
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
                      args_save_file,
                      model_save_file,
                      optim_save_file,
                      prior_mbert, # prior options
                      prior_classes,
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
                                  classes_embeddings,
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
                                  lt_scheduler,
                                  sample_sizes=sample_sizes)

            all_losses, params, saved_grads, optimizer, model, classes_embeddings, lt_scheduler = train_outputs
            log_msg = 'Epoch {} | Iter {}'.format(epoch, step_iter)

            for loss_name, loss_value in all_losses.items():
                if step_iter % args.test_steps == 0:
                    log_msg += '|'+ loss_name + ' Loss = {:.4f} '.format(loss_value.mean())

        if args.use_leitner_queue:
            left_over_size = len(next_item_ids) % args.batch_size
        else:
            left_over_size = subtask_size[train_idx] % args.batch_size

        if left_over_size > 0:
            train_outputs = train(left_over_size,
                                  optimizer,
                                  classes_embeddings,
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
                                  lt_scheduler,
                                  sample_sizes=sample_sizes)

            all_losses, params, saved_grads, optimizer, model, classes_embeddings, lt_scheduler = train_outputs
            log_msg = 'Epoch {} | Iter {}'.format(epoch, step_iter)

            for loss_name, loss_value in all_losses.items():
                if step_iter % args.test_steps == 0:
                    log_msg += '|'+ loss_name + ' Loss = {:.4f} '.format(loss_value.mean())


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
                                                                    prior_intents=prior_classes[conversion_lang[test_subtask_lang]],
                                                                    prior_slots=prior_slots[conversion_lang[test_subtask_lang]],
                                                                    prior_adapter=prior_adapter[conversion_lang[test_subtask_lang]])

            with open(os.path.join(metrics_dir,
                                   "epoch_"+str(epoch)+"_metrics_"+str(train_idx)+".pickle"), "wb") \
                    as output_file:
                pickle.dump(metrics, output_file)

    return params, saved_grads, best_model, optimizer, classes_embeddings


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
                         prior_classes,
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
                                                            prior_classes=prior_classes[conversion_lang[test_subtask_lang]],
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

    metrics_dir = create_new_dir(results_dir, "metrics")
    checkpoint_dir = create_new_dir(results_dir, "checkpoint")
    prediction_dir = create_new_dir(results_dir, "predictions")

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
                                          output_hidden_states=True,
                                          output_attentions=True)

    tokenizer = tokenizer_alias.from_pretrained(model_load_alias,
                                                do_lower_case=True,
                                                do_basic_tokenize=False)

    model_trans = model_trans_alias.from_pretrained(model_load_alias,
                                                    config=config)

    dataset = MultiPurposeDataset(args,
                                  tokenizer)

    eff_num_class = len(dataset.class_types)
    num_classes = len(dataset.class_types)
    eff_num_slot = len(dataset.slot_types)

    if args.setup_opt in ["cll", "multi-incr-cll", "cll-er_kd", "multi"]:
        train_stream = dataset.train_stream
        dev_stream = dataset.dev_stream
        test_stream = dataset.test_stream
    else:
        train_stream = dataset.train_stream[args.cil_stream_lang]
        dev_stream = dataset.dev_stream[args.cil_stream_lang]
        test_stream = dataset.test_stream[args.cil_stream_lang]

    """ eff_num_class/eff_num_slot """
    if args.setup_opt == "cil-other":
        eff_num_class += 1
        num_classes += 1

    if args.setup_opt == "multi":
        num_tasks = 1
        eff_num_classes_task = eff_num_class
    elif args.setup_opt in ["multi-incr-cll", "cll", "cll-er_kd"]:
        if len(args.order_lst) > 0:
            num_tasks = len(args.order_lst.split("_"))
        else:
            num_tasks = len(args.languages)
        eff_num_classes_task = eff_num_class
    else:  # args.setup_opt in ["cil", "cil-other", "cil-ll"]:
        range_classes = range(0, len(dataset.class_types), args.num_class_tasks)
        num_tasks = len(list(range_classes))
        eff_num_classes_task = [len(range(i, i+args.num_intent_tasks))
                                if i+args.num_intent_tasks < len(dataset.class_types)
                                else len(range(i, len(dataset.intent_types))) for i in range_classes]

    if args.task_name == "tod":
        model = TransNLUCRF(args=args,
                            trans_model=model_trans,
                            num_tasks=num_tasks,
                            num_labels=num_classes,
                            eff_num_classes_task=eff_num_classes_task,
                            device=device,
                            num_slots=eff_num_slot)

    elif args.task_name == "nli":
        model = TransNLI(args=args,
                         trans_model=model_trans,
                         num_tasks=num_tasks,
                         num_labels=num_classes,
                         eff_num_classes_task=eff_num_classes_task,
                         device=device)

    else: # qa
        model = TransQA(args=args,
                        trans_model=model_trans,
                        num_tasks=num_tasks,
                        num_labels=num_classes,
                        eff_num_classes_task=eff_num_classes_task,
                        device=device)


    if torch.cuda.device_count() > 1:
        app_log.info("torch.cuda.device_count(): %d", torch.cuda.device_count())
        gpus_list = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=gpus_list, dim=0)

    if device != torch.device("cpu"):
        model.cuda()

    ## TODO Continuous Learning Algorithms
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
                              num_classes,
                              eff_num_classes_task,
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
    prior_classes = [None for _ in train_stream]
    prior_slots = [None for _ in train_stream]
    prior_adapter = [None for _ in train_stream]

    if args.multi_head_in:
        if args.emb_enc_subtask_spec.split("_") == ["all"]:
            prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root,
                                                                                            model_name)).
                            named_parameters()} for _ in train_stream]
        else:
            prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root,
                                                                                            model_name)).
                           named_parameters() if name_in_list(args.emb_enc_subtask_spec.split("_"), k)}
                           for _ in train_stream]


    if args.multi_head_out:
        # TODO change to accommodate different numbers of intents
        if "cil" in args.setup_opt:
            prior_classes = [{k: v for k, v in nn.Linear(model.trans_model.config.hidden_size, eff_num_classes_task[i]).
                             named_parameters()} for i in range(len(train_stream))]
        else:
            prior_classes = [{k: v for k, v in nn.Linear(model.trans_model.config.hidden_size, num_classes).
                             named_parameters()} for _ in range(len(train_stream))]

        if args.task_name == "tod":
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
        original_class = copy.deepcopy(model.gclassifier)
        if args.task_name == "tod":
            original_slot = copy.deepcopy(model.slot_classifier)

        if args.use_adapters:
            original_adapter = copy.deepcopy(model.adapter) # TODO CHANGE MANY ADAPTERS
        else:
            original_adapter = None

        mean_all_stream = []
        sum_all_stream = []
        class_embeddings = {label: [] for label in range(len(dataset.class_types))}

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
                        if p.requires_grad and 'gclassifier' in n:
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
                    classifier_dict = {"gclassifier."+str(train_idx)+"."+k: v for k, v in original_class[train_idx].named_parameters()}
                else:
                    classifier_dict = {"gclassifier."+k: v for k, v in original_class.named_parameters()}
                model_dict.update(classifier_dict)

                if args.task_name == "tod":
                    slot_classifier_dict = {"slot_classifier."+k: v for k, v in original_slot.named_parameters()}
                    model_dict.update(slot_classifier_dict)

            if args.use_adapters:
                adapter_dict = {"adapter."+k: v for k, v in
                                original_adapter.named_parameters()}
                model_dict.update(adapter_dict)

            if args.multi_head_in or args.multi_head_out or args.use_adapters:
                model.load_state_dict(model_dict)

            params, saved_grads, best_model, optimizer, class_embeddings = train_task_epochs(model,
                                                                                             lt_scheduler,
                                                                                             class_embeddings,
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
                                                                                             args_save_file,
                                                                                             model_save_file,
                                                                                             optim_save_file,
                                                                                             prior_mbert,  # prior options
                                                                                             prior_classes,
                                                                                             prior_slots,
                                                                                             prior_adapter)

            """ 2. Saving the trained weights of MBERT in each language to be used later on in testing stage"""

            if args.multi_head_in:
                mbert_task = copy.deepcopy(best_model.trans_model)
                prior_mbert[train_idx] = {k: v for k, v in mbert_task.named_parameters()
                                          if name_in_list(args.emb_enc_subtask_spec.split("_"), k)}

                if train_idx == 0:
                    mbert_task_test = copy.deepcopy(model.trans_model)

                    params_map = {"trans_model": (mbert_task_test, original_mbert), 
                                  "gclassifier": (original_class, original_class)}

                    if args.task_name == "tod":
                        params_map.update({"slot_classifier": (original_slot, original_slot)})

                    mean_all, sum_all = compute_change(params_map)

                    print("JUST DOUBLE CHECKING ON THE BUG OF MULTI-HEAD-IN mean_all comparing mbert with the original ", mean_all, sum_all)

            if args.multi_head_out:
                # TODO DOUBLE CHECK model.intent_classifier[train_idx]
                if "cil" in args.setup_opt:
                    classes_task = copy.deepcopy(best_model.gclassifier[train_idx])
                else:
                    classes_task = copy.deepcopy(best_model.gclassifier)
                prior_classes[train_idx] = {k: v for k, v in classes_task.named_parameters()}

                if args.task_name == "tod":
                    slots_task = copy.deepcopy(best_model.slot_classifier)
                    prior_slots[train_idx] = {k: v for k, v in slots_task.named_parameters()}

            if args.use_adapters:
                adapter = copy.deepcopy(best_model.adapter)
                prior_adapter[train_idx] = {k: v for k, v in adapter.named_parameters()}

            if args.save_change_params:
                app_log.info("After training on language: %d", train_idx)

                params_map = {"trans_model": (mbert_task, original_mbert), 
                              "gclassifier": (classes_task, original_class)}

                if args.task_name == "tod":
                    params_map.update({"slot_classifier": (slots_task, original_slot)})

                mean_all, sum_all = compute_change(params_map)
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
                                 prior_classes,
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
        prior_classes = [None, None, None, None, None, None]
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

        class_embeddings = {label: [] for label in range(len(dataset.class_types))}
        memory = None

        params, saved_grads, best_model, optimizer, class_embeddings = train_task_epochs(model,
                                                                                          lt_scheduler,
                                                                                          class_embeddings,
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
                                                                                          args_save_file,
                                                                                          model_save_file,
                                                                                          optim_save_file,
                                                                                          prior_mbert,  # prior options
                                                                                          prior_classes,
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
                             prior_classes,
                             prior_slots,
                             prior_adapter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./main.py", description="Different options/arguments for running "
                                                              "continuous learning algorithms.")
    add_path_arguments(parser)
    add_setup_arguments(parser)
    add_dataset_arguments(parser)
    add_base_model_arguments(parser)
    add_model_expansion_arguments(parser)
    add_freezing_arguments(parser)
    cont_learn_arguments(parser)
    add_checkpoint_arguments(parser)
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

    print("args:", args)

    run(results_dir, args, app_log)

    if args.no_debug:
        sys.stdout.close()
        sys.stdout = stdoutOrigin
