from data_utils import *
import argparse
from consts import intent_types, slot_types
import gc
import numpy as np
from transformers_config import MODELS_dict
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR as SchedulerLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
import logstats
import os
import pickle
from copy import deepcopy
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
import copy
from contlearnalg.EWC_grads import EWC
from contlearnalg.GEM import GEM
from utils import format_store_grads


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

    input_ids, lengths, token_type_ids, input_masks, attention_mask, intent_labels, slot_labels, \
        input_texts = batch

    input_ids = input_ids.cuda()
    lengths = lengths.cuda()
    token_type_ids = token_type_ids.cuda() # TODO check how this is done
    input_masks = input_masks.cuda()
    attention_mask = attention_mask.cuda() # TODO make this to good usage
    intent_labels = intent_labels.cuda()
    slot_labels = slot_labels.cuda()

    if args.use_slots:
        logits_intents, logits_slots, intent_loss, slot_loss, loss = model(input_ids=input_ids,
                                                                           input_masks=input_masks,
                                                                           lengths=lengths,
                                                                           intent_labels=intent_labels,
                                                                           slot_labels=slot_labels)

        writer.add_scalar('train_intent_loss_'+str(i_task), intent_loss.mean(), num_steps*epoch)
        writer.add_scalar('train_slot_loss_'+str(i_task), slot_loss.mean(), num_steps*epoch)
    else:
        logits_intents, intent_loss, loss = model(input_ids=input_ids,
                                                  input_masks=input_masks,
                                                  lengths=lengths,
                                                  intent_labels=intent_labels)
        writer.add_scalar('train_intent_loss_'+str(i_task), intent_loss.mean(), num_steps*epoch)

    if args.cont_learn_alg == "ewc":
        if i_task > 0:
            loss += (args.ewc_lambda / 2) * cont_learn_alg.penalty(i_task)

        loss = loss.mean()

        loss.backward()

        params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        saved_grads = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            saved_grads[n] = variable(p.data)
            # saved_grads[n] = p.data.cpu()#.detach().numpy()

        for n, p in model.named_parameters():
            if p.grad is not None and p.requires_grad:
                saved_grads[n].data += p.grad.data ** 2 / train_examples_size[i_task]
                # saved_grads[n].data += p.grad.data.cpu() ** 2 / subtask_size[i_task]

        saved_grads = {n: p for n, p in saved_grads.items()}

    elif args.cont_learn_alg == "gem":

        loss = loss.mean()
        loss.backward()
        format_store_grads(model.named_parameters(),
                           grad_dims,
                           checkpoint_dir,
                           i_task)  # storing for the current task
        if i_task > 0:
            cont_learn_alg.run(i_task,
                               sample_sizes,
                               model)

        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        saved_grads = None

    else:
        loss = loss.mean()
        loss.backward()

        params = None
        saved_grads = None

    optimizer.step()

    if args.use_slots:
        return intent_loss, slot_loss, params, saved_grads

    return intent_loss, params, saved_grads


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
                   prior_adapter_up=None):

    print("Evaluating on i_task:", test_idx)
    if prior_mbert or prior_intents or prior_slots or prior_adapter_norm_before or prior_adapter_down_1 \
            or prior_adapter_up:

        model_dict = model.state_dict()

        if prior_mbert:
            print("Using prior_mbert")
            ### 1. wanted keys, values are in trans_model
            trans_model_dict = {"trans_model."+k: v for k, v in prior_mbert.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(trans_model_dict)

        if prior_intents:
            print("Using prior_intents")
            ### 1. wanted keys, values are in trans_model
            intent_classifier_dict = {"intent_classifier."+k: v for k, v in prior_intents.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(intent_classifier_dict)

        if prior_slots:
            print("Using prior_slots")
            ### 1. wanted keys, values are in trans_model
            slot_classifier_dict = {"slot_classifier."+k: v for k, v in prior_slots.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(slot_classifier_dict)

        if prior_adapter_norm_before:
            adapter_norm_before_dict = {"adapter.adapter_norm_before."+k: v for k, v in prior_adapter_norm_before.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_norm_before_dict)

        if prior_adapter_down_1:
            adapter_down_1_dict = {"adapter.adapter_down.1."+k: v for k, v in prior_adapter_down_1.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_down_1_dict)

        if prior_adapter_up:
            adapter_up_dict = {"adapter.adapter_up."+k: v for k, v in prior_adapter_up.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_up_dict)

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

        (input_ids, lengths, token_type_ids, input_masks, attention_mask, intent_labels, slot_labels, input_texts), text \
            = dataset.next_batch(1, dataset_test)

        input_ids = input_ids.cuda()
        lengths = lengths.cuda()
        input_masks = input_masks.cuda()
        intent_labels = intent_labels.cuda()
        slot_labels = slot_labels.cuda()

        if use_slots:
            intent_logits, slot_logits, intent_loss, slot_loss, loss = model(input_ids=input_ids,
                                                                             input_masks=input_masks,
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
                    print("Sent :", sents_text[i][0], " True Intent: ", intent_types[intents_true[i]],
                          " Intent Prediction :", intent_types[intents_pred[i]],
                          " True Slots: ", " ".join(slots_true[i]), " Slot Prediction:", " ".join(slots_pred[i]))

                text = sents_text[i][0] + "\t" + intent_types[intents_true[i]] + "\t" + intent_types[intents_pred[i]] \
                         + "\t" + " ".join(slots_true[i]) + "\t" + " ".join(slots_pred[i])
                writer.write(text+"\n")

    if verbose:
        print(test_idx, " -----------intents_true:", set(intents_true))
        print(test_idx, " -----------intents_pred:", set(intents_pred))

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
                    prior_adapter_up=None):

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
                             prior_adapter_up=prior_adapter_up)

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

    print(output_text_format)
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

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    order_lang_dict = {0: "high2lowlang",
                       1: "low2highlang",
                       2: "randomlang"}

    order_class_dict = {0: "high2lowclass",
                        1: "low2highclass",
                        2: "randomclass"}

    ## Setup option
    setup_opt_dir = os.path.join(args.out_dir,
                                 args.setup_opt)

    if not os.path.isdir(setup_opt_dir):
        os.mkdir(setup_opt_dir)

    ## Slot Usage
    if not args.use_slots:
        setup_opt_dir = os.path.join(setup_opt_dir,
                                     "Intents_only")

        if not os.path.isdir(setup_opt_dir):
            os.mkdir(setup_opt_dir)

    ## Trans model
    model_dir = os.path.join(setup_opt_dir,
                             args.trans_model)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    if args.random_pred:
        out_dir = os.path.join(model_dir,
                               "random_init")

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        metrics_path = os.path.join(out_dir, "metrics")
        if not os.path.isdir(metrics_path):
            os.mkdir(metrics_path)
        return out_dir, metrics_path

    if args.use_mono:
        out_dir = os.path.join(model_dir, "MONO", args.languages[0])
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        metrics_path = os.path.join(out_dir, "metrics")
        if not os.path.isdir(metrics_path):
            os.mkdir(metrics_path)

        return out_dir, metrics_path

    if args.setup_opt not in ["multi"]:
        if len(args.order_str) > 0:
            order_lang = "_".join(args.order_str)
        else:
            order_lang = order_lang_dict[args.order_lang]

        order_lang_dir = os.path.join(model_dir, order_lang)

        if not os.path.isdir(order_lang_dir):
            os.mkdir(order_lang_dir)

        model_dir = os.path.join(order_lang_dir,
                                 order_class_dict[args.order_class])

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        if args.cont_learn_alg != "vanilla":
            alg_option = args.cont_learn_alg
            if args.cont_learn_alg == "ewc":
                alg_option += "_"+str(args.old_task_prop)
                if args.use_online:
                    alg_option += "_use-online_gamma-" + str(args.gamma_ewc)
                else:
                    alg_option += "_no-online"
            model_dir = os.path.join(model_dir, alg_option)

            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

    ## SEED
    if args.freeze_bert:
        model_dir = os.path.join(model_dir, "freeze_bert")

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

    if args.freeze_linear:
        model_dir = os.path.join(model_dir, "freeze_linear")

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

    ## Multi-headed architecture
    if args.multi_head_in and not args.multi_head_out:
        model_dir = os.path.join(model_dir, "multi_head_in_"+"-".join(args.emb_enc_lang_spec))

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

    elif args.multi_head_in and args.multi_head_out:
        model_dir = os.path.join(model_dir, "multi_head_in_out")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

    elif args.multi_head_out and not args.multi_head_in:
        model_dir = os.path.join(model_dir, "multi_head_out")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

    if args.use_adapters:
        model_dir = os.path.join(model_dir, "adapters")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

    out_dir = os.path.join(model_dir, "SEED_"+str(args.seed))

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    metrics_path = os.path.join(out_dir, "metrics")
    if not os.path.isdir(metrics_path):
        os.mkdir(metrics_path)

    return out_dir, metrics_path


def name_in_list(list, name):
    flag = False
    for el in list:
        if el in name:
            flag = True
    return flag


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
                      checkpoint_dir,
                      args_save_file,
                      model_save_file,
                      optim_save_file,
                      prior_mbert, # prior options
                      prior_intents,
                      prior_slots,
                      prior_adapter_norm_before,
                      prior_adapter_down_1,
                      prior_adapter_up):

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
                    print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(step_iter,
                                                                                       intent_loss.mean(),
                                                                                       slot_loss.mean()))
            else:
                intent_loss, params, saved_grads, optimizer, model = train_outputs
                if step_iter % args.test_steps == 0:
                    print('Iter {} | Intent Loss = {:.4f} '.format(step_iter,
                                                                   intent_loss.mean()))

        print(">>>>>>> Dev Performance >>>>>")
        dev_out_path = None
        if args.save_dev_pred:
            dev_out_path = os.path.join(out_dir,
                                        "Dev_perf-Epoch_" + str(epoch) + "-train_" + train_idx)

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
                metrics[test_subtask_lang], _ = evaluate_report(dataset,
                                                                test_stream[test_idx],
                                                                best_model,
                                                                train_lang,
                                                                train_idx,
                                                                test_subtask_lang,
                                                                test_idx,
                                                                num_steps,
                                                                writer,
                                                                name="test",
                                                                out_path=os.path.join(out_dir,
                                                                                      "Test_perf-Epoch_" + str(epoch)
                                                                                      + "-train_" + train_lang
                                                                                      + "-test_" + test_subtask_lang),
                                                                verbose=args.verbose,
                                                                prior_mbert=prior_mbert[test_idx],
                                                                prior_intents=prior_intents[test_idx],
                                                                prior_slots=prior_slots[test_idx],
                                                                prior_adapter_norm_before=prior_adapter_norm_before[test_idx],
                                                                prior_adapter_down_1=prior_adapter_down_1[test_idx],
                                                                prior_adapter_up=prior_adapter_up[test_idx])

            with open(os.path.join(metrics_path,
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
                         prior_mbert,
                         prior_intents,
                         prior_slots,
                         prior_adapter_norm_before,
                         prior_adapter_down_1,
                         prior_adapter_up):

    print("------------------------------------ TESTING At the end of the training")
    metrics = {task: {} for task in test_stream} # could be either per subtask or language
    for test_idx, test_subtask_lang in enumerate(test_stream):
        metrics[test_subtask_lang], _ = evaluate_report(dataset,
                                                        test_stream[test_idx],
                                                        best_model,
                                                        train_lang,
                                                        train_idx,
                                                        test_subtask_lang,
                                                        test_idx,
                                                        num_steps,
                                                        writer,
                                                        name="test",
                                                        out_path=os.path.join(out_dir, "End_test_perf-train_"+train_lang
                                                                              + "-test_" + test_subtask_lang),
                                                        verbose=args.verbose,
                                                        prior_mbert=prior_mbert[test_idx],
                                                        prior_intents=prior_intents[test_idx],
                                                        prior_slots=prior_slots[test_idx],
                                                        prior_adapter_norm_before=prior_adapter_norm_before[test_idx],
                                                        prior_adapter_down_1=prior_adapter_down_1[test_idx],
                                                        prior_adapter_up=prior_adapter_up[test_idx])

    with open(os.path.join(metrics_path, "final_metrics_"+str(train_idx)+".pickle"), "wb") as output_file:
        pickle.dump(metrics, output_file)


def run(out_dir, metrics_path, args):
    """
    The main of training over different streams and evaluating different approaches in terms of catastrophic forgetting
    and generalizability to new classes/languages
    :param args:
    :return:
    """
    checkpoint_dir = os.path.join(out_dir, "checkpoint")
    writer = SummaryWriter(os.path.join(out_dir, 'runs'))

    args_save_file = os.path.join(checkpoint_dir, "training_args.bin")
    model_save_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    optim_save_file = os.path.join(checkpoint_dir, "optimizer.pt")

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

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
        eff_num_intents_task = [args.num_intent_tasks[i:i+args.num_intent_tasks] for i in range_intents]

    model = TransNLUCRF(args=args,
                        trans_model=model_trans,
                        num_tasks=num_tasks,
                        num_intents=num_intents,
                        eff_num_intents_task=eff_num_intents_task,
                        device=device,
                        use_multi_head_in=args.multi_head_in,
                        use_multi_head_out=args.multi_head_out,
                        adapter_layers=args.adapter_layers,
                        use_slots=args.use_slots,
                        num_slots=eff_num_slot)

    if torch.cuda.device_count() > 1:
        print("torch.cuda.device_count():", torch.cuda.device_count())
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
                                                    out_path=os.path.join(out_dir, "initial_perf.txt"),
                                                    verbose=args.verbose)

        with open(os.path.join(metrics_path, "initial_metrics.pickle"), "wb") as output_file:
            pickle.dump(metrics, output_file)

    optimizer, scheduler = set_optimizer(model)

    grad_dims = []
    for p in model.parameters():
        grad_dims.append(p.data.numel())

    ## Continuous Learning Algorithms
    if args.cont_learn_alg == "ewc":
        cont_learn_alg = EWC(device,
                             checkpoint_dir,
                             args.use_online,
                             args.gamma_ewc)

    elif args.cont_learn_alg == "gem":
        cont_learn_alg = GEM(dataset,
                             args.use_slots,
                             args.use_a_gem,
                             args.a_gem_n,
                             checkpoint_dir)
    else:
        cont_learn_alg = None

    prior_mbert = [None for _ in dataset.train_stream]
    prior_intents = [None for _ in dataset.train_stream]
    prior_slots = [None for _ in dataset.train_stream]
    prior_adapter_norm_before = [None for _ in dataset.train_stream]
    prior_adapter_down_1 = [None for _ in dataset.train_stream]
    prior_adapter_up = [None for _ in dataset.train_stream]

    if args.multi_head_in:
        if args.emb_enc_lang_spec == ["all"]:
            prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root,
                                                                                            model_name)).
                named_parameters()} for _ in dataset.train_stream]
        else:
            prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root,
                                                                                            model_name)).
                named_parameters() if name_in_list(args.emb_enc_lang_spec, k)}
                           for _ in dataset.train_stream]

    if args.multi_head_out:
        # TODO change to accommodate different numbers of intents
        prior_intents = [{k: v for k, v in nn.Linear(model.trans_model.config.hidden_size, num_intents).
            named_parameters()} for _ in dataset.train_stream]

        prior_slots = [{k: v for k, v in nn.Linear(model.trans_model.config.hidden_size, eff_num_slot).
            named_parameters()} for _ in dataset.train_stream]

    if args.use_adapters:
        prior_adapter_norm_before = [{k: v for k, v in nn.LayerNorm(768).named_parameters()}
                                     for _ in dataset.train_stream]

        prior_adapter_down_1 = [{k: v for k, v in nn.Linear(768, 768//2).named_parameters()}
                                for _ in dataset.train_stream]

        prior_adapter_up = [{k: v for k, v in nn.Linear(768//2, 768).named_parameters()}
                            for _ in dataset.train_stream]

    if args.setup_opt in ["cll", "multi-incr-cll", "cil", "cil-other", "multi-incr-cil"]:
        """ Continuous Learning Scenarios """

        sample_sizes = []
        best_saved_grads = None

        subtask_size = {i: 0 for i in range(len(dataset.train_stream))}

        best_model = None

        # Used for comparison only and to reinitialize trans_model but without using it as a reference
        original_mbert = model_trans_alias.from_pretrained(os.path.join(args.model_root, model_name))
        original_intent = copy.deepcopy(model.intent_classifier)
        original_slot = copy.deepcopy(model.slot_classifier)
        original_ada_norm_before = copy.deepcopy(model.adapter.adapter_norm_before)
        original_ada_down_1 = copy.deepcopy(model.adapter.adapter_down[1])
        original_ada_up = copy.deepcopy(model.adapter.adapter_up)

        mean_all_stream = []
        sum_all_stream = []

        for train_idx, train_subtask_lang in enumerate(train_stream):
            if train_subtask_lang["size"] == 0:
                print("Skipped subtask/language:", train_idx)
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
                    if args.freeze_bert:
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
                                        if name_in_list(args.emb_enc_subtask_spec, k)} # TODO rename this argument

                model_dict.update(trans_model_dict)

            if args.multi_head_out:
                intent_classifier_dict = {"intent_classifier."+k: v for k, v in original_intent.named_parameters()}
                model_dict.update(intent_classifier_dict)

                slot_classifier_dict = {"slot_classifier."+k: v for k, v in original_slot.named_parameters()}
                model_dict.update(slot_classifier_dict)

            if args.use_adapters:
                adapter_norm_before_dict = {"adapter.adapter_norm_before."+k: v for k, v in
                                            original_ada_norm_before.named_parameters()}
                model_dict.update(adapter_norm_before_dict)

                adapter_down_1_dict = {"adapter.adapter_down.1."+k: v for k, v in
                                       original_ada_down_1.named_parameters()}
                model_dict.update(adapter_down_1_dict)

                adapter_up_dict = {"adapter.adapter_up."+k: v for k, v in original_ada_up.named_parameters()}
                model_dict.update(adapter_up_dict)

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
                                                                           checkpoint_dir,
                                                                           args_save_file,
                                                                           model_save_file,
                                                                           optim_save_file,
                                                                           prior_mbert,  # prior options
                                                                           prior_intents,
                                                                           prior_slots,
                                                                           prior_adapter_norm_before,
                                                                           prior_adapter_down_1,
                                                                           prior_adapter_up)

            """ 2. Saving the trained weights of MBERT in each language to be used later on in testing stage"""

            if args.multi_head_in:
                mbert_task = copy.deepcopy(model.trans_model)
                prior_mbert[train_idx] = {k: v for k, v in mbert_task.named_parameters()
                                          if name_in_list(args.emb_enc_lang_spec, k)}

            if args.multi_head_out:
                intents_task = copy.deepcopy(model.intent_classifier)
                prior_intents[train_idx] = {k: v for k, v in intents_task.named_parameters()}

                slots_task = copy.deepcopy(model.slot_classifier)
                prior_slots[train_idx] = {k: v for k, v in slots_task.named_parameters()}

            if args.use_adapters:
                adapter_norm_before = copy.deepcopy(model.adapter.adapter_norm_before)
                prior_adapter_norm_before[train_idx] = {k: v for k, v in adapter_norm_before.named_parameters()}

                adapter_down_1 = copy.deepcopy(model.adapter.adapter_down[1])
                prior_adapter_down_1[train_idx] = {k: v for k, v in adapter_down_1.named_parameters()}

                adapter_up = copy.deepcopy(model.adapter.adapter_up)
                prior_adapter_up[train_idx] = {k: v for k, v in adapter_up.named_parameters()}

            if args.save_change_params:
                print("After training on language: ", train_idx)
                mean_all, sum_all = compute_change(mbert_task, original_mbert,
                                                   intents_task, original_intent,
                                                   slots_task, original_slot)
                mean_all_stream.append(mean_all)
                sum_all_stream.append(sum_all)

            print("Saving the best model at the end of the training stream for that language ....")
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
                                 prior_mbert,
                                 prior_intents,
                                 prior_slots,
                                 prior_adapter_norm_before,
                                 prior_adapter_down_1,
                                 prior_adapter_up)

        with open(os.path.join(metrics_path, "mean_all_stream_mbert.pickle"), "wb") as output_file:
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
                print("Skipped task:", i)
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
                        print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                           intent_loss.mean(),
                                                                                           slot_loss.mean()))
                    else:
                        print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

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
                                                    out_path=os.path.join(out_dir,
                                                                          "Test_perf-Epoch_"+epoch+"-train_"+str(i)+"-test_"+str(k)),
                                                    verbose=args.verbose)
                                else:
                                    print("Skipped task:", k, " in lang:", lang)

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
                                                      out_path=os.path.join(out_dir,
                                                                            "End-Test_perf-train_"+str(i)+"-test_"+str(k)),
                                                      verbose=args.verbose)
                        else:
                            print("Skipped task:", k, " in lang:", lang)

            with open(os.path.join(metrics_path, "final_metrics_"+str(i)+".pickle"), "wb") as output_file:
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
                                                                       checkpoint_dir,
                                                                       args_save_file,
                                                                       model_save_file,
                                                                       optim_save_file,
                                                                       prior_mbert,  # prior options
                                                                       prior_intents,
                                                                       prior_slots,
                                                                       prior_adapter_norm_before,
                                                                       prior_adapter_down_1,
                                                                       prior_adapter_up)

        test_at_end_training(best_model,
                             dataset,
                             test_stream,
                             train_idx,
                             train_lang,
                             num_steps,
                             writer,
                             prior_mbert,
                             prior_intents,
                             prior_slots,
                             prior_adapter_norm_before,
                             prior_adapter_down_1,
                             prior_adapter_up)


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

    setup_params.add_argument("--cil_stream_lang", help="Which lang to work on for the CIL setup if it is picked.")

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
                              nargs='+', default=[])

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
                                   type=str, default="BertBaseMultilingualCased")

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
    freezing_params.add_argument("--freeze_bert", help="Whether to freeze all layers in the Transformer encoder.",
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

    model_expansion_params.add_argument("--emb_enc_lang_spec", help="Which layer in the embeddings or the encoder "
                                                                    "to tune for each language independently.",
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

    model_expansion_params.add_argument("--adapter_layers", help="List of layers to which adapters are applied.",
                                        nargs="+", default=[0, 1, 2, 3, 4, 5])

    ## MODEL EXPANSION OPTIONS
    cont_learn_params = parser.add_argument_group("Continuous Learning Options")
    cont_learn_params.add_argument("--cont_learn_alg", help="vanilla fine-tuning or some continuous learning algorithm:"
                                                            "(ewc, gem, mbpa, metambpa, etc) or vanilla if no specific"
                                                            "continuous learning algorithm is used.",
                                   choices=["vanilla", "ewc", "gem"],  # TODO to be covered next "mbpa", "metambpa", "icarl", "xdg", "si", "lwf", "gr", "rtf", "er"
                                   type=str, default="vanilla")

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

    if args.use_adapters:
        from downstreammodels.transNLUCRFAdapters import TransNLUCRF
    else:
        from downstreammodels.transNLUCRF import TransNLUCRF

    out_dir, metrics_path = set_out_dir()

    if args.no_debug:
        stdoutOrigin = sys.stdout
        print("out_dir: ", out_dir)

        sys.stdout = open(os.path.join(out_dir, args.log_file), "w")
        logstats.init(os.path.join(out_dir, args.stats_file))
        config_path = os.path.join(out_dir, 'config.json')
        logstats.add_args('config', args)
        logstats.write_json(vars(args), config_path)

    run(out_dir, metrics_path, args)

    if args.no_debug:
        sys.stdout.close()
        sys.stdout = stdoutOrigin

