from data_utils import *
import argparse
from consts import intent_types, slot_types
import gc
import numpy as np
from downstreammodels.transNLUCRF import TransNLUCRF
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
from contlearnalg.GEM import overwrite_grad, project2cone2
from downstreammodels.crf import CRFLayer
import nvidia_smi
import copy

gpus_list = list(range(torch.cuda.device_count()))


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


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


def store_grads(pp,
                grads,
                grad_dims,
                tid,
                checkpoint_dir):
    """
        This stores parameter gradients of one task at a time.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads.fill_(0.0)
    cnt = 0
    for n, p in pp:
        if p.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(p.grad.data.view(-1))
        cnt += 1

    with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(tid)), "wb") as file:
        pickle.dump(grads, file)


def train(optimizer,
          model,
          checkpoint_dir,
          dataset,
          subtask,
          subtask_size,
          writer,
          epoch,
          i_task,
          num_steps,
          old_dataset=None,
          sample_sizes=[]):

    optimizer.zero_grad()
    model.train()
    # Take batch by batch and move to cuda
    batch, _ = dataset.next_batch(args.batch_size, subtask)

    input_ids, lengths, token_type_ids, input_masks, attention_mask, intent_labels, slot_labels, \
        input_texts = batch

    input_ids = input_ids.cuda()
    lengths = lengths.cuda()
    token_type_ids = token_type_ids.cuda()
    input_masks = input_masks.cuda()
    attention_mask = attention_mask.cuda()
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
            # the list of previous tasks
            # batches_list = []
            # for sub_task_i in range(i_task):
            #     for _ in range(sample_sizes[sub_task_i]):
            #         batch, _ = dataset.next_batch(1, old_dataset[sub_task_i]["stream"])
            #         batches_list.append(batch)
            #
            # ewc = EWC(model,
            #           batches_list,
            #           args.use_slots,
            #           device,
            #           args.use_online)
            # print((args.ewc_lambda / 2) * ewc.penalty(model))

            reg_term = 0
            if args.use_online:
                fisher_param = {n: 0 for n, p in model.named_parameters() if p.requires_grad}
                for fj in range(1, i_task):
                    with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(i_task-fj)), "rb") as file:
                        grads = pickle.load(file)

                    for n, p in model.named_parameters():
                        grad_k = grads[n]
                        fisher_param[n] += (args.gamma_ewc ** (fj-1)) * grad_k
                        del grad_k

                with open(os.path.join(checkpoint_dir, "pytorch_params_"+str(i_task-1)), "rb") as file:
                    params = pickle.load(file)

                for n, p in model.named_parameters():
                    p_k = params[n]
                    _reg_term = fisher_param[n] * (p - p_k) ** 2
                    del p_k
                    reg_term += _reg_term.sum()
            else:
                for k in range(0, i_task):
                    with open(os.path.join(checkpoint_dir, "pytorch_params_"+str(k)), "rb") as file:
                        params = pickle.load(file)

                    with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(k)), "rb") as file:
                        grads = pickle.load(file)

                    for n, p in model.named_parameters():
                        torch.cuda.empty_cache()
                        if p.grad is not None and p.requires_grad:
                            p_k = params[n]
                            grad_k = grads[n]
                            _reg_term = grad_k * (p - p_k) ** 2
                            del grad_k
                            del p_k
                            reg_term += _reg_term.sum()

            print("i_task: ", i_task, " reg_term:", reg_term, " reg_term.grad:", reg_term.grad)

            loss += (args.ewc_lambda / 2) * reg_term

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
                saved_grads[n].data += p.grad.data ** 2 / subtask_size[i_task]
                # saved_grads[n].data += p.grad.data.cpu() ** 2 / subtask_size[i_task]

        saved_grads = {n: p for n, p in saved_grads.items()}

    elif args.cont_learn_alg == "gem":
        grad_dims = []
        for p in model.parameters():
            grad_dims.append(p.data.numel())

        grads = torch.Tensor(sum(grad_dims))

        loss = loss.mean()
        loss.backward()
        store_grads(model.named_parameters(), grads, grad_dims, i_task, checkpoint_dir) # storing for the current task
        if i_task > 0:
            # forward pass on the previous data stored in the memory
            for old_task_i in range(i_task): # for all old tasks before current task i

                for _ in range(sample_sizes[old_task_i]):
                    model.zero_grad()
                    batch, _ = dataset.next_batch(1, old_dataset[old_task_i]["stream"])

                    input_ids, lengths, token_type_ids, input_masks, attention_mask, intent_labels, slot_labels, \
                    input_texts = batch

                    input_ids = input_ids.cuda()
                    lengths = lengths.cuda()
                    input_masks = input_masks.cuda()
                    intent_labels = intent_labels.cuda()
                    slot_labels = slot_labels.cuda()

                    if args.use_slots:
                        logits_intents, logits_slots, intent_loss, slot_loss, loss = model(input_ids=input_ids,
                                                                                           input_masks=input_masks,
                                                                                           lengths=lengths,
                                                                                           intent_labels=intent_labels,
                                                                                           slot_labels=slot_labels)

                    else:
                        logits_intents, intent_loss, loss = model(input_ids=input_ids,
                                                                  input_masks=input_masks,
                                                                  lengths=lengths,
                                                                  intent_labels=intent_labels)

                    loss.backward()
                # Store the gradients for the current step
                store_grads(model.named_parameters(), grads, grad_dims, old_task_i, checkpoint_dir)

            # Solve the dual of the quadratic equation # TODO double check if this is done only once at the end of all
            indx = torch.LongTensor(list(range(i_task))).cuda() if torch.cuda.device_count() > 0 \
                else torch.LongTensor(list(range(i_task)))

            print("index:", indx, "index.data.cpu():", indx.data.cpu().numpy()[0])
            with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(indx.data.cpu().numpy()[0])), "rb") as file:
                grads_old = pickle.load(file).cuda()

            with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(i_task)), "rb") as file:
                grads = pickle.load(file).cuda()

            print("grads_old:", grads_old.shape, "grads_old:", grads_old.unsqueeze(1))
            print("grads.unsqueeze(0):", grads.unsqueeze(0).shape, " grads:", grads.unsqueeze(0))

            dotp = torch.mm(grads.unsqueeze(0),
                            grads_old.unsqueeze(1))#grads.index_select(1, indx))

            print("Computed multiplication")

            # check if the constraints have been violated
            if (dotp < 0).sum() != 0:
                # If it is the case:
                # Copy the gradients

                print("Project2cone2")
                project2cone2(grads.unsqueeze(1),
                              grads_old.unsqueeze(1))#grads.index_select(1, indx))
                print("Overwriting grads")

                # Update the named_parameters accordingly
                overwrite_grad(model.named_parameters(),
                               grads,
                               grad_dims)
                print("Finished")

        saved_grads = grads
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    else:
        loss = loss.mean()

        loss.backward()
        params = None
        saved_grads = None

    optimizer.step()

    if args.use_slots:
        return intent_loss, slot_loss, params, saved_grads
    else:
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
                   prior_slots=None):

    print("Evaluating on i_task:", test_idx)
    if prior_mbert or prior_intents or prior_slots:
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
                    train_lang,
                    train_idx,
                    test_lang,
                    test_idx,
                    num_steps,
                    writer,
                    name,
                    out_path=None,
                    verbose=False,
                    prior_mbert=None,
                    prior_intents=None,
                    prior_slots=None):

    outputs = nlu_evaluation(dataset,
                             data_stream["stream"],
                             data_stream["size"],
                             model,
                             args.use_slots,
                             test_idx,
                             out_path=out_path,
                             verbose=verbose,
                             prior_mbert=prior_mbert,
                             prior_intents=prior_intents,
                             prior_slots=prior_slots)

    output_text_format = "----size=%d, test_index=%d, and lang=%s" % (data_stream["size"],
                                                                      test_idx,
                                                                      test_lang)

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

        metrics.update({train_lang+'_'+str(train_idx)+'_'+name+'_slot_prec_'+test_lang+'_'+str(test_idx): slot_prec})
        metrics.update({train_lang+'_'+str(train_idx)+'_'+name+'_slot_rec_'+test_lang+'_'+str(test_idx): slot_rec})
        metrics.update({train_lang+'_'+str(train_idx)+'_'+name+'_slot_f1_'+test_lang+'_'+str(test_idx): slot_f1})

    metrics.update({train_lang+'_'+str(train_idx)+'_'+name+'_intent_acc_'+test_lang+'_'+str(test_idx): intent_acc})
    metrics.update({train_lang+'_'+str(train_idx)+'_'+name+'_intent_prec_'+test_lang+'_'+str(test_idx): intent_prec})
    metrics.update({train_lang+'_'+str(train_idx)+'_'+name+'_intent_rec_'+test_lang+'_'+str(test_idx): intent_rec})
    metrics.update({train_lang+'_'+str(train_idx)+'_'+name+'_intent_f1_'+test_lang+'_'+str(test_idx): intent_f1})

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
        mono_lang_dir = os.path.join(model_dir, "MONO")
        if not os.path.isdir(mono_lang_dir):
            os.mkdir(mono_lang_dir)

        out_dir = os.path.join(mono_lang_dir, args.languages[0])
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        metrics_path = os.path.join(out_dir, "metrics")
        if not os.path.isdir(metrics_path):
            os.mkdir(metrics_path)

        return mono_lang_dir, metrics_path

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


def run(out_dir, metrics_path, args):
    """
    The main of training over different streams and evaluating different approaches in terms of catastrophic forgetting
    and generalizability to new classes/languages
    :param args:
    :return:
    """
    checkpoint_dir = os.path.join(out_dir, "checkpoint")
    args_save_file = os.path.join(checkpoint_dir, "training_args.bin")
    model_save_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    optim_save_file = os.path.join(checkpoint_dir, "optimizer.pt")

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    writer = SummaryWriter(os.path.join(out_dir, 'runs'))

    model_name, tokenizer_alias, model_trans_alias = MODELS_dict[args.trans_model]

    if "mmhamdi" in args.data_root or "jonmay_231" in args.data_root:
        tokenizer = tokenizer_alias.from_pretrained(os.path.join(args.model_root, model_name),
                                                    do_lower_case=True,
                                                    do_basic_tokenize=False)

        model_trans = model_trans_alias.from_pretrained(os.path.join(args.model_root, model_name))
    else:
        tokenizer = tokenizer_alias.from_pretrained(model_name,
                                                    do_lower_case=True,
                                                    do_basic_tokenize=False)

        model_trans = model_trans_alias.from_pretrained(model_name)

    dataset = Dataset(args.data_root,
                      args.setup_opt,
                      args.setup_3,
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

    # # args.setup_opt => ['multi', 'multi-incremental-lang', 'cll', 'cil', 'cil-other', 'cil-ll']
    # eff_num_intent = len(dataset.intent_types)
    # eff_num_slot = len(dataset.slot_types)
    #
    # # Incrementally adding new intents as they are added to the setup
    # if args.setup_opt in ["cil", "cil-ll"]:
    #     eff_num_intent = len(dataset.intent_types) #args.num_intent_tasks
    #     eff_num_slot = len(dataset.slot_types)  # to be changed later
    #     num_tasks = range(0, len(dataset.intent_types), args.num_intent_tasks) # TODO to be changed later for cil-ll
    #     # TODO to be checked for cil-other
    # else:
    #     if args.setup_opt == "multi":
    #         num_tasks = 1
    #     elif args.setup_opt in ["cll", "multi-incremental-lang"]:
    #         if len(args.order_str) > 0:
    #             num_tasks = len(args.order_str)
    #         else:
    #             num_tasks = len(args.languages)

    """ eff_num_intent/eff_num_slot """
    if args.setup_opt == "cil-other":
        eff_num_intent += 1
        num_intents += 1

    """ num_tasks"""

    if args.setup_opt == "multi":
        num_tasks = 1
        eff_num_intents_task = eff_num_intent
    elif args.setup_opt in ["multi-incremental-lang", "cll"]:
        if len(args.order_str) > 0:
            num_tasks = len(args.order_str)
        else:
            num_tasks = len(args.languages)
        eff_num_intents_task = eff_num_intent
    else: # args.setup_opt in ["cil", "cil-other", "cil-ll"]:
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

    optimizer, scheduler = set_optimizer(model, args)

    if args.setup_opt == "cil":
        """ 
        Setup 1: Cross-CIL, Fixed LL: "Monolingual CIL":
        - Trnt(ain over every task of classes continuously independently for every language. 
        - We then average over all languages.
        """
        for lang in args.languages:
            optimizer, scheduler = set_optimizer(model, args)
            sample_sizes = []
            # Iterating over the stream
            params_old_tasks = {i: {} for i, subtask in enumerate(dataset.train_stream[lang])}
            grads_old_tasks = {i: {} for i, subtask in enumerate(dataset.train_stream[lang])}
            for i, subtask in enumerate(dataset.train_stream[lang]):
                sample_sizes.append(int(subtask["size"]*args.old_task_prop))
                num_steps = 0
                num_iter = subtask["size"] // args.batch_size
                stream = subtask["stream"]
                if subtask["size"] == 0:
                    print("Skipped task:", i, " in lang:", lang)
                    continue

                for epoch in tqdm(range(args.epochs)):
                    gc.collect()
                    num_steps += 1

                    for j in range(num_iter):
                        if args.use_slots:
                            intent_loss, slot_loss = train(params_old_tasks,
                                                           grads_old_tasks,
                                                           args,
                                                           optimizer,
                                                           model,
                                                           dataset,
                                                           stream,
                                                           subtask["size"],
                                                           writer,
                                                           epoch,
                                                           i,
                                                           j,
                                                           old_dataset=dataset.train_stream[lang], # all dataset to get old stream
                                                           sample_sizes=sample_sizes)
                        else:
                            intent_loss = train(params_old_tasks,
                                                grads_old_tasks,
                                                args,
                                                optimizer,
                                                model,
                                                dataset,
                                                stream,
                                                subtask["size"],
                                                writer,
                                                epoch,
                                                i,
                                                j,
                                                old_dataset=dataset.train_stream[lang], # all dataset to get old stream
                                                sample_sizes=sample_sizes)

                        if args.use_slots:
                            print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                               intent_loss.mean(),
                                                                                               slot_loss.mean()))
                        else:
                            print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

                params = {n: p for n, p in model.named_parameters() if p.requires_grad} # current task parameters to be saved

                _means = {n: {} for n, p in model.named_parameters() if p.requires_grad}

                for n, p in deepcopy(params).items():
                    _means[n] = variable(p.data) # Previous task parameters

                params_old_tasks[i] = _means

                grads_old_tasks[i] = {n: p.grad.data ** 2 /subtask["size"] for n, p in model.named_parameters() if p.requires_grad} # current task parameters to be saved

                metrics = {k: {} for k in range(0, i+1)}

                for k in range(0, i+1):
                    if dataset.test_stream[lang][k]["size"] > 0:
                        metrics[k] = evaluate_report(dataset,
                                                     dataset.test_stream[lang][k],
                                                     model,
                                                     lang,
                                                     i,
                                                     lang,
                                                     k,
                                                     num_steps,
                                                     writer,
                                                     name="test",
                                                     out_path=os.path.join(out_dir,
                                                                           lang+"_test_perf-train_"+str(i)+"-test_"
                                                                           + str(k)),
                                                     verbose=args.verbose)
                    else:
                        print("Skipped task:", k, " in lang:", lang)

                with open(os.path.join(metrics_path, "final_metrics_"+lang+"_trainsubtask_"+str(i)+".pickle"), "wb") as output_file:
                    pickle.dump(metrics, output_file)

            print("------------------------------------")
            metrics = {k: {} for k in range(0, len(dataset.train_stream[lang]))}
            for k in range(0, len(dataset.train_stream[lang])):
                if dataset.test_stream[lang][k]["size"] > 0:
                    metrics[k] = evaluate_report(dataset,
                                                 dataset.test_stream[lang][k],
                                                 model,
                                                 lang,
                                                 i,
                                                 lang,
                                                 k,
                                                 num_steps,
                                                 writer,
                                                 name="test",
                                                 out_path=os.path.join(out_dir,
                                                                       lang+"End_perf-train_"+str(i)+"-test_"+str(k)),
                                                 verbose=args.verbose)
                else:
                    print("Skipped task:", k, " in lang:", lang)

            with open(os.path.join(metrics_path, "final_metrics_"+lang+".pickle"), "wb") as output_file:
                pickle.dump(metrics, output_file)
            print("/////////////////////////////////////////////")

    elif args.setup_opt == "cil-other":
        """ 
        Setup 2: CIL with other option:  incremental version of cil where previous intents' subtasks are added 
            in addition to other labels for subsequent intents' subtasks.
        """

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.cuda()

        optimizer, scheduler = set_optimizer(model, args)

        _lambda = 20
        for l, lang in enumerate(args.languages):
            sample_sizes = []
            # Iterating over the stream
            params_old_tasks = {i: {} for i, subtask in enumerate(dataset.train_stream[lang])}
            grads_old_tasks = {i: {} for i, subtask in enumerate(dataset.train_stream[lang])}
            for i, subtask in enumerate(dataset.train_stream[lang]):
                sample_sizes.append(subtask["size"] // 10)
                num_steps = 0
                num_iter = subtask["size"] // args.batch_size
                stream = subtask["stream"]
                if subtask["size"] == 0:
                    print("Skipped task:", i, " in lang:", lang)
                    continue

                for epoch in tqdm(range(args.epochs)):
                    gc.collect()
                    num_steps += 1

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
                                                           old_dataset=dataset.train_stream[lang], # all dataset to get old stream
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
                                                old_dataset=dataset.train_stream[lang], # all dataset to get old stream
                                                sample_sizes=sample_sizes,
                                                _lambda=_lambda)

                        if args.use_slots:
                            print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                               intent_loss.mean(),
                                                                                               slot_loss.mean()))
                        else:
                            print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

                metrics = {k: {} for k in range(0, i+1)}
                for k in range(0, i+1):
                    if dataset.test_stream[lang][k]["size"] > 0:
                        metrics[k] = evaluate_report(dataset.test_stream[lang][k],
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
                                                                           lang+"Test_perf-train_"+str(i)+"-test_"+str(k)),
                                                     verbose=args.verbose)
                    else:
                        print("Skipped task:", k, " in lang:", lang)

                with open(os.path.join(metrics_path, "final_metrics_"+lang+"_trainsubtask_"+str(i)+".pickle"), "wb") as output_file:
                    pickle.dump(metrics, output_file)

            print("------------------------------------")
            metrics = {k: {} for k in range(0, len(dataset.train_stream[lang]))}
            for k in range(0, len(dataset.train_stream[lang])):
                if dataset.test_stream[lang][k]["size"] > 0:
                    metrics[k] = evaluate_report(dataset.test_stream[lang][k],
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
                                                                       lang+"End_perf-train_"+str(i)+"-test_"+str(k)),
                                                 verbose=args.verbose)
                else:
                    print("Skipped task:", k, " in lang:", lang)

            with open(os.path.join(metrics_path, "final_metrics_"+lang+".pickle"), "wb") as output_file:
                pickle.dump(metrics, output_file)
            print("/////////////////////////////////////////////")

    elif args.setup_opt in ["cll", "multi-incremental-lang"]:
        """
        Setup 3: Cross-LL, Fixed CIL: "Conventional Cross-lingual Transfer Learning or Stream learning" 
        - Stream consisting of different combinations of languages.
        => Each stream sees all intents
        """
        # Iterating over the stream of languages

        sample_sizes = []
        best_saved_grads = None

        subtask_size = {i: 0 for i in range(len(dataset.train_stream))}

        best_model = None

        if args.multi_head_in:
            if args.emb_enc_lang_spec == ["all"]:
                prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root, model_name))
                    .named_parameters()} for _ in dataset.train_stream]
            else:
                prior_mbert = [{k: v for k, v in model_trans_alias.from_pretrained(os.path.join(args.model_root, model_name))
                    .named_parameters() if name_in_list(args.emb_enc_lang_spec, k)} for _ in dataset.train_stream]

        else:
            prior_mbert = [None for _ in dataset.train_stream]

        if args.multi_head_out:
            prior_intents = [{k: v for k, v in nn.Linear(model.trans_model.config.hidden_size, num_intents)
                .named_parameters()} for _ in dataset.train_stream]

            prior_slots = [{k: v for k, v in nn.Linear(model.trans_model.config.hidden_size, eff_num_slot)
                .named_parameters()} for _ in dataset.train_stream]

            #prior_crf = [CRFLayer(eff_num_slot, device) for _ in dataset.train_stream]
        else:
            prior_intents = [None for _ in dataset.train_stream]
            prior_slots = [None for _ in dataset.train_stream]
            #prior_crf = [None for _ in dataset.train_stream]

        # Used for comparison only and to reinitialize trans_model but without using it as a reference
        original_mbert = model_trans_alias.from_pretrained(os.path.join(args.model_root, model_name))
        original_intent = copy.deepcopy(model.intent_classifier)
        original_slot = copy.deepcopy(model.slot_classifier)
        #original_crf = copy.deepcopy(model.crf_layer)

        mean_all_stream = []
        sum_all_stream = []

        for train_idx, subtask in enumerate(dataset.train_stream):
            dev_perf_best = 0.0
            sample_sizes.append(int(subtask["size"]*args.old_task_prop))
            num_steps = 0
            num_iter = subtask["size"]//args.batch_size
            train_lang = subtask["lang"]
            stream = subtask["stream"]
            if args.first_lang:
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

            subtask_size[train_idx] = subtask["size"]

            """ 1. Reinitialize linear layers for each new language """
            if args.multi_head_in or args.multi_head_out:
                model_dict = model.state_dict()

            if args.multi_head_in:
                if args.emb_enc_lang_spec == ["all"]:
                    trans_model_dict = {"trans_model."+k: v for k, v in original_mbert.named_parameters()}
                else:
                    trans_model_dict = {"trans_model."+k: v for k, v in original_mbert.named_parameters()
                                        if name_in_list(args.emb_enc_lang_spec, k)}

                model_dict.update(trans_model_dict)

            if args.multi_head_out:
                intent_classifier_dict = {"intent_classifier."+k: v for k, v in original_intent.named_parameters()}
                model_dict.update(intent_classifier_dict)

                slot_classifier_dict = {"slot_classifier."+k: v for k, v in original_slot.named_parameters()}
                model_dict.update(slot_classifier_dict)

                # crf_dict = {k: v for k, v in original_crf.named_parameters()}
                # model_dict.update(crf_dict)

            if args.multi_head_in or args.multi_head_out:
                model.load_state_dict(model_dict)

            for epoch in tqdm(range(args.epochs)):
                gc.collect()
                num_steps += 1
                for step_iter in range(num_iter):
                    train_outputs = train(optimizer,
                                          model,
                                          checkpoint_dir,
                                          dataset,
                                          stream,
                                          subtask_size,
                                          writer,
                                          epoch,
                                          train_idx,
                                          step_iter,
                                          old_dataset=dataset.train_stream, # all dataset to get old stream
                                          sample_sizes=sample_sizes)

                    if args.use_slots:
                        intent_loss, slot_loss, params, saved_grads = train_outputs
                        if step_iter % args.eval_steps == 0:
                            print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(step_iter,
                                                                                               intent_loss.mean(),
                                                                                               slot_loss.mean()))
                    else:
                        intent_loss, params, saved_grads = train_outputs
                        if j % args.eval_steps == 0:
                            print('Iter {} | Intent Loss = {:.4f} '.format(step_iter,
                                                                           intent_loss.mean()))

                print(">>>>>>> Dev Performance >>>>>")
                dev_out_path = None
                if args.save_dev_pred:
                    dev_out_path = os.path.join(out_dir,
                                                "Dev_perf-Epoch_" + str(epoch) + "-train_" + train_lang)

                _, dev_perf = evaluate_report(dataset,
                                              dataset.dev_stream[train_idx],
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
                    metrics = {lang: {} for lang in dataset.test_stream}

                    for test_idx, test_lang in enumerate(dataset.test_stream):
                        metrics[test_lang], _ = evaluate_report(dataset,
                                                                dataset.test_stream[test_lang],
                                                                best_model,
                                                                train_lang,
                                                                train_idx,
                                                                test_lang,
                                                                test_idx,
                                                                num_steps,
                                                                writer,
                                                                name="test",
                                                                out_path=os.path.join(out_dir,
                                                                                      "Test_perf-Epoch_" + str(epoch)
                                                                                      + "-train_" + train_lang
                                                                                      + "-test_" + test_lang),
                                                                verbose=args.verbose,
                                                                prior_mbert=prior_mbert[test_idx],
                                                                prior_intents=prior_intents[test_idx],
                                                                prior_slots=prior_slots[test_idx])

                    with open(os.path.join(metrics_path,
                                           "epoch_"+str(epoch)+"_metrics_"+str(train_idx)+".pickle"), "wb") \
                            as output_file:
                        pickle.dump(metrics, output_file)

            """ 2. Saving the trained weights of MBERT in each language to be used later on in testing stage"""

            mbert_task = copy.deepcopy(model.trans_model)
            intents_task = copy.deepcopy(model.intent_classifier)
            slots_task = copy.deepcopy(model.slot_classifier)

            if args.multi_head_in:
                prior_mbert[train_idx] = {k: v for k, v in mbert_task.named_parameters()
                                          if name_in_list(args.emb_enc_lang_spec, k)}

            if args.multi_head_out:
                prior_intents[train_idx] = {k: v for k, v in intents_task.named_parameters()}
                prior_slots[train_idx] = {k: v for k, v in slots_task.named_parameters()}

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

            print("------------------------------------ TESTING At the end of the training")
            metrics = {lang: {} for lang in dataset.test_stream}
            for test_idx, test_lang in enumerate(dataset.test_stream):
                metrics[test_lang], _ = evaluate_report(dataset,
                                                        dataset.test_stream[test_lang],
                                                        best_model,
                                                        train_lang,
                                                        train_idx,
                                                        test_lang,
                                                        test_idx,
                                                        num_steps,
                                                        writer,
                                                        name="test",
                                                        out_path=os.path.join(out_dir, "End_test_perf-train_"+train_lang
                                                                               + "-test_" + test_lang),
                                                        verbose=args.verbose,
                                                        prior_mbert=prior_mbert[test_idx],
                                                        prior_intents=prior_intents[test_idx],
                                                        prior_slots=prior_slots[test_idx])

            with open(os.path.join(metrics_path, "final_metrics_"+str(train_idx)+".pickle"), "wb") as output_file:
                pickle.dump(metrics, output_file)

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
                stream = subtask["stream"]

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

                    if j % args.eval_steps == 0:
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
        for epoch in tqdm(range(args.epochs)):
            gc.collect()

            num_steps = 0
            # There is only one task here no subtasks
            task = dataset.train_stream["stream"]
            num_steps += 1

            num_iter = dataset.train_stream["size"]//args.batch_size

            for j in range(num_iter):

                if args.use_slots:
                    intent_loss, slot_loss = train(args,
                                                   optimizer,
                                                   model,
                                                   dataset,
                                                   task,
                                                   writer,
                                                   epoch,
                                                   0,
                                                   j)
                else:
                    intent_loss = train(args,
                                        optimizer,
                                        model,
                                        dataset,
                                        task,
                                        writer,
                                        epoch,
                                        0,
                                        j)

                if args.use_slots:
                    print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                       intent_loss.mean(),
                                                                                       slot_loss.mean()))
                else:
                    print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

            metrics = {lang: {} for lang in dataset.test_stream}
            for lang in dataset.test_stream:
                metrics[lang] = evaluate_report(dataset.test_stream[lang],
                                                0,
                                                "all",
                                                lang,
                                                args,
                                                dataset,
                                                model,
                                                writer,
                                                0,
                                                num_steps,
                                                out_path=os.path.join(out_dir,
                                                                      "Test_perf-Epoch_"+epoch+ "-test_"+lang),
                                                verbose=args.verbose)

            with open(os.path.join(metrics_path, "epoch_"+str(epoch)+"_metrics.pickle"), "wb") as output_file:
                pickle.dump(metrics, output_file)

        print("Evaluation at the end of all epochs")
        metrics = {lang: {} for lang in dataset.test_stream}
        for lang in dataset.test_stream:
            metrics[lang] = evaluate_report(dataset.test_stream[lang],
                                            0,
                                            "all",
                                            lang,
                                            args,
                                            dataset,
                                            model,
                                            writer,
                                            0,
                                            num_steps,
                                            out_path=os.path.join(out_dir, "End_perf-"+lang),
                                            verbose=args.verbose)

        with open(os.path.join(metrics_path, "final_metrics.pickle"), "wb") as output_file:
            pickle.dump(metrics, output_file)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## PATHS
    parser.add_argument("--data_root", help="Root directory of the data",
                        type=str, default="")

    parser.add_argument("--model_root", help="Path to the root directory hosting the trans model if offline",
                        type=str, default="")

    parser.add_argument("--out_dir", help="The root directory of the results for this project",
                        type=str, default="")

    parser.add_argument("--stats_file", help="Filename of the stats file",
                        type=str, default="stats.txt")

    parser.add_argument("--log_file", help="Filename of the log file",
                        type=str, default="log.txt")

    ## SETUP OPTION
    parser.add_argument("--setup_opt", help="Whether to pick setup "
                                            "   cil: Cross-CIL with fixed LL, "
                                            "   cil-other: incremental version of cil where previous intents' subtasks "
                                            "              are added in addition to other labels for subsequent "
                                            "              intents'subtasks, "
                                            "   cll: Cross-LL with fixed CIL,"
                                            "   cil-ll: Cross-CIL-LL,"
                                            "   multi: multi-tasking one model on all tasks and langs,"
                                            "   multi-incremental-lang: weaker version of Multi-task/Joint Learning "
                                            "                           where we gradually fine-tune",
                        choices=['cil', 'cil-other', 'cll', 'cil-ll', 'multi', 'multi-incremental-lang'],
                        type=str, default="cll")

    parser.add_argument("--order_class", help="Different ways of ordering the classes"
                                              "0: decreasing order (from high to low-resource), "
                                              "1: increasing order (from low to high-resource),"
                                              "2: random order",
                        type=int, default=0)

    parser.add_argument("--order_lang", help="Different ways of ordering the languages"
                                             "0: decreasing order (from high to low-resource) , "
                                             "1: increasing order (from low to high-resource),"
                                             "2: random order",
                        type=int, default=0)

    parser.add_argument("--order_str", help="specific order for languages",
                        nargs='+', default=[])

    parser.add_argument("--setup_3", help="intents: traversing subtasks horizontally over all classes first then "
                                          "         to languages,"
                                          "langs: traversing subtasks vertically over all languages first then "
                                          "         to classes",
                        type=str, default="intents")

    parser.add_argument('--use_slots', help='If true, optimize for slot filling loss too',
                        action='store_true')

    parser.add_argument('--verbose', help='If true, return golden labels and predictions ',
                        action='store_true')

    parser.add_argument('--save_dev_pred', help='If true, save dev perf ',
                        action='store_true')

    parser.add_argument('--save_test_every_epoch', help='If true, save test at the end of each epoch ',
                        action='store_true')

    parser.add_argument('--save_change_params', help='If true, save test at the end of each epoch ',
                        action='store_true')

    parser.add_argument('--no_debug', help='If true, return golden labels and predictions ',
                        action='store_true')

    parser.add_argument('--random_pred', help='Whether to predict directly the random initialization of the model when'
                                              'tested directly on the languages without any fine-tuning',
                        action='store_true')

    ## TRAINING OPTIONS
    parser.add_argument("--trans_model", help="Name of transformer model",
                        type=str, default="BertBaseMultilingualCased")

    parser.add_argument("--freeze_bert", help="Whether to freeze all layers in bert after first language",
                        action='store_true')

    parser.add_argument("--first_lang", help="Whether to freeze from the first language",
                        action='store_true')

    parser.add_argument("--freeze_linear", help="Whether to freeze all task-specific layers after first language",
                        action='store_true')

    parser.add_argument("--use_mono", help="Whether to train monolingually",
                        action='store_true')

    parser.add_argument("--multi_head_in", help="Whether to use multiple heads in the input of inputs that would"
                                                "imply the separation of the vocabulary of M-BERT",
                        action='store_true')

    parser.add_argument("--emb_enc_lang_spec", help="Which layer in the embeddings or the encoder to tune for each"
                                                    "language independently.",
                        nargs="+", default=["embeddings"],
                        choices=["embeddings",
                                 "encoder.layer.0",
                                 "encoder.layer.1",
                                 "encoder.layer.2",
                                 "encoder.layer.3",
                                 "encoder.layer.4",
                                 "encoder.layer.5",
                                 "encoder.layer.6",
                                 "encoder.layer.7",
                                 "encoder.layer.8",
                                 "encoder.layer.9",
                                 "encoder.layer.10",
                                 "encoder.layer.11",
                                 "pooler",
                                 "all"])

    parser.add_argument("--multi_head_out", help="Whether to use multiple heads in the outputs that would"
                                                 "imply the use of different linear layers",
                        action='store_true')

    parser.add_argument('--data_format', help='Whether it is tsv (MTOD), json, or txt (MTOP)',
                        type=str, default="txt")

    parser.add_argument("--languages", help="train languages list",
                        nargs="+", default=["de", "en", "es", "fr", "hi", "th"])

    parser.add_argument("--num_intent_tasks", help="The number of intent per task",
                        type=int, default=10)

    parser.add_argument("--num_lang_tasks", help="The number of lang per task",
                        type=int, default=2)

    parser.add_argument("--epochs", help="The total number of epochs",
                        type=int, default=10)

    parser.add_argument("--eval_steps", help="The total number of epochs for the model to evaluate (test mode)",
                        type=int, default=200)

    parser.add_argument("--dev_steps", help="The total number of epochs for the model to evaluate (dev mode)",
                        type=int, default=200)

    parser.add_argument("--batch_size", help="The total number of epochs for the model to evaluate",
                        type=int, default=32)

    parser.add_argument("--step_size", help="The step size for the scheduler",
                        type=int, default=7)

    parser.add_argument("--gamma", help="gamma for the scheduler",
                        type=float, default=0.1)

    parser.add_argument("--adam_lr", help="The learning rate",
                        type=float, default=1e-03)

    parser.add_argument("--adam_eps", help="epsilon",
                        type=float, default=1e-08)

    parser.add_argument("--beta_1", help="beta_1 for Adam",
                        type=float, default=0.9)

    parser.add_argument("--beta_2", help="beta_2 for Adam",
                        type=float, default=0.99)

    parser.add_argument("--seed", help="The total number of epochs",
                        type=int, default=42)

    parser.add_argument("--cont_learn_alg", help="vanilla fine-tuning or some continuous learning algorithm "
                                                 "(ewc, gem, mbpa, metambpa, etc)",
                        type=str, default="vanilla")

    parser.add_argument("--old_task_prop", help="the percentage of old tasks used for replay or regularization",
                        type=float, default=1.0)

    parser.add_argument("--gamma_ewc", help="percentage of decay",
                        type=int, default=0.01)

    parser.add_argument("--use_online", help="Whether to use the online version of EWC or not",
                        action='store_true')

    parser.add_argument("--ewc_lambda", help="lambda for regularization in ewc",
                        type=int, default=20)

    args = parser.parse_args()
    set_seed()

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

