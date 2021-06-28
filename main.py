from pytorch_transformers import AdamW, WarmupLinearSchedule
from data_utils import *
import argparse
from consts import domain_types, intent_types, slot_types
import gc
import numpy as np
from models.transNLU import TransNLU
from models.transNLUCRF import TransNLUCRF
from transformers_config import MODELS_dict
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR as SchedulerLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
import logstats
import os
import pickle
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def nlu_evaluation(model, dataset, dataset_test, nb_examples, use_slots, i, verbose=False):
    model.eval()

    intent_corrects = 0
    intents_true = []
    intents_pred = []

    slots_true = []
    slots_pred = []

    for _ in range(nb_examples):
        (input_ids, lengths, token_type_ids, input_masks, attention_mask, intent_labels, slot_labels, input_texts), text \
            = dataset.next_batch(1, dataset_test)

        input_ids = input_ids.cuda()
        lengths = lengths.cuda()
        input_masks = input_masks.cuda()
        intent_labels = intent_labels.cuda()
        slot_labels = slot_labels.cuda()

        if use_slots:
            intent_logits, slot_logits, intent_loss, slot_loss = model(input_ids=input_ids,
                                                                       lengths=lengths,
                                                                       input_masks=input_masks,
                                                                       intent_labels=intent_labels,
                                                                       slot_labels=slot_labels)

            # Slot Golden Truth/Predictions
            true_slot = slot_labels[0]
            pred_slot = list(slot_logits[0])

            true_slot_l = [dataset.slot_types[s] for s in true_slot]
            pred_slot_l = [dataset.slot_types[s] for s in pred_slot]

            true_slot_no_x = []
            pred_slot_no_x = []

            for i, slot in enumerate(true_slot_l):
                if slot != "X":
                    true_slot_no_x.append(true_slot_l[i])
                    pred_slot_no_x.append(pred_slot_l[i])

            slots_true.extend(true_slot_no_x)
            slots_pred.extend(pred_slot_no_x)


        else:
            intent_logits, intent_loss = model(input_ids=input_ids,
                                               lengths=lengths,
                                               input_masks=input_masks,
                                               intent_labels=intent_labels)

        # Intent Golden Truth/Predictions
        true_intent = intent_labels.squeeze().item()
        pred_intent = intent_logits.squeeze().max(0)[1]

        intent_corrects += int(pred_intent == true_intent)

        masked_text = ' '.join(dataset.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()))
        intents_true.append(true_intent)
        intents_pred.append(pred_intent.item())

    if verbose:
        print(i, " -----------intents_true:", set(intents_true))
        print(i, " -----------intents_pred:", set(intents_pred))

    intent_accuracy = float(intent_corrects) / nb_examples
    intent_prec = precision_score(intents_true, intents_pred, average="macro")
    intent_rec = recall_score(intents_true, intents_pred, average="macro")
    intent_f1 = f1_score(intents_true, intents_pred, average="macro")

    if use_slots:
        slot_prec = precision_score(slots_true, slots_pred, average="macro")
        slot_rec = recall_score(slots_true, slots_pred, average="macro")
        slot_f1 = f1_score(slots_true, slots_pred, average="macro")

        return intent_accuracy, intent_prec, intent_rec, intent_f1, slot_prec, slot_rec, slot_f1

    return intent_accuracy, intent_prec, intent_rec, intent_f1


def train(args, optimizer, model, dataset, subtask, writer, epoch, i, j):
    optimizer.zero_grad()
    model.train()
    # Take batch by batch and move to cuda
    batch, _ = dataset.next_batch(args.batch_size, subtask)

    input_ids, lengths, token_type_ids, input_masks, attention_mask, intent_labels, slot_labels, \
    input_texts = batch

    input_ids = input_ids.to(device)#cuda()
    lengths = lengths.to(device)#cuda()
    token_type_ids = token_type_ids.to(device)#cuda()
    input_masks = input_masks.to(device)#cuda()
    attention_mask = attention_mask.to(device)#cuda()
    intent_labels = intent_labels.to(device)#cuda()
    slot_labels = slot_labels.to(device)#cuda()

    if args.use_slots:
        logits_intents, logits_slots, intent_loss, slot_loss = model(input_ids,
                                                                     lengths=lengths,
                                                                     input_masks=input_masks,
                                                                     intent_labels=intent_labels,
                                                                     slot_labels=slot_labels)
        loss = intent_loss + slot_loss

        writer.add_scalar('train_slot_loss_'+str(i), slot_loss.mean(), j*epoch)
    else:
        logits_intents, intent_loss = model(input_ids,
                                            lengths=lengths,
                                            input_masks=input_masks,
                                            intent_labels=intent_labels)
        loss = intent_loss

    loss = loss.mean()
    loss.backward()

    writer.add_scalar('train_intent_loss_'+str(i), intent_loss.mean(), j*epoch)

    optimizer.step()

    if args.use_slots:
        return intent_loss, slot_loss
    else:
        return intent_loss


def evaluate_report(data_stream, k, train_lang, lang, args, dataset, model, writer, i, num_steps, verbose=False):

    outputs = nlu_evaluation(model,
                             dataset,
                             data_stream["stream"],
                             data_stream["size"],
                             args.use_slots,
                             i,
                             verbose)

    output_text_format = "----size=%d, k=%d, i=%d, and lang=%s" % (data_stream["size"],
                                                                   k,
                                                                   i,
                                                                   lang)

    metrics = {}
    if not args.use_slots:
        test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1 = outputs

    else:
        test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1, \
            test_slot_prec, test_slot_rec, test_slot_f1 = outputs

        output_text_format += " SLOTS perf: (prec=%.1f, rec=%.1f, f1=%.1f) " % (test_slot_prec*100,
                                                                                test_slot_rec*100,
                                                                                test_slot_f1*100)

        metrics.update({train_lang+'_'+str(i)+'_test_slot_prec_'+str(k)+'_'+lang: test_slot_prec})
        metrics.update({train_lang+'_'+str(i)+'_test_slot_rec_'+str(k)+'_'+lang: test_slot_rec})
        metrics.update({train_lang+'_'+str(i)+'_test_slot_f1_'+str(k)+'_'+lang: test_slot_f1})

    metrics.update({train_lang+'_'+str(i)+'_test_intent_acc_'+str(k)+'_'+lang: test_intent_acc})
    metrics.update({train_lang+'_'+str(i)+'_test_intent_prec_'+str(k)+'_'+lang: test_intent_prec})
    metrics.update({train_lang+'_'+str(i)+'_test_intent_rec_'+str(k)+'_'+lang: test_intent_rec})
    metrics.update({train_lang+'_'+str(i)+'_test_intent_f1_'+str(k)+'_'+lang: test_intent_f1})

    output_text_format += " INTENTS perf: (acc: %.1f, prec: %.1f, rec: %.1f, f1: %.1f)" % (test_intent_acc*100,
                                                                                           test_intent_prec*100,
                                                                                           test_intent_rec*100,
                                                                                           test_intent_f1*100)

    print(output_text_format)

    for k, v in metrics.items():
        writer.add_scalar(k, v, num_steps)

    return metrics


def set_optimizer(model, args):
    optimizer = Adam(model.parameters(),
                     betas=(args.beta_1, args.beta_2),
                     eps=args.adam_eps,
                     lr=args.adam_lr)

    scheduler = SchedulerLR(optimizer,
                            step_size=args.step_size,
                            gamma=args.gamma)

    model.zero_grad()

    return optimizer, scheduler


def set_out_dir(args):

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
        print("---- CREATED ", setup_opt_dir, " SUCCESS !")

    ## Slot Usage
    if not args.use_slots:
        setup_opt_dir = os.path.join(setup_opt_dir,
                                     "Intents_only")

        if not os.path.isdir(setup_opt_dir):
            os.mkdir(setup_opt_dir)
            print("---- CREATED ", setup_opt_dir, " SUCCESS !")

    ## Trans model
    model_dir = os.path.join(setup_opt_dir,
                             args.trans_model)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
        print("---- CREATED ", model_dir, " SUCCESS !")

    if args.setup_opt not in ["multi"]:
        order_lang_dir = os.path.join(model_dir,
                                      order_lang_dict[args.order_lang])

        if not os.path.isdir(order_lang_dir):
            os.mkdir(order_lang_dir)
            print("---- CREATED ", order_lang_dir, " SUCCESS !")

        model_dir = os.path.join(order_lang_dir,
                                 order_class_dict[args.order_class])

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
            print("---- CREATED ", model_dir, " SUCCESS !")

    ## SEED
    out_dir = os.path.join(model_dir,
                           "SEED_"+str(args.seed))

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        print("---- CREATED ", out_dir, " SUCCESS !")

    metrics_path = os.path.join(out_dir, "metrics")
    if not os.path.isdir(metrics_path):
        os.mkdir(metrics_path)
        print("---- CREATED ", metrics_path, " SUCCESS !")

    return out_dir, metrics_path


def run(out_dir, metrics_path, args):
    """
    The main of training over different streams and evaluating different approaches in terms of catastrophic forgetting
    and generalizability to new classes/languages
    :param args:
    :return:
    """

    writer = SummaryWriter(os.path.join(out_dir, 'runs'))

    model_name, tokenizer_alias, model_trans_alias = MODELS_dict[args.trans_model]
    if "mmhamdi" in args.data_root:
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
                      args.num_intent_tasks,
                      args.num_lang_tasks,
                      intent_types=intent_types,
                      slot_types=slot_types)

    # Incrementally adding new intents as they are added to the setup
    if args.setup_opt in ["cil", "cil-ll"]:
        eff_num_intent = len(dataset.intent_types) #args.num_intent_tasks
        eff_num_slot = len(dataset.slot_types)  # to be changed later
    else:
        eff_num_intent = len(dataset.intent_types)
        eff_num_slot = len(dataset.slot_types)

    model = TransNLUCRF(model_trans,
                        eff_num_intent,
                        device=device,
                        use_slots=args.use_slots,
                        num_slots=eff_num_slot)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    optimizer, scheduler = set_optimizer(model, args)
    print("BEFORE addition of OTHER type => num_intents = ", model.num_intents)

    if args.setup_opt == "cil":
        """ 
        Setup 1: Cross-CIL, Fixed LL: "Monolingual CIL":
        - Trnt(ain over every task of classes continuously independently for every language. 
        - We then average over all languages.
        """

        for lang in args.languages:
            # Iterating over the stream
            for i, subtask in enumerate(dataset.train_stream[lang]):
                num_steps = 0
                num_iter = subtask["size"] // args.batch_size
                stream = subtask["stream"]
                if subtask["size"] == 0:
                    print("Skipped task:", i, " in lang:", lang)
                    continue
                    
                #for epoch in tqdm(range(args.epochs)):
                for epoch in tqdm(range(3)):
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
                                                           j)
                        else:
                            intent_loss = train(args,
                                                optimizer,
                                                model,
                                                dataset,
                                                stream,
                                                writer,
                                                epoch,
                                                i,
                                                j)

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

        model = TransNLUCRF(model_trans,
                            eff_num_intent+1,
                            device=device,
                            use_slots=args.use_slots,
                            num_slots=eff_num_slot)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(device)

        optimizer, scheduler = set_optimizer(model, args)
        print("After addition of OTHER type => num_intents = ", model.num_intents)
        for lang in args.languages:
            # Iterating over the stream
            for i, subtask in enumerate(dataset.train_stream[lang]):
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
                                                           j)
                        else:
                            intent_loss = train(args,
                                                optimizer,
                                                model,
                                                dataset,
                                                stream,
                                                writer,
                                                epoch,
                                                i,
                                                j)

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
                                                 verbose=args.verbose)
                else:
                    print("Skipped task:", k, " in lang:", lang)

            with open(os.path.join(metrics_path, "final_metrics_"+lang+".pickle"), "wb") as output_file:
                pickle.dump(metrics, output_file)
            print("/////////////////////////////////////////////")

    elif args.setup_opt == "cll":
        """
        Setup 3: Cross-LL, Fixed CIL: "Conventional Cross-lingual Transfer Learning or Stream learning" 
        - Stream consisting of different combinations of languages.
        => Each stream sees all intents
        """

        # Iterating over the stream of languages
        for i, subtask in enumerate(dataset.train_stream):
            num_steps = 0
            num_iter = subtask["size"]//args.batch_size
            train_lang = subtask["lang"]
            stream = subtask["stream"]
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
                                                       j)

                        if j % args.eval_steps == 0:
                            print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                               intent_loss.mean(),
                                                                                               slot_loss.mean()))

                    else:
                        intent_loss = train(args,
                                            optimizer,
                                            model,
                                            dataset,
                                            stream,
                                            writer,
                                            epoch,
                                            i,
                                            j)

                        if j % args.eval_steps == 0:
                            print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

                metrics = {lang: {} for lang in dataset.test_stream}
                for lang in dataset.test_stream:
                    metrics[lang] = evaluate_report(dataset.test_stream[lang],
                                                    0,
                                                    train_lang,
                                                    lang,
                                                    args,
                                                    dataset,
                                                    model,
                                                    writer,
                                                    i,
                                                    num_steps,
                                                    verbose=args.verbose)

                with open(os.path.join(metrics_path, "epoch_"+str(epoch)+"_metrics_"+str(i)+".pickle"), "wb") as output_file:
                    pickle.dump(metrics, output_file)

            print("------------------------------------")
            metrics = {lang: {} for lang in dataset.test_stream}
            for lang in dataset.test_stream:
                metrics[lang] = evaluate_report(dataset.test_stream[lang],
                                                0,
                                                train_lang,
                                                lang,
                                                args,
                                                dataset,
                                                model,
                                                writer,
                                                i,
                                                num_steps,
                                                verbose=args.verbose)

            with open(os.path.join(metrics_path, "final_metrics_"+str(i)+".pickle"), "wb") as output_file:
                pickle.dump(metrics, output_file)

    elif args.setup_opt == "cil-ll":
        """
        Setup 4: Cross-CIL-LL: "Cross-lingual combinations of languages/intents"
        - Stream consisting of different combinations 
        """

        for i, subtask in enumerate(dataset.train_stream):
            num_steps = 0
            # Iterating over the stream of languages and intents

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
                                                       j)
                    else:
                        intent_loss = train(args,
                                            optimizer,
                                            model,
                                            dataset,
                                            stream,
                                            writer,
                                            epoch,
                                            i,
                                            j)

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
                                            verbose=args.verbose)

        with open(os.path.join(metrics_path, "final_metrics.pickle"), "wb") as output_file:
            pickle.dump(metrics, output_file)

    elif args.setup_opt == "multi-incremental-lang":
        """
        Setup 6: Multi-task/Joint Learning: train on all languages and intent classes at the same time 
        """
        for i, subtask in enumerate(dataset.train_stream):
            num_steps = 0
            num_iter = subtask["size"]//args.batch_size
            train_lang = subtask["lang"]
            stream = subtask["stream"]
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
                                                       0,
                                                       j)

                        if j % args.eval_steps == 0:
                            print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                               intent_loss.mean(),
                                                                                               slot_loss.mean()))
                    else:
                        intent_loss = train(args,
                                            optimizer,
                                            model,
                                            dataset,
                                            stream,
                                            writer,
                                            epoch,
                                            0,
                                            j)

                        if j % args.eval_steps == 0:
                            print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

                metrics = {lang: {} for lang in dataset.test_stream}
                for lang in dataset.test_stream:
                    metrics[lang] = evaluate_report(dataset.test_stream[lang],
                                                    i,
                                                    train_lang,
                                                    lang,
                                                    args,
                                                    dataset,
                                                    model,
                                                    writer,
                                                    i,
                                                    num_steps,
                                                    verbose=args.verbose)

                with open(os.path.join(metrics_path, train_lang+"_epoch_"+str(epoch)+"_metrics.pickle"), "wb") as output_file:
                    pickle.dump(metrics, output_file)

            print("Evaluation at the end of all epochs")
            metrics = {lang: {} for lang in dataset.test_stream}
            for lang in dataset.test_stream:
                metrics[lang] = evaluate_report(dataset.test_stream[lang],
                                                i,
                                                train_lang,
                                                lang,
                                                args,
                                                dataset,
                                                model,
                                                writer,
                                                i,
                                                num_steps,
                                                verbose=args.verbose)

            with open(os.path.join(metrics_path, "final_metrics_"+train_lang+".pickle"), "wb") as output_file:
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
    parser.add_argument("--data-root", help="Root directory of the data",
                        type=str, default="")

    parser.add_argument("--model-root", help="Path to the root directory hosting the trans model if offline",
                        type=str, default="")

    parser.add_argument("--out-dir", help="The root directory of the results for this project",
                        type=str, default="")

    parser.add_argument("--stats-file", help="Filename of the stats file",
                        type=str, default="stats.txt")

    parser.add_argument("--log-file", help="Filename of the log file",
                        type=str, default="log.txt")

    ## SETUP OPTION
    parser.add_argument("--setup-opt", help="Whether to pick setup "
                                            "   cil: Cross-CIL with fixed LL, "
                                            "   cil-other: incremental version of cil where previous intents' subtasks "
                                            "              are added in addition to other labels for subsequent "
                                            "              intents'subtasks, "
                                            "   cll: Cross-LL with fixed CIL,"
                                            "   cil-ll: Cross-CIL-LL,"
                                            "   multi: multi-tasking one model on all tasks and langs,"
                                            "   multi-incremental-lang: weaker version of Multi-task/Joint Learning "
                                            "                           where we gradually fine-tune",
                        type=str, default="cll")

    parser.add_argument("--order-class", help="Different ways of ordering the classes"
                                              "0: decreasing order (from high to low-resource), "
                                              "1: increasing order (from low to high-resource),"
                                              "2: random order",
                        type=int, default=0)

    parser.add_argument("--order-lang", help="Different ways of ordering the languages"
                                             "0: decreasing order (from high to low-resource) , "
                                             "1: increasing order (from low to high-resource),"
                                             "2: random order",
                        type=int, default=0)

    parser.add_argument("--setup-3", help="intents: traversing subtasks horizontally over all classes first then "
                                          "         to languages,"
                                          "langs: traversing subtasks vertically over all languages first then "
                                          "         to classes",
                        type=str, default="intents")

    parser.add_argument('--use-slots', help='If true, optimize for slot filling loss too',
                        action='store_true')

    parser.add_argument('--verbose', help='If true, return golden labels and predictions ',
                        action='store_true')

    parser.add_argument('--no-debug', help='If true, return golden labels and predictions ',
                        action='store_true')

    ## TRAINING OPTIONS
    parser.add_argument("--trans-model", help="Name of transformer model",
                        type=str, default="BertBaseMultilingualCased")

    parser.add_argument('--data-format', help='Whether it is tsv (MTOD), json, or txt (MTOP)',
                        type=str, default="txt")

    parser.add_argument("--languages", help="train languages list",
                        nargs="+", default=["de", "en", "es", "fr", "hi", "th"])

    parser.add_argument("--num-intent-tasks", help="The number of intent per task",
                        type=int, default=10)

    parser.add_argument("--num-lang-tasks", help="The number of lang per task",
                        type=int, default=2)

    parser.add_argument("--epochs", help="The total number of epochs",
                        type=int, default=10)

    parser.add_argument("--eval-steps", help="The total number of epochs for the model to evaluate",
                        type=int, default=200)

    parser.add_argument("--batch-size", help="The total number of epochs for the model to evaluate",
                        type=int, default=32)

    parser.add_argument("--step-size", help="The step size for the scheduler",
                        type=int, default=7)

    parser.add_argument("--gamma", help="gamma for the scheduler",
                        type=float, default=0.1)

    parser.add_argument("--adam-lr", help="The learning rate",
                        type=float, default=1e-03)

    parser.add_argument("--adam-eps", help="epsilon",
                        type=float, default=1e-08)

    parser.add_argument("--beta-1", help="beta_1 for Adam",
                        type=float, default=0.9)

    parser.add_argument("--beta-2", help="beta_2 for Adam",
                        type=float, default=0.99)

    parser.add_argument("--seed", help="The total number of epochs",
                        type=int, default=42)

    args = parser.parse_args()
    set_seed(args)

    out_dir, metrics_path = set_out_dir(args)

    if args.no_debug:
        stdoutOrigin = sys.stdout

        sys.stdout = open(os.path.join(out_dir, args.log_file), "w")
        logstats.init(os.path.join(out_dir, args.stats_file))
        config_path = os.path.join(out_dir, 'config.json')
        logstats.add_args('config', args)
        logstats.write_json(vars(args), config_path)

    run(out_dir, metrics_path, args)

    if args.no_debug:
        sys.stdout.close()
        sys.stdout = stdoutOrigin

