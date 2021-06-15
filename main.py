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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def nlu_evaluation(model, dataset, dataset_test, nb_examples, use_slots):
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

            #if torch.cuda.device_count() > 1:
            #    slot_logits = model.module.crf.decode(slot_logits, input_masks.byte())
            #else:
            #    slot_logits = model.crf.decode(slot_logits, input_masks.byte())

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


def evaluate_report(data_stream, lang, train_lang, args, dataset, model, writer, epoch, i, j):
    test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1, test_slot_prec, \
    test_slot_rec, test_slot_f1 = nlu_evaluation(model,
                                                 dataset,
                                                 data_stream["stream"],
                                                 data_stream["size"],
                                                 args.use_slots)

    print("----lang:", lang, " intent_acc:", test_intent_acc, " slot_f1:", test_slot_f1)

    writer.add_scalar(train_lang+'_test_intent_acc_'+str(i)+'_'+lang, test_intent_acc, j*epoch)
    writer.add_scalar(train_lang+'_test_intent_f1_'+str(i)+'_'+lang, test_intent_f1, j*epoch)
    writer.add_scalar(train_lang+'_test_intent_prec_'+str(i)+'_'+lang, test_intent_prec, j*epoch)
    writer.add_scalar(train_lang+'_test_intent_rec_'+str(i)+'_'+lang, test_intent_rec, j*epoch)
    writer.add_scalar(train_lang+'_test_slot_prec_'+str(i)+'_'+lang, test_slot_prec, j*epoch)
    writer.add_scalar(train_lang+'_test_slot_rec_'+str(i)+'_'+lang, test_slot_rec, j*epoch)
    writer.add_scalar(train_lang+'_test_slot_f1_'+str(i)+'_'+lang, test_slot_f1, j*epoch)


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
    setups_dict = {"cil": "Cross-CIL_fixed-LL",
                   "cll": "Cross-LL_Fixed-CIL",
                   "cil-ll": "Cross-CIL_Cross-LL",
                   "multi": "Multi-Task"}

    order_lang_dict = {0: "high2lowlang",
                       1: "low2highlang",
                       2: "randomlang"}

    order_class_dict = {0: "high2lowclass",
                        1: "low2highclass",
                        2: "randomclass"}

    if args.setup_opt == "multi":
        out_dir = os.path.join(os.path.join(args.out_dir, args.setup_opt),
                               args.trans_model
                               + "/SEED_"+str(args.seed)+"/")
    else:
        out_dir = os.path.join(os.path.join(args.out_dir, args.setup_opt),
                               args.trans_model + "/" +
                               order_lang_dict[args.order_lang] + "/" +
                               order_class_dict[args.order_class] +
                               "/SEED_"+str(args.seed)+"/")
    return out_dir


def run(out_dir, args):
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

    if args.setup_opt == "cil":
        """ 
        Setup 1: Cross-CIL, Fixed LL: "Monolingual CIL":
        - Train over every task of classes continuously independently for every language. 
        - We then average over all languages.
        """
        for lang in args.languages:

            number_steps = 0
            for epoch in tqdm(range(args.epochs)):
                gc.collect()
                # Iterating over the stream
                for i, subtask in enumerate(dataset.train_stream[lang]):
                    number_steps += 1
                    num_iterations = subtask["size"] // args.batch_size
                    stream = subtask["stream"]
                    for j in range(num_iterations):
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
                            evaluate_report(dataset.test_stream[lang][i],
                                            lang,
                                            lang,
                                            args,
                                            dataset,
                                            model,
                                            writer,
                                            epoch,
                                            i,
                                            j)

                    scheduler.step()
                    print("------------------------------------")
                print("/////////////////////////////////////////////")

    elif args.setup_opt == "cil-ll":
        number_steps = 0
        for epoch in tqdm(range(args.epochs)):
            gc.collect()

            # Iterating over the stream of languages
            for i, subtask in enumerate(dataset.train_stream):
                num_iterations = subtask["size"] // args.batch_size
                lang = subtask["lang"]
                stream = subtask["stream"]
                for j in range(num_iterations):
                    number_steps += 1

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
                            if dataset.test_stream[i][lang]["size"] > 0:
                                print("dataset size:", dataset.test_stream[i][lang]["size"])
                                print("dataset stream:", dataset.test_stream[i][lang]["stream"])
                                evaluate_report(dataset.test_stream[i][lang],
                                                lang,
                                                lang,
                                                args,
                                                dataset,
                                                model,
                                                writer,
                                                epoch,
                                                i,
                                                j)


                scheduler.step()
    elif args.setup_opt == "cll":
        """
        Setup 2: Cross-LL, Fixed CIL: "Conventional Cross-lingual Transfer Learning or Stream learning" 
        - Stream consisting of different combinations of languages.
        => Each stream sees all intents
        """

        # Iterating over the stream of languages
        for i, subtask in enumerate(dataset.train_stream):
            number_steps = 0
            for epoch in tqdm(range(args.epochs)):
                gc.collect()
                number_steps += 1
                num_iterations = subtask["size"]//args.batch_size
                train_lang = subtask["lang"]
                stream = subtask["stream"]
                for j in range(num_iterations):
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
                        for lang in dataset.test_stream:
                            evaluate_report(dataset.test_stream[lang],
                                            lang,
                                            train_lang,
                                            args,
                                            dataset,
                                            model,
                                            writer,
                                            epoch,
                                            i,
                                            j)

                scheduler.step()
            print("------------------------------------")

    elif args.setup_opt == "multi":
        for epoch in tqdm(range(args.epochs)):
            gc.collect()

            number_steps = 0
            # There is only one task here no subtasks
            task = dataset.train_stream["data"]
            number_steps += 1

            num_iterations = dataset.train_stream["size"]//args.batch_size

            for j in range(num_iterations):

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
                if j % args.eval_steps == 0:
                    for lang in dataset.test_stream:
                        evaluate_report(dataset.test_stream[lang],
                                        lang,
                                        "all",
                                        args,
                                        dataset,
                                        model,
                                        writer,
                                        epoch,
                                        0,
                                        j)

            scheduler.step()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="", help="Root directory of the data")

    parser.add_argument("--out-dir", type=str, default="", help="The root directory of the results for this project")

    parser.add_argument("--stats-file", type=str, default="stats.txt", help="Filename of the stats file")
    parser.add_argument("--log-file", type=str, default="log.txt", help="Filename of the log file")

    parser.add_argument("--setup-opt", type=str, default="cll", help="Whether to pick setup "
                                                                     "cil: Cross-CIL with fixed LL, "
                                                                     "cll: Cross-LL with fixed CIL,"
                                                                     "cil-ll: Cross-CIL-LL,"
                                                                     "multi: multi-tasking one model on all tasks "
                                                                     "and langs")

    parser.add_argument("--order-class", type=int, default=0, help="Different ways of ordering the classes"
                                                                   "0: decreasing order (from high to low-resource), "
                                                                   "1: increasing order (from low to high-resource),"
                                                                   "2: random order")

    parser.add_argument("--order-lang", type=int, default=0, help="Different ways of ordering the languages"
                                                                  "0: decreasing order (from high to low-resource) , "
                                                                  "1: increasing order (from low to high-resource),"
                                                                  "2: random order")

    parser.add_argument("--setup-3", type=str, default="intents",
                        help="intents: traversing subtasks horizontally over all classes first then to languages,"
                             "langs: traversing subtasks vertically over all languages first then to classes")

    parser.add_argument("--trans-model", type=str, default="BertBaseMultilingualCased",
                        help="Name of transformer model")

    parser.add_argument("--model-root", type=str, default="",
                        help="Path to the root directory hosting the trans model")

    parser.add_argument('--data-format', type=str, default="txt", help='Whether it is tsv (MTOD), json, or txt (MTOP)')
    parser.add_argument('--use-slots', action='store_true', help='If true, optimize for slot filling loss too')
    parser.add_argument("--languages", nargs="+", default=["de", "en", "es", "fr", "hi", "th"],
                        help="train languages list")

    parser.add_argument("--num-intent-tasks", type=int, default=10, help="The number of intent per task")
    parser.add_argument("--num-lang-tasks", type=int, default=2, help="The number of lang per task")

    parser.add_argument("--epochs", type=int, default=10, help="The total number of epochs")
    parser.add_argument("--eval-steps", type=int, default=200, help="The total number of epochs for the model to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="The total number of epochs for the model to evaluate")
    parser.add_argument("--step-size", type=int, default=7, help="The step size for the scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="gamma for the scheduler")
    parser.add_argument("--adam-lr", type=float, default=1e-03, help="The learning rate")
    parser.add_argument("--adam-eps", type=float, default=1e-08, help="epsilon")
    parser.add_argument("--beta-1", type=float, default=0.9, help="beta_1 for Adam")
    parser.add_argument("--beta-2", type=float, default=0.99, help="beta_2 for Adam")
    parser.add_argument("--seed", type=int, default=42, help="The total number of epochs")

    args = parser.parse_args()
    set_seed(args)

    stdoutOrigin = sys.stdout

    out_dir = set_out_dir(args)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    sys.stdout = open(out_dir + args.log_file, "w")
    logstats.init(out_dir + args.stats_file)
    logstats.add_args('config', args)

    run(out_dir, args)

    sys.stdout.close()
    sys.stdout = stdoutOrigin

