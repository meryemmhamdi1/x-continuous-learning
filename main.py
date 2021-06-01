from pytorch_transformers import AdamW, WarmupLinearSchedule
from data_utils import *
import argparse
from consts import domain_types, intent_types, slot_types
import gc
import numpy as np
from models.transformerNLU import *
from transformers_config import MODELS_dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score


def nlu_evaluation(model, dataset, lang, nb_examples, use_slots):
    model.eval()

    intent_corrects = 0
    intents_true = []
    intents_pred = []

    slots_true = []
    slots_pred = []

    seen_examples = []
    print("nb_examples:", nb_examples)
    for kkk in range(nb_examples):
        (input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts), text \
            = dataset.next_batch(1, dataset.test, [lang])

        input_ids = input_ids.cuda()
        lengths = lengths.cuda()
        intent_labels = intent_labels.cuda()
        slot_labels = slot_labels.cuda()

        if use_slots:
            intent_logits, slot_logits, intent_loss, slot_loss = model(input_ids=input_ids,
                                                                       intent_labels=intent_labels,
                                                                       slot_labels=slot_labels)

            # Slot Golden Truth/Predictions
            true_slot = slot_labels[0]
            pred_slot = list(slot_logits.cpu().squeeze().max(-1)[1].numpy())

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
                                               intent_labels=intent_labels)

        # Intent Golden Truth/Predictions
        true_intent = intent_labels.squeeze().item()
        pred_intent = intent_logits.squeeze().max(0)[1]

        intent_corrects += int(pred_intent == true_intent)

        masked_text = ' '.join(dataset.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist()))
        intents_true.append(true_intent)
        intents_pred.append(pred_intent.item())

    print("LEN(SEEN_EXAMPLES):", len(seen_examples))

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


def set_optimizer(model, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.adam_lr,
                      eps=args.adam_eps)

    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=0,
                                     t_total=2000)

    model.zero_grad()

    return optimizer, scheduler


def run(args):
    """
    The main of training over different streams and evaluating different approaches in terms of catastrophic forgetting
    and generalizability to new classes/languages
    :param args:
    :return:
    """

    setups_dict = {"cil": "Cross-CIL_fixed-LL",
                   "cll": "Cross-LL_Fixed-CIL",
                   "cil-ll": "Cross-CIL_Cross-LL",
                   "multi": "Multi-Task"}

    out_dir = os.path.join(os.path.join(args.out_dir, args.setup_opt), "SEED_"+str(args.seed)+"/")
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
    if args.setup_opt == 1:
        eff_num_intent = args.num_intent_tasks
    elif args.setup_opt == 2:
        eff_num_intent = len(dataset.intent_types)
        eff_num_slot = len(dataset.slot_types)
    else:
        eff_num_intent = args.num_intent_tasks

    model = TransformerNLU(model_trans,
                           eff_num_intent,
                           use_slots=args.use_slots,
                           num_slots=len(dataset.slot_types))

    if torch.cuda.device_count() > 0:
        model.cuda()

    optimizer, scheduler = set_optimizer(model, args)


    best_sum_metrics = 0
    count = 0
    best_sum_dev_metrics = 0

    if args.setup_opt == "cil":
        """ 
        Setup 1: Cross-CIL, Fixed LL: "Monolingual CIL":
        - Train over every task of classes continuously independently for every language. 
        - We then average over all languages.
        """
        for epoch in tqdm(range(args.epochs)):
            optimizer.zero_grad()
            gc.collect()

            number_steps = 0
            for lang in args.language:
                # Iterating over the stream
                for i, subtask in enumerate(dataset.train_stream[lang]):
                    number_steps += 1
                    num_iterations = subtask["size"]// args.batch_size
                    for j in range(num_iterations):
                        # Take batch by batch and move to cuda
                        batch, _ = dataset.next_batch(args.batch_size, subtask["stream"])
                        intent_classes = subtask["intent_list"]

                        input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts = batch

                        input_ids = input_ids.cuda()
                        lengths = lengths.cuda()
                        token_type_ids = token_type_ids.cuda()
                        attention_mask = attention_mask.cuda()
                        intent_labels = intent_labels.cuda()
                        slot_labels = slot_labels.cuda()

                        if args.use_slots:
                            logits_intents, logits_slots, intent_loss, slot_loss = model(input_ids,
                                                                                         intent_labels=intent_labels,
                                                                                         slot_labels=slot_labels)
                            loss = intent_loss + slot_loss

                            writer.add_scalar('train_slot_loss_'+str(i), slot_loss.mean(), j*epoch)
                        else:
                            logits_intents, intent_loss = model(input_ids,
                                                                intent_labels=intent_labels)
                            loss = intent_loss

                        loss = loss.mean()
                        loss.backward()

                        writer.add_scalar('train_intent_loss_'+str(i), intent_loss.mean(), j*epoch)

                        optimizer.step()
                        scheduler.step()

                        if j > 0 and j % 10 == 0:
                            if args.use_slots:
                                print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                                   intent_loss.mean(),
                                                                                                   slot_loss.mean()))
                            else:
                                print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

                            test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1, test_slot_prec, \
                            test_slot_rec, test_slot_f1 = nlu_evaluation(model,
                                                                         dataset.test_stream[lang][j]["stream"],
                                                                         lang,
                                                                         dataset.test_stream[lang][j]["size"],
                                                                         args.use_slots)

                            writer.add_scalar('test_slot_prec_'+str(i)+'_'+lang, test_slot_prec, j*epoch)
                            writer.add_scalar('test_slot_rec_'+str(i)+'_'+lang, test_slot_rec, j*epoch)
                            writer.add_scalar('test_slot_f1_'+str(i)+'_'+lang, test_slot_f1, j*epoch)

    elif args.setup_opt == "cll":
        """
        Setup 2: Cross-LL, Fixed CIL: "Conventional Cross-lingual Transfer Learning or Stream learning" 
        - Stream consisting of different combinations of languages.
        => Each stream sees all intents
        """
        for epoch in tqdm(range(args.epochs)):
            optimizer.zero_grad()
            gc.collect()

            number_steps = 0
            # Iterating over the stream of languages
            for i, subtask in enumerate(dataset.train_stream):
                number_steps += 1
                num_iterations = subtask["size"]//args.batch_size
                lang = subtask["lang"]
                for j in range(num_iterations):
                    # Take batch by batch and move to cuda
                    batch, _ = dataset.next_batch(args.batch_size, subtask["stream"])
                    intent_classes = subtask["intent_list"]

                    input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts = batch

                    input_ids = input_ids.cuda()
                    lengths = lengths.cuda()
                    token_type_ids = token_type_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    intent_labels = intent_labels.cuda()
                    slot_labels = slot_labels.cuda()

                    if args.use_slots:
                        logits_intents, logits_slots, intent_loss, slot_loss = model(input_ids,
                                                                                     intent_labels=intent_labels,
                                                                                     slot_labels=slot_labels)
                        loss = intent_loss + slot_loss

                        writer.add_scalar('train_slot_loss_'+str(i), slot_loss.mean(), j*epoch)
                    else:
                        logits_intents, intent_loss = model(input_ids,
                                                            intent_labels=intent_labels)
                        loss = intent_loss

                    loss = loss.mean()
                    loss.backward()

                    writer.add_scalar('train_intent_loss_'+str(i), intent_loss.mean(), j*epoch)

                    optimizer.step()
                    scheduler.step()

                    if j > 0 and j % 10 == 0:
                        if args.use_slots:
                            print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                               intent_loss.mean(),
                                                                                               slot_loss.mean()))
                        else:
                            print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

                        test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1, test_slot_prec, \
                        test_slot_rec, test_slot_f1 = nlu_evaluation(model,
                                                                     dataset.test_stream[lang][j]["stream"],
                                                                     lang,
                                                                     dataset.test_stream[lang][j]["size"],
                                                                     args.use_slots)

                        writer.add_scalar('test_slot_prec_'+str(i)+'_'+lang, test_slot_prec, j*epoch)
                        writer.add_scalar('test_slot_rec_'+str(i)+'_'+lang, test_slot_rec, j*epoch)
                        writer.add_scalar('test_slot_f1_'+str(i)+'_'+lang, test_slot_f1, j*epoch)

    elif args.setup_opt == "cil-ll":
        for epoch in tqdm(range(args.epochs)):
            optimizer.zero_grad()
            gc.collect()

            number_steps = 0
            # Iterating over the stream of languages
            for i, subtask in enumerate(dataset.train_stream):
                number_steps += 1
                num_iterations = subtask["size"]//args.batch_size
                lang = subtask["lang"]
                for j in range(num_iterations):
                    # Take batch by batch and move to cuda
                    batch, _ = dataset.next_batch(args.batch_size, subtask["stream"])
                    intent_classes = subtask["intent_list"]

                    input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts = batch

                    input_ids = input_ids.cuda()
                    lengths = lengths.cuda()
                    token_type_ids = token_type_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    intent_labels = intent_labels.cuda()
                    slot_labels = slot_labels.cuda()

                    if args.use_slots:
                        logits_intents, logits_slots, intent_loss, slot_loss = model(input_ids,
                                                                                     intent_labels=intent_labels,
                                                                                     slot_labels=slot_labels)
                        loss = intent_loss + slot_loss

                        writer.add_scalar('train_slot_loss_'+str(i), slot_loss.mean(), j*epoch)
                    else:
                        logits_intents, intent_loss = model(input_ids,
                                                            intent_labels=intent_labels)
                        loss = intent_loss

                    loss = loss.mean()
                    loss.backward()

                    writer.add_scalar('train_intent_loss_'+str(i), intent_loss.mean(), j*epoch)

                    optimizer.step()
                    scheduler.step()

                    if j > 0 and j % 10 == 0:
                        if args.use_slots:
                            print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                               intent_loss.mean(),
                                                                                               slot_loss.mean()))
                        else:
                            print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

                        test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1, test_slot_prec, \
                        test_slot_rec, test_slot_f1 = nlu_evaluation(model,
                                                                     dataset.test_stream[lang][j]["stream"],
                                                                     lang,
                                                                     dataset.test_stream[lang][j]["size"],
                                                                     args.use_slots)

                        writer.add_scalar('test_slot_prec_'+str(i)+'_'+lang, test_slot_prec, j*epoch)
                        writer.add_scalar('test_slot_rec_'+str(i)+'_'+lang, test_slot_rec, j*epoch)
                        writer.add_scalar('test_slot_f1_'+str(i)+'_'+lang, test_slot_f1, j*epoch)

    elif args.setup_opt == "multi":
        for epoch in tqdm(range(args.epochs)):
            optimizer.zero_grad()
            gc.collect()

            number_steps = 0
            # There is only one task here no subtasks
            task = dataset.train_stream["data"]
            number_steps += 1
            num_iterations = task["size"]//args.batch_size
            for j in range(num_iterations):
                # Take batch by batch and move to cuda
                batch, _ = dataset.next_batch(args.batch_size, task["stream"])
                intent_classes = task["intent_list"]

                input_ids, lengths, token_type_ids, attention_mask, intent_labels, slot_labels, input_texts = batch

                input_ids = input_ids.cuda()
                lengths = lengths.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                intent_labels = intent_labels.cuda()
                slot_labels = slot_labels.cuda()

                if args.use_slots:
                    logits_intents, logits_slots, intent_loss, slot_loss = model(input_ids,
                                                                                 intent_labels=intent_labels,
                                                                                 slot_labels=slot_labels)
                    loss = intent_loss + slot_loss

                    writer.add_scalar('train_slot_loss', slot_loss.mean(), j*epoch)
                else:
                    logits_intents, intent_loss = model(input_ids,
                                                        intent_labels=intent_labels)
                    loss = intent_loss

                loss = loss.mean()
                loss.backward()

                writer.add_scalar('train_intent_loss', intent_loss.mean(), j*epoch)

                optimizer.step()
                scheduler.step()

                if j > 0 and j % 10 == 0:
                    if args.use_slots:
                        print('Iter {} | Intent Loss = {:.4f} | Slot Loss = {:.4f}'.format(j,
                                                                                           intent_loss.mean(),
                                                                                           slot_loss.mean()))
                    else:
                        print('Iter {} | Intent Loss = {:.4f} '.format(j, intent_loss.mean()))

                    for lang in dataset.test_stream:
                        test_intent_acc, test_intent_prec, test_intent_rec, test_intent_f1, test_slot_prec, \
                        test_slot_rec, test_slot_f1 = nlu_evaluation(model,
                                                                     dataset.test_stream[lang][j]["stream"],
                                                                     lang,
                                                                     dataset.test_stream[lang][j]["size"],
                                                                     args.use_slots)

                        writer.add_scalar('test_slot_prec_'+lang, test_slot_prec, j*epoch)
                        writer.add_scalar('test_slot_rec_'+lang, test_slot_rec, j*epoch)
                        writer.add_scalar('test_slot_f1_'+lang, test_slot_f1, j*epoch)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="", help="Root directory of the data")
    parser.add_argument("--setup-opt", type=str, default="cil", help="Whether to pick setup "
                                                                     "cil: Cross-CIL with fixed LL, "
                                                                     "cll: Cross-LL with fixed CIL,"
                                                                     "cil-ll: Cross-CIL-LL,"
                                                                     "multi: Multi-Task one model over all tasks and languages")

    parser.add_argument("--setup-3", type=str, default="intents",
                        help="intents: traversing subtasks horizontally over all classes first then to languages,"
                             "langs: traversing subtasks vertically over all languages first then to classes")

    parser.add_argument("--trans-model", type=str, default="BertBaseMultilingualCased",
                        help="Name of transformer model")

    parser.add_argument("--model-root", type=str, default="/home1/mmhamdi/Models/",
                        help="Path to the root directory hosting the trans model")

    parser.add_argument('--data-format', type=str, help='Whether it is tsv (MTOD), json, or txt (MTOP)', default="txt")
    parser.add_argument('--use-slots', help='If true, optimize for slot filling loss too', action='store_true')
    parser.add_argument("--languages", help="train languages list", nargs="+", default=["de", "en", "es", "fr", "hi", "th"])

    parser.add_argument("--order-class", type=int, default= 0, help= "Different ways of ordering the classes"
                                                                     "0: decreasing order (from high to low-resource), "
                                                                     "1: increasing order (from low to high-resource),"
                                                                     "2: random order")

    parser.add_argument("--order-lang", type=int, default= 0, help= "Different ways of ordering the languages"
                                                                    "0: decreasing order (from high to low-resource) , "
                                                                    "1: increasing order (from low to high-resource),"
                                                                    "2: random order")

    parser.add_argument("--num-intent-tasks", type=int, default=10, help="The number of intent per task")
    parser.add_argument("--num-lang-tasks", type=int, default=2, help="The number of lang per task")
    parser.add_argument("--out-dir", type=str, default="", help="The root directory of the results for this project")

    parser.add_argument("--epoch", type=int, default=10, help="The total number of epochs")
    parser.add_argument("--batch-size", type=int, default=10, help="The total number of epochs")

    args = parser.parse_args()
    set_seed(args)

    run(args)

