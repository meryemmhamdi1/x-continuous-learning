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


#def evaluate(args):

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

    setups_dict = {1: "Cross-CIL_fixed-LL",
                   2: "Cross-LL_Fixed-CIL",
                   3: "Cross-CIL_Cross-LL",
                   4: "Multi-Task"}

    out_dir = os.path.join(os.path.join(args.out_dir, setups_dict[args.setup_opt]), "SEED_"+str(args.seed)+"/")
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
    else:
        eff_num_intent = args.num_intent_tasks

    model = TransformerNLU(model_trans,
                           eff_num_intent,
                           use_slots=args.use_slots,
                           num_slots=len(dataset.slot_types))

    if torch.cuda.device_count() > 0:
        model.cuda()

    optimizer, scheduler = set_optimizer(model, args)

    # Iterating over the stream
    number_steps = 0
    best_sum_metrics = 0
    count = 0
    best_sum_dev_metrics = 0
    opt = torch.optim.Adam(model.parameters(),  lr=args.alpha_lr)
    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        gc.collect()
        if args.setup_opt == 1:
            """ 
            Setup 1: Cross-CIL, Fixed LL: "Monolingual CIL":
            - Train over every task of classes continuously independently for every language. 
            - We then average over all languages.
            """
            for lang in args.language:
                for subtask in dataset.train_stream[lang]:
                    # Batchify this subtask # TODO HERE
                    # Take batch by batch and move to cuda # TODO HERE
                    if args.use_slots:
                        logits_intents, logits_slots, intent_loss, slot_loss = model(input_ids, intent_labels=intent_labels,
                                                                                     slot_labels=slot_labels)
                        loss = intent_loss + slot_loss

                        writer.add_scalar('train_slot_loss', slot_loss.mean(), i) # TODO: save per task id
                    else:
                        logits_intents, intent_loss = model(input_ids, intent_labels=intent_labels)
                        loss = intent_loss

                    loss = loss.mean()
                    loss.backward()

                    writer.add_scalar('train_intent_loss', intent_loss.mean(), l) # TODO: save per task id

                    optimizer.step()
                    scheduler.step()


    return


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="", help="Root directory of the data")
    parser.add_argument("--setup-opt", type=int, default= 1, help="Whether to pick setup "
                                                                  "1: Cross-CIL with fixed LL, "
                                                                  "2: Cross-LL with fixed CIL,"
                                                                  "3: Cross-CIL-LL, "
                                                                  "4: Multi-Task one model over all tasks and languages")

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

    parser.add_argument("--epoch", type=int, help="The total number of epochs")


    args = parser.parse_args()
    set_seed(args)

    run(args)

