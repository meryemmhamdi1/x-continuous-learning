import torch, sys
import torch.optim as optim
import torch.nn as nn
from torch import LongTensor
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, AdamW
import argparse, os
import time, math
from tqdm import tqdm
import numpy as np
import random
import json
import importlib

sys.path.append("/home1/mmhamdi/x-continuous-learning")
import src.logstats as logstats
from src.transformers_config import MODELS_dict
from src.basemodels.transNLI import TransNLI
from src.data_utils import MultiPurposeDataset, AugmentedList
# import src.schedulers.ltn as SpacedRepetitionModel
from src.utils import epoch_time, categorical_accuracy, transfer_batch_cuda, logger

BATCH_SIZE = 4
SEED = 42
N_EPOCHS = 2 # 10
WARMUP_PERCENT = 0
NUM_CLASSES = 3
ADAM_LR=2e-5 #3e-01 # 2e-5
GRAD_ACC = 4
ADAM_EPS=1e-08 # 1e-6
BETA_1=0.9
BETA_2=0.99
OUT_PATH = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/"

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def get_arguments():
    parser = argparse.ArgumentParser("./test_base_models.py", description='Testing of transNLU module.')
    parser.add_argument('--data_root', type=str, default="/project/jonmay_231/meryem/Datasets/NLI/XNLI/")
    parser.add_argument('--task_name', type=str, default="nli")
    parser.add_argument("--data_name", help="Whether it is mtop, matis, nli, or tydiqa.",
                        choices=["schuster", "jarvis", "mtop", "multiatis", "xnli", "tydiqa"],
                        type=str, default="xnli")
    parser.add_argument('--data_format', type=str, default="tsv")
    parser.add_argument('--order_lst', type=str, default="en")#default="en_zh_vi_ar_tr_bg_el_ur")
    parser.add_argument('--languages', nargs="+", default=["en"])#default=["zh", "vi", "ar", "tr", "bg", "el", "ur", "en" ])
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--setup_opt', type=str, default="cll-er_kd")
    parser.add_argument('--num_class_tasks', type=int, default=10)
    parser.add_argument('--num_lang_tasks', type=int, default=2)
    parser.add_argument('--use_mono', action="store_true")
    parser.add_argument('--order_class', type=int, default=0)
    parser.add_argument('--order_lang', type=int, default=0)
    parser.add_argument('--xnli_max_length', type=int, default=512)
    parser.add_argument('--pad_token', type=int, default=0)
    parser.add_argument('--mask_padding_with_zero', type=bool, default=True)
    parser.add_argument('--pad_token_segment_id', type=int, default=0)
    parser.add_argument('--trans_model', type=str, default="BertBaseMultilingualCased")#default="XLMRoberta_base")
    parser.add_argument('--model_root', type=str, default="/project/jonmay_231/meryem/Models/")
    parser.add_argument('--use_slots', action="store_true")
    parser.add_argument("--use_crf", action="store_true") 
    parser.add_argument('--use_adapters', action="store_true")
    parser.add_argument('--multi_head_out', action="store_true")
    parser.add_argument('--adapter_type', type=str, default="")
    
    parser.add_argument('--rand_perf', action="store_true")

    parser.add_argument("--stats_file", help="Filename of the stats file.", 
                        type=str, default="stats.txt")
    parser.add_argument("--log_file", help="Filename of the log file.",
                        type=str, default="log.txt")
    
    parser.add_argument("--use_leitner", action="store_true")
    parser.add_argument("--demote_to_first_deck", action="store_true")
    parser.add_argument("--num_decks", type=int, default=5)
    parser.add_argument("--lt_sampling_mode", choices=["fifo", "rand"], type=str, default="fifo")
    parser.add_argument("--lt_queue_mode", choices=["mono", "cont-mono", "cont-multi", "multi"], default="mono")
    parser.add_argument("--ltn_model", choices=["ltn", "rbf"], default="ltn")
    parser.add_argument("--sample_batch_epoch", choices=["batch", "epoch"], default="epoch", help="Whether to sample training examples at the end of each batch or epoch")
    parser.add_argument("--update_batch_epoch", choices=["batch", "epoch"], default="batch", help="Whether to update leitner queue at the end of each batch or epoch")
    parser.add_argument("--er_lq_scheduler_mode",
                        help="The mode used for training ER with LQ whether we use a sampling proportion that we fix or we use a frequency proportion in the interleaving of the two losses",
                        choices=["sample_prop", "interleave_prop"],
                        type=str, default="interleave_prop")
    
    parser.add_argument("--ltn_promote_thresh", type=float, default=1.0)
    
    parser.add_argument("--nu",
                        help="confidence parameter for RBF Algorithm",
                        type=float, default=0.5)

    parser.add_argument("--kern",
                        help="Kernel function for RBF Algorithm",
                        type=str, default="cos")
    
    ### ER Parameters
    parser.add_argument("--use_er", action="store_true")
    parser.add_argument("--max_mem_sz", help="The maximum size of the memory to be used in replay",
                        type=int, default=100)
    parser.add_argument("--use_er_only", help="Whether to do the backward over both main and memory batch or just the memory batch every 100 steps",
                        action="store_true")
    parser.add_argument("--er_lq_scheduler_prop",
                        help="The proportion of ER memory used in Leitner Queue in next_items.",
                        type=float, default=1.0)
    parser.add_argument("--er_lq_scheduler_type", choices=["er", "main", "both"], type=str, default="main")
    
    args = parser.parse_args()
    return args

def setup_config(args):
    results_dir = os.path.join(OUT_PATH,
                               args.data_name.upper(),
                               "HyperparamSearch",
                               args.trans_model,
                               "LtnScheduler" if args.use_leitner else "Baseline")
    
    if not args.rand_perf:
        if args.use_er:
            results_dir = os.path.join(results_dir,
                                    "ER")
        
        if args.use_leitner:
            results_dir = os.path.join(results_dir,
                                        "DemoteFirst" if args.demote_to_first_deck else "DemotePrevious",
                                        "ltnmodel-"+args.ltn_model,
                                        args.lt_sampling_mode,
                                        "schedtype-"+args.er_lq_scheduler_type,
                                        "ltmode-"+args.lt_queue_mode,
                                        args.order_lst,
                                        "ndecks-"+str(args.num_decks),
                                        "sample-"+str(args.sample_batch_epoch),
                                        "update-"+str(args.update_batch_epoch))
        else:
            results_dir = os.path.join(results_dir,
                                    "ltmode-"+args.lt_queue_mode,
                                    args.order_lst)
            
        results_dir = os.path.join(results_dir,
                                "W_SCHEDULER_"+str(WARMUP_PERCENT*100)+"_WARMUP")
            
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        
    model_name, tokenizer_alias, model_trans_alias, config_alias = MODELS_dict[args.trans_model]
    model_load_alias = os.path.join(args.model_root, model_name)

    config = config_alias.from_pretrained(model_load_alias,
                                          output_hidden_states=True,
                                          output_attentions=True)
    
    tokenizer = tokenizer_alias.from_pretrained(model_load_alias,
                                                do_lower_case=True,
                                                do_basic_tokenize=False)
    
    model_trans = model_trans_alias.from_pretrained(model_load_alias,
                                                    config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                               
    args.lt_sampling_mode
    return model_trans, tokenizer, config, device, results_dir

def prepare_utils_data(args, tokenizer):
    dataset = MultiPurposeDataset(args,
                                  tokenizer)
    app_log.info("dataset.train_stream: {}".format(dataset.train_stream))
    return dataset, dataset.train_stream, dataset.dev_stream, dataset.test_stream
    # return dataset, dataset.train_stream[0], dataset.dev_stream[0], dataset.test_stream[args.languages[0]], dataset.train_stream[0]["size"]

def test_one_batch(examples, model):
    eval_outcomes = {}
    model.eval()
    with torch.no_grad():
        for example in examples:
            unique_id = example.unique_id
            sequence = LongTensor([example.input_ids]).cuda()
            attn_mask = LongTensor([example.attention_mask]).cuda()
            token_type = LongTensor([example.token_type_ids]).cuda()
            labels = LongTensor([example.label]).cuda()
            # model.to(torch.device("cpu"))
            output = model(sequence, attn_mask, token_type, 0, labels)
            labels = labels.to(torch.device("cpu"))
            acc = categorical_accuracy(output.logits["class"], labels)
            eval_outcomes.update({unique_id:acc})

    return eval_outcomes

def evaluate_nli(model, dataset, iterator):
    epoch_losses = []
    epoch_accs = []
    model.eval()
    with torch.no_grad():
        # for batch in iterator:
        # num_iter = iterator["size"]//BATCH_SIZE
        for _ in range(iterator["size"]):
            batch, _, _ = dataset.next_batch(1, iterator["examples"])
            batch = transfer_batch_cuda(batch, device)
            sequence = batch["input_ids"]
            attn_mask = batch["input_masks"]
            token_type = batch["token_type_ids"]
            labels = batch["labels"]
            output = model(sequence, attn_mask, token_type, 0, labels)
            labels = labels.to(torch.device("cpu"))
            acc = categorical_accuracy(output.logits["class"], labels)
            loss = output.loss["overall"]
            epoch_losses.append(loss.item())
            epoch_accs.append(acc.item())

    return np.mean(epoch_losses), np.mean(epoch_accs), epoch_losses, epoch_accs

def train_nli(model, dataset, iterator, optimizer, scheduler, lt_scheduler, er_lt_scheduler, epoch, args, i_task):
    update_batch_epoch = args.update_batch_epoch
    ltn_model = args.ltn_model
    use_er = args.use_er
    er_lq_scheduler_type = args.er_lq_scheduler_type
    use_leitner = args.use_leitner

    epoch_losses = []
    epoch_accs = []
    model.train()
    if use_leitner and er_lq_scheduler_type != "er":
        next_item_ids = lt_scheduler.next_items(epoch)
        scheduler_examples = AugmentedList([dataset.get_item_by_id(id_) for id_ in next_item_ids],
                                            shuffle_between_epoch=False)
        train_examples = scheduler_examples # 
        total_num = len(next_item_ids) #
    else:
        train_examples = iterator["examples"]
        total_num = iterator["size"] 
    
    num_iter = total_num//BATCH_SIZE
    left_over_batch = total_num %  BATCH_SIZE

    if use_er: 
        er_sample_freq = num_iter // 10 # 
        if use_leitner and er_lq_scheduler_type != "main":
            if i_task > 0 and epoch == 0:
                er_lt_scheduler.init_first_deck_er(items_l=lt_scheduler.decks[-1]) #
        else:
            memory = iterator["memory"] # 

    for step_num in range(num_iter):
        model.train()
        batch, batch_examples, _ = dataset.next_batch(BATCH_SIZE, train_examples)
        # app_log.info("example 0 text_a: %s | text_b: %s | label: %s", batch_examples[0].text_a, batch_examples[0].text_b, batch_examples[0].label) 

        batch = transfer_batch_cuda(batch, device)
        optimizer.zero_grad() # clear gradients first
        torch.cuda.empty_cache() # releases all unoccupied cached memory
        sequence = batch["input_ids"]
        attn_mask = batch["input_masks"]
        token_type = batch["token_type_ids"]
        label = batch["labels"]
        output = model(sequence, attn_mask, token_type, 0, label)
        label = label.to(torch.device("cpu"))
        acc = categorical_accuracy(output.logits["class"], label)
        loss = output.loss["overall"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(acc.item())

        if use_er and i_task > 0:
            if step_num % er_sample_freq == 0:
                if er_lq_scheduler_type == "main":
                        for task_memory_id, task_memory in enumerate(memory):
                            er_batch, _ , _ = dataset.next_batch(BATCH_SIZE, task_memory)
                            optimizer.zero_grad()
                            er_batch = transfer_batch_cuda(er_batch, device)

                            er_sequence = er_batch["input_ids"]
                            er_attn_mask = er_batch["input_masks"]
                            er_token_type = er_batch["token_type_ids"]
                            er_label = er_batch["labels"]

                            er_outputs = model(er_sequence, er_attn_mask, er_token_type, 0, er_label)
                            er_loss = er_outputs.loss["overall"]
        
                            er_loss = er_loss.mean()
                            er_loss.backward()
                            optimizer.step()
                            scheduler.step()
                else: # ER or both 
                    for _ in range(i_task):
                        er_batch = er_lt_scheduler.next_items(epoch)

        if use_leitner and update_batch_epoch == "batch" and ltn_model == "ltn":
            eval_output = test_one_batch(batch_examples,
                                         model)
            
            lt_scheduler.place_items(eval_output)

        if use_er and use_leitner and er_lq_scheduler_type != "main" and update_batch_epoch == "batch":
            # Update the ER Scheduler
            er_examples = [dataset.get_item_by_id(id_) for id_ in er_lt_scheduler.all_items.keys()]
            eval_output = test_one_batch(er_examples,
                                         model)
            
            er_lt_scheduler.place_items(eval_output)

    if left_over_batch > 0:
        model.train()
        batch, batch_examples, _ = dataset.next_batch(left_over_batch, train_examples)
        batch = transfer_batch_cuda(batch, device)
        optimizer.zero_grad() # clear gradients first
        torch.cuda.empty_cache() # releases all unoccupied cached memory
        sequence = batch["input_ids"]
        attn_mask = batch["input_masks"]
        token_type = batch["token_type_ids"]
        label = batch["labels"]
        output = model(sequence, attn_mask, token_type, 0, label)
        label = label.to(torch.device("cpu"))
        acc = categorical_accuracy(output.logits["class"], label)
        loss = output.loss["overall"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(acc.item())

        if use_leitner and update_batch_epoch == "batch" and ltn_model == "ltn":
            eval_output = test_one_batch(batch_examples,
                                         model)
            
            lt_scheduler.place_items(eval_output)

    if use_leitner and update_batch_epoch == "epoch" and ltn_model == "ltn":
        _, all_examples, _ = dataset.next_batch(total_num, train_examples)
        eval_output = test_one_batch(all_examples,
                                     model)
        
        lt_scheduler.place_items(eval_output)

    if use_er and use_leitner and er_lq_scheduler_type != "main" and update_batch_epoch == "epoch":
        # Update the ER Scheduler
        er_examples = [dataset.get_item_by_id(id_) for id_ in er_lt_scheduler.all_items.keys()]
        eval_output = test_one_batch(er_examples,
                                     model)
        
        er_lt_scheduler.place_items(eval_output)


    return np.mean(epoch_losses), np.mean(epoch_accs), epoch_losses, epoch_accs, lt_scheduler, er_lt_scheduler, total_num

def test_nli_ltn_model(args, model_trans, device):
    ## Base Model
    app_log.info("Loading Base Model ...")
    model = TransNLI(args=args,
                     trans_model=model_trans,
                     num_labels=NUM_CLASSES,
                     num_tasks=-1,
                     eff_num_classes_task=-1,
                     device=device)
    
    model.to(device)

    ## Optimizer/Scheduler
    app_log.info("Optimizer/Scheduler ...")
    # optimizer = optim.Adam(model.parameters(),#filter(lambda p: p.requires_grad, model.parameters()),
    #                        lr=ADAM_LR,
    #                        eps=ADAM_EPS,
    #                        betas=(BETA_1, BETA_2))
    optimizer = optim.AdamW(model.parameters(), lr=ADAM_LR, eps=ADAM_EPS) #lr=2e-5,eps=1e-6)

    ##  Dataset
    app_log.info("Loading the Dataset ...")
    dataset, train_iterator, valid_iterator, test_iterator = prepare_utils_data(args, tokenizer)

    #### return dataset, dataset.train_stream, dataset.dev_stream, dataset.test_stream
    #### return dataset, dataset.train_stream[0], dataset.dev_stream[0], dataset.test_stream[args.languages[0]], dataset.train_stream[0]["size"]
    langs_order = dataset.order_lst
    lt_scheduler = SpacedRepetitionModel.LeitnerQueue(args)
    er_lt_scheduler = SpacedRepetitionModel.LeitnerQueue(args)

    if args.rand_perf:
        test_accs = {lang: [] for lang in langs_order}
        for lang in langs_order:
            test_loss, test_acc, test_loss_l, test_acc_l = evaluate_nli(model, dataset, test_iterator[lang])
            test_accs[lang].append(test_acc)
            
        with open(os.path.join(results_dir, "random_perf_test.json"), "w") as output_file:
            json.dump(test_accs, output_file)
        exit(0)

    metrics = []
    app_log.info("Training steps in the stream: %d", len(dataset.train_stream))
    for i_train in range(len(dataset.train_stream)):

        def get_scheduler(optimizer, warmup_steps):
            t_total = train_iterator[i_train]["size"] // GRAD_ACC * N_EPOCHS
            # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
            return scheduler
    
        app_log.info("********* Training i_train: %d", i_train)
        train_data_len = train_iterator[i_train]["size"] 
        total_steps = math.ceil(N_EPOCHS*train_data_len*1./BATCH_SIZE)
        warmup_steps = int(total_steps*WARMUP_PERCENT)
        scheduler = get_scheduler(optimizer, warmup_steps)
        # scheduler = None

        ## Leitner Queues
        # if args.use_leitner:
        if args.lt_queue_mode in ["mono", "cont-mono"]:
            lt_scheduler = SpacedRepetitionModel.LeitnerQueue(args)
        
        lt_scheduler.init_first_deck(dataset=dataset,
                                     train_examples=train_iterator[i_train]["examples"], 
                                     nb_examples=train_data_len)
        # else:
        #     lt_scheduler = None

        best_valid_acc = 0
        i_best_test = N_EPOCHS - 1
        train_accs = []
        train_losses = []
        valid_accs = []
        valid_losses = []
        test_accs = {lang: [] for lang in langs_order}
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss, train_acc, train_acc_l, train_loss_l, lt_scheduler, er_lt_scheduler, total_num = train_nli(model, dataset, train_iterator[i_train], optimizer, scheduler, lt_scheduler, er_lt_scheduler, epoch, args, i_train)
            app_log.info("LTN_SCHEDULER REPRESENTATION: %s", lt_scheduler.rep_sched())
            app_log.info("ER_LTN_SCHEDULER REPRESENTATION: %s", er_lt_scheduler.rep_sched())
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            _, _, train_loss_l_batch_1, train_acc_l_batch_1 = evaluate_nli(model, dataset, train_iterator[i_train])
            valid_loss, valid_acc, valid_loss_l, valid_acc_l = evaluate_nli(model, dataset, valid_iterator[i_train])
            if args.ltn_model == "rbf":
                lt_scheduler.place_items(valid_acc_l, valid_loss_l, train_loss_l_batch_1, train_acc_l_batch_1)
            valid_accs.append(valid_acc)
            valid_losses.append(valid_acc)
            for lang in langs_order:
                test_loss, test_acc, test_loss_l, test_acc_l = evaluate_nli(model, dataset, test_iterator[lang])
                test_accs[lang].append(test_acc)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                i_best_test = epoch
            app_log.info("Epoch: %d | Epoch Time: %dm %ds | total_num: %d",epoch, epoch_mins, epoch_secs, total_num)
            app_log.info("\t Train Loss: %.3f | Train Acc: %.2f", train_loss, train_acc*100)
            app_log.info("\t Val. Loss: %.3f | Val. Acc: %.2f", valid_loss, valid_acc*100)
            for lang in langs_order:
                app_log.info("\t Test. Lang: %s Loss: %.3f | Test. Acc: %.2f", lang, test_loss, test_accs[lang][epoch]*100)

            app_log.info("Counts of decks: {}".format(lt_scheduler.print_count_decks()))

        app_log.info("Best Val. Acc: {} | Corresponding Best Test Acc: {}".format(best_valid_acc, {lang: test_accs[lang][i_best_test] for lang in langs_order}))
    
        metrics.append({"train": {"loss": train_losses, "acc": train_accs}, "val": {"loss": valid_losses, "acc": valid_accs}, "test": {"acc": test_accs}})
    return model, metrics
    
print("Parsing arguments ...")
args = get_arguments()
args.languages = args.order_lst.split("_")

print("Setup config ...")
model_trans, tokenizer, config, device, results_dir = setup_config(args)

SpacedRepetitionModel = importlib.import_module('schedulers.' + args.ltn_model)

app_log = logger(os.path.join(results_dir, args.log_file))
app_log.info("Saving to results_dir %s", results_dir)

stdoutOrigin = sys.stdout
sys.stdout = open(os.path.join(results_dir, args.log_file), "w")
app_log.info("Initializing stats file")
logstats.init(os.path.join(results_dir, args.stats_file))
config_path = os.path.join(results_dir, 'config.json')
logstats.add_args('config', args)
logstats.write_json(vars(args), config_path)

app_log.info("NLI Model Training/Validation/Testing Execution ...")
model, metrics = test_nli_ltn_model(args, model_trans, device)
with open(os.path.join(results_dir, "metrics.json"), "w") as output_file:
    json.dump(metrics, output_file)

sys.stdout.close()
sys.stdout = stdoutOrigin