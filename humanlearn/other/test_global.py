import sys, os
import argparse

sys.path.append("/home1/mmhamdi/x-continuous-learning")
import src.logstats as logstats
from src.transformers_config import MODELS_dict
from src.basemodels.transNLI import TransNLI
from src.data_utils import MultiPurposeDataset, AugmentedList
import src.schedulers.ltn as SpacedRepetitionModel
from src.utils import epoch_time, categorical_accuracy, transfer_batch_cuda, logger

SEED = 42
def get_arguments():
    parser = argparse.ArgumentParser("./test_base_models.py", description='Testing of transNLU module.')
    parser.add_argument('--data_root', type=str, default="/project/jonmay_231/meryem/Datasets/NLI/XNLI/")
    parser.add_argument('--task_name', type=str, default="nli")
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
    parser.add_argument('--trans_model', type=str, default="BertBaseMultilingualCased")
    parser.add_argument('--model_root', type=str, default="/project/jonmay_231/meryem/Models/")
    parser.add_argument('--use_slots', action="store_true")
    parser.add_argument("--use_crf", action="store_true") 
    parser.add_argument('--use_adapters', action="store_true")
    parser.add_argument('--multi_head_out', action="store_true")
    parser.add_argument('--adapter_type', type=str, default="")

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

def add_ltn(el, lt_scheduler, dataset, train_iterator, i_train, train_data_len):
    el += 1
    lt_scheduler.init_first_deck(dataset=dataset,
                                 train_examples=train_iterator[i_train]["examples"], 
                                 nb_examples=train_data_len)

def prepare_utils_data(args, tokenizer):
    dataset = MultiPurposeDataset(args,
                                  tokenizer)
    print("dataset.train_stream:", dataset.train_stream)
    return dataset, dataset.train_stream, dataset.dev_stream, dataset.test_stream

def test_func():
    args = get_arguments()
    lt_scheduler = SpacedRepetitionModel.LeitnerQueue(args)

    model_name, tokenizer_alias, model_trans_alias, config_alias = MODELS_dict[args.trans_model]
    model_load_alias = os.path.join(args.model_root, model_name)

    tokenizer = tokenizer_alias.from_pretrained(model_load_alias,
                                                do_lower_case=True,
                                                do_basic_tokenize=False)

    dataset, train_iterator, valid_iterator, test_iterator = prepare_utils_data(args, tokenizer)

    i_train = 0
    train_data_len = train_iterator[i_train]["size"] 
    
    print("BEFORE lt_scheduler.rep_sched():", lt_scheduler.rep_sched())
    ltn = 0
    add_ltn(ltn, lt_scheduler, dataset, train_iterator, i_train, train_data_len)

    print("AFTER lt_scheduler.rep_sched():", lt_scheduler.rep_sched())

test_func()
