import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        "./test_base_models.py", description="Testing of transNLU module."
    )
    # DATA ARGUMENTS
    parser.add_argument(
        "--task_name", type=str, default="nli", choices=["tod", "nli", "qa", "ner"]
    )

    parser.add_argument(
        "--data_name",
        type=str,
        default="xnli",
        choices=["schuster", "mtop", "multiatis", "xnli", "tydiqa", "panx"],
        help="Whether it is mtop, matis, nli, or tydiqa.",
    )

    parser.add_argument(
        "--data_format", type=str, default="txt", choices=["tsv", "json", "txt"]
    )

    ## QA Dataset Related Arguments
    parser.add_argument(
        "--do_lower_case",
        help="Set this flag if you are using an uncased model.",
        action="store_true",
    )

    parser.add_argument(
        "--verbose_logging",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
        action="store_true",
    )

    parser.add_argument(
        "--version_2_with_negative",
        help="If true, the SQuAD examples contain some that do not have an answer.",
        action="store_true",
    )

    parser.add_argument(
        "--null_score_diff_threshold",
        help="If null_score - best_non_null is greater than the threshold predict null.",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--save_per_hop",
        help="Whether to save the model at the end of a hop",
        action="store_true",
    )

    # SETUP OPTIONS
    parser.add_argument("--setup_opt", type=str, default="cll-er_kd")
    parser.add_argument("--use_mono", action="store_true")
    parser.add_argument("--rand_perf", action="store_true")

    parser.add_argument(
        "--order_lst", type=str, default="en"
    )  # default="en_zh_vi_ar_tr_bg_el_ur")
    parser.add_argument(
        "--languages", nargs="+", default=["en"]
    )  # default=["zh", "vi", "ar", "tr", "bg", "el", "ur", "en" ])

    parser.add_argument("--order_class", type=int, default=0)
    parser.add_argument("--order_lang", type=int, default=0)

    # MODEL ARGUMENTS
    parser.add_argument(
        "--trans_model",
        type=str,
        default="BertBaseMultilingualCased",
        choices=["BertBaseMultilingualCased", "XLMRoberta_base"],
    )

    parser.add_argument("--use_slots", action="store_true")
    parser.add_argument("--use_crf", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    # parser.add_argument("--do_lower_case", help="Set this flag if you are using an uncased model.",
    #                     action="store_true")

    # NOT REALLY NEEDED WAS USED FOR MODEL EXPANSION
    parser.add_argument("--use_adapters", action="store_true")
    parser.add_argument("--multi_head_out", action="store_true")
    parser.add_argument("--adapter_type", type=str, default="")
    parser.add_argument(
        "--multi_head_in",
        help="Whether to use multiple heads that would imply multiple subtask/language-specific heads at the input level.",
        action="store_true",
    )

    ## LEITNER QUEUE ARGUMENTS
    parser.add_argument("--use_leitner", action="store_true")
    parser.add_argument("--demote_to_first_deck", action="store_true")
    parser.add_argument("--num_decks", type=int, default=5)
    parser.add_argument(
        "--lt_sampling_mode",
        choices=["fifo", "rand", "rand-prop"],
        type=str,
        default="fifo",
    )
    parser.add_argument(
        "--lt_queue_mode",
        choices=["mono", "cont-mono", "cont-multi", "multi", "multi-incr-cll"],
        default="mono",
    )
    parser.add_argument("--ltn_model", choices=["ltn", "rbf"], default="ltn")
    parser.add_argument(
        "--sample_batch_epoch",
        choices=["batch", "epoch"],
        default="epoch",
        help="Whether to sample training examples at the end of each batch or epoch",
    )
    parser.add_argument(
        "--update_everything", choices=["everything", "current"], default="current"
    )
    parser.add_argument(
        "--update_batch_epoch",
        choices=["batch", "epoch"],
        default="batch",
        help="Whether to update leitner queue at the end of each batch or epoch",
    )
    parser.add_argument(
        "--er_lq_scheduler_mode",
        help="The mode used for training ER with LQ whether we use a sampling proportion that we fix or we use a frequency proportion in the interleaving of the two losses",
        choices=["sample_prop", "interleave_prop"],
        type=str,
        default="interleave_prop",
    )

    parser.add_argument("--ltn_promote_thresh", type=float, default=1.0)

    parser.add_argument(
        "--nu", help="confidence parameter for RBF Algorithm", type=float, default=0.5
    )

    parser.add_argument(
        "--kern", help="Kernel function for RBF Algorithm", type=str, default="cos"
    )

    ### ER Parameters
    parser.add_argument("--use_er", action="store_true")

    parser.add_argument("--use_k_means", action="store_true")

    parser.add_argument(
        "--max_mem_sz",
        help="The maximum size of the memory to be used in replay",
        type=int,
        default=1000,  # 10105,
    )

    parser.add_argument(
        "--use_er_only",
        help="Whether to do the backward over both main and memory batch or just the memory batch every 100 steps",
        action="store_true",
    )

    parser.add_argument(
        "--use_wipe", help="Whether to wipe memory", action="store_true"
    )

    parser.add_argument(
        "--er_lq_scheduler_type",
        choices=["er-only", "er-main", "er-both"],
        type=str,
        default="er-main",
    )

    parser.add_argument(
        "--er_strategy_prop",
        type=float,
        default=0.0,
        help="If 0.0 is picked then we just take whatever is in the deck, otherwise, we go by percentages",
    )

    parser.add_argument(
        "--er_strategy",
        choices=[
            "hard",
            "easy",
            "extreme",
            "balanced",
            "exponential",
            "random",  # could be used in Baseline ER(Rand) or ER(LTN)
            "per_movement",
            "equal-lang",  # could be used in Baseline MER(Rand) or MER(LTN)
        ],
        type=str,
        default="random",
        help="The strategy type used in populating ER memory:"
        "- hard: Hardest examples: 1st deck 100%"
        "- easy: Easiest examples: 5st deck 100%"
        "- Extreme examples:  1st deck 50% and 5st deck 50%"
        "- Balanced: from all decks"
        "- Exponential: "
        "- Proportional: following the same distribution as the queues."
        "- Random: from all decks we pick something random"
        "- Per movement: keep track of items movements across queues"
        "- equal-lang: like the ER baseline with multiple multilingual leitner queues",
    )

    parser.add_argument(
        "--wipe_strategy",
        choices=[
            "hard",
            "easy",
            "extreme",
            "balanced",
            "exponential",
            "random",  # could be used in Baseline ER(Rand) or ER(LTN)
            "per_movement",
            "equal-lang",  # could be used in Baseline MER(Rand) or MER(LTN)
        ],
        type=str,
        default="random",
    )

    args = parser.parse_args()
    return args
