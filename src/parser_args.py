def add_path_arguments(parser):
    path_params = parser.add_argument_group("Path Parameters")
    path_params.add_argument(
        "--data_root", help="Root directory of the dataset.", type=str, default=""
    )

    path_params.add_argument(
        "--model_root",
        help="Path to the root directory hosting the trans model, if offline.",
        type=str,
        default="",
    )

    path_params.add_argument(
        "--out_dir",
        help="The root directory of the results for this project.",
        type=str,
        default="",
    )

    path_params.add_argument(
        "--stats_file",
        help="Filename of the stats file.",  # TODO CHECK WHAT THIS DOES EXACTLY
        type=str,
        default="stats.txt",
    )

    path_params.add_argument(
        "--log_file",
        help="Filename of the log file.",  # TODO DO PROPER CHECKPOINTING
        type=str,
        default="log.txt",
    )


def add_setup_arguments(parser):
    setup_params = parser.add_argument_group("Setup Scenarios Parameters")
    setup_params.add_argument(
        "--setup_opt",
        help="The different setup scenarios to pick from:"
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
        "* multi:      Multilingually training one model on all languages at the same time."
        "* cll-er_kd: Cross-CLL optimized specifically for Experience Replay and Knowledge distillation."
        "* multi-equal: Multilingually training one model on a balanced version of all languages"
        "* cll-equal: cll trained with a balanced version of the dataset."
        "* cll-equal-er_kd: cll-er_kd trained with a balanced version of the dataset."
        "* cll-n-ways: cll under n-ways setup."
        "* cll-k-shots: cll under k-shot setup",
        choices=[
            "cil",
            "cil-other",
            "multi-incr-cil",
            "cll",
            "multi-incr-cll",
            "cil-ll",
            "multi",
            "multi-equal",
            "cll-er_kd",
            "cll-equal",
            "cll-equal-er_kd",
            "cll-n-ways",
            "cll-k-shots",
        ],
        type=str,
        default="cll-er_kd",
    )

    setup_params.add_argument(
        "--order_lst",
        help="Specific order for subtasks and languages: list of languages "
        "or subtasks delimited by underscores.",
        type=str,
        default="",
    )

    setup_params.add_argument(
        "--order_lang",
        help="Standard ways/categories of ordering the languages:"
        "* 0: high2lowlang: decreasing order (from high to low-resource)."
        "* 1: low2highlang: increasing order (from low to high-resource)."
        "* 2: randomlang: random order.",
        type=int,
        default=0,
    )

    setup_params.add_argument(
        "--random_pred",
        help="Whether to predict directly the random initialization of the model"
        "when tested directly on the languages without any fine-tuning.",
        action="store_true",
    )

    # CILIA parameters: those are usually not used for the specific context of our cross-lingual continual learning framework
    setup_params.add_argument(
        "--cil_stream_lang",
        help="Which lang to work on for the CIL setup if it is picked.",
        type=str,
        default="en",
    )

    setup_params.add_argument(
        "--setup_cillia",
        help="Different ways of ordering mixture of both cll and cil:"
        "* intents: traversing subtasks horizontally over all intent "
        "           classes first then to languages."
        "* langs: traversing subtasks vertically over all languages first"
        "         then to classes.",
        type=str,
        default="intents",
    )

    setup_params.add_argument(
        "--order_class",
        help="Different ways of ordering the classes:"
        "* 0: high2lowclass: decreasing order (from high to low-resource)."
        "* 1: low2highclass: increasing order (from low to high-resource)."
        "* 2: randomclass: random order.",
        type=int,
        default=0,
    )

    setup_params.add_argument(
        "--num_class_tasks",
        help="The number of classes per task.",
        type=int,
        default=10,
    )  # specific to CIL related setups

    setup_params.add_argument(
        "--num_lang_tasks", help="The number of lang per task.", type=int, default=2
    )  # specific to CILIA related setups


def add_dataset_arguments(parser):
    dataset_params = parser.add_argument_group("Dataset Options")

    dataset_params.add_argument(
        "--task_name",
        help="The name of the task.",
        choices=["tod", "nli", "qa", "ner"],
        type=str,
        default="tod",
    )

    dataset_params.add_argument(
        "--data_name",
        help="Whether it is mtop, matis, nli, or tydiqa.",
        choices=["schuster", "jarvis", "mtop", "multiatis", "xnli", "tydiqa", "panx"],
        type=str,
        default="mtop",
    )

    dataset_params.add_argument(
        "--data_format",
        help="Whether it is tsv (MTOD), json, or txt (MTOP).",
        choices=["txt", "tsv", "json"],
        type=str,
        default="txt",
    )

    dataset_params.add_argument(
        "--languages",
        help="Train/Test languages list.",
        nargs="+",
        default=["de", "en", "es", "fr", "hi", "th"],
    )

    dataset_params.add_argument(
        "--xnli_max_length",
        help="For XNLI: the maximum number of tokens.",
        type=int,
        default=512,
    )

    dataset_params.add_argument(
        "--xner_max_length",
        help="For XNER: the maximum number of tokens.",
        type=int,
        default=128,
    )

    dataset_params.add_argument(
        "--pad_token", help="For XNLI, pad token", type=int, default=0
    )

    dataset_params.add_argument(
        "--max_seq_length",
        help="For QA: the max total input sequence length after WordPiece tokenization. Sequences longer"
        "than this will be truncated, and sequences shorter than this will be padded.",
        type=int,
        default=384,
    )

    dataset_params.add_argument(
        "--max_query_length",
        help="For QA: the maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
        type=int,
        default=64,
    )

    dataset_params.add_argument(
        "--doc_stride",
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
        type=int,
        default=128,
    )

    dataset_params.add_argument(
        "--mask_padding_with_zero",
        help="Whether to pad with zeros",
        type=bool,
        default=True,
    )

    dataset_params.add_argument(
        "--pad_token_segment_id", help="mask padding token", type=int, default=0
    )

    dataset_params.add_argument(
        "--n_best_size",
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
        default=20,
        type=int,
    )

    dataset_params.add_argument(
        "--max_answer_length",
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
        type=int,
        default=30,
    )

    dataset_params.add_argument(
        "--do_lower_case",
        help="Set this flag if you are using an uncased model.",
        action="store_true",
    )

    dataset_params.add_argument(
        "--verbose_logging",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
        action="store_true",
    )

    dataset_params.add_argument(
        "--version_2_with_negative",
        help="If true, the SQuAD examples contain some that do not have an answer.",
        action="store_true",
    )

    dataset_params.add_argument(
        "--null_score_diff_threshold",
        help="If null_score - best_non_null is greater than the threshold predict null.",
        type=float,
        default=0.0,
    )


def add_base_model_arguments(parser):
    base_model_params = parser.add_argument_group("Base Model Parameters")
    base_model_params.add_argument(
        "--trans_model",
        help="Name of the Transformer encoder model.",
        choices=[
            "BertBaseMultilingualCased",
            "BertLarge",
            "BertBaseCased",
            "Xlnet_base",
            "Xlnet_large",
            "XLM",
            "DistilBert_base",
            "DistilBert_large",
            "Roberta_base",
            "Roberta_large",
            "XLMRoberta_base",
            "XLMRoberta_large",
            "ALBERT-base-v1",
            "ALBERT-large-v1",
            "ALBERT-xlarge-v1",
            "ALBERT-xxlarge-v1",
            "ALBERT-base-v2",
            "ALBERT-large-v2",
            "ALBERT-xlarge-v2",
            "ALBERT-xxlarge-v2",
        ],
        type=str,
        default="BertBaseMultilingualCased",
    )

    base_model_params.add_argument(
        "--use_slots",
        help="If true, optimize for slot filling loss too.",
        action="store_true",
    )

    base_model_params.add_argument(
        "--use_crf", help="If called, use CRF.", action="store_true"
    )

    base_model_params.add_argument(
        "--use_mono", help="Whether to train monolingually.", action="store_true"
    )

    base_model_params.add_argument(
        "--epochs", help="The total number of epochs.", type=int, default=10
    )

    base_model_params.add_argument(
        "--dev_steps",
        help="The total number of epochs to evaluate the model on the dev.",
        type=int,
        default=200,
    )  # TODO DEV IS EVALUATED ON ONLY AFTER EACH EPOCH

    base_model_params.add_argument(
        "--test_steps",
        help="The total number of epochs to evaluate the model on the test.",
        type=int,
        default=200,
    )  # TODO THIS IS NOT USED CONSISTENTLY

    base_model_params.add_argument(
        "--batch_size",
        help="The total number of epochs for the model to evaluate.",
        type=int,
        default=32,
    )

    # Hyperparameters for the optimizer
    base_model_params.add_argument(
        "--adam_lr",
        help="The learning rate for Adam Optimizer.",
        type=float,
        default=1e-03,
    )

    base_model_params.add_argument(
        "--adam_eps", help="Epsilon for the Adam Optimizer.", type=float, default=1e-08
    )

    base_model_params.add_argument(
        "--beta_1", help="Beta_1 for the Adam Optimizer.", type=float, default=0.9
    )

    base_model_params.add_argument(
        "--beta_2", help="Beta_2 for the Adam Optimizer.", type=float, default=0.99
    )

    base_model_params.add_argument(
        "--step_size", help="The step size for the scheduler.", type=int, default=7
    )

    base_model_params.add_argument(
        "--gamma", help="Gamma for the scheduler.", type=float, default=0.1
    )

    base_model_params.add_argument("--seed", help="Random Seed.", type=int, default=42)

    base_model_params.add_argument(
        "--param_tune_idx",
        help="Index of the tuning hyperparameters.",
        type=str,
        default="0",
    )


def add_model_expansion_arguments(parser):
    model_expansion_params = parser.add_argument_group("Model Expansion Options")
    model_expansion_params.add_argument(
        "--multi_head_in",
        help="Whether to use multiple heads "
        "that would imply multiple subtask/language-specific "
        "heads at the input level.",
        action="store_true",
    )

    model_expansion_params.add_argument(
        "--emb_enc_subtask_spec",
        help="Which layer in the embeddings or the encoder "
        "to tune for each subtask/language independently.",
        # choices=["embeddings",
        #          "encoder.layer.0.",
        #          "encoder.layer.1.",
        #          "encoder.layer.2.",
        #          "encoder.layer.3.",
        #          "encoder.layer.4.",
        #          "encoder.layer.5.",
        #          "encoder.layer.6.",
        #          "encoder.layer.7.",
        #          "encoder.layer.8.",
        #          "encoder.layer.9.",
        #          "encoder.layer.10.",
        #          "encoder.layer.11.",
        #          "pooler",
        #          "all"],
        type=str,
        default="all",
    )

    model_expansion_params.add_argument(
        "--multi_head_out",
        help="Whether to use multiple heads in the outputs that "
        "would imply the use of different task-specific layers.",
        action="store_true",
    )

    model_expansion_params.add_argument(
        "--use_adapters", help="whether to use adapters.", action="store_true"
    )

    model_expansion_params.add_argument(
        "--use_pretrained_adapters",
        help="Whether to use pre-trained adapters",
        action="store_true",
    )

    model_expansion_params.add_argument(
        "--adapter_type",
        help="Which adapter to use.",
        type=str,
        default="MADX",
        choices=["Houlsby", "MADX"],
    )

    model_expansion_params.add_argument(
        "--adapter_layers",
        help="List of layers (delimited by underscore) to which adapters are applied.",
        type=str,
        default="0_1_2_3_4_5",
    )


def add_freezing_arguments(parser):
    freezing_params = parser.add_argument_group("Freezing Options")
    freezing_params.add_argument(
        "--freeze_trans",
        help="Whether to freeze all layers in Transformer encoder/embed.",
        action="store_true",
    )

    freezing_params.add_argument(
        "--freeze_first",
        help="Whether to freeze from the first subtask/language.",
        action="store_true",
    )

    freezing_params.add_argument(
        "--freeze_linear",
        help="Whether to freeze all task-specific layers.",
        action="store_true",
    )


def cont_learn_arguments(parser):
    ## CONTINUAL LEARNING ALGORITHMS
    cont_learn_params = parser.add_argument_group("Continuous Learning Options")
    cont_learn_params.add_argument(
        "--cont_learn_alg",
        help="vanilla fine-tuning or some continuous learning algorithm:"
        "(ewc, gem, mbpa, metambpa, etc) or vanilla if no specific"
        "continuous learning algorithm is used.",
        choices=[
            "vanilla",
            "ewc",
            "gem",
            "er",
            "mbpa",
            "kd-logits",
            "kd-rep",
            "reptile-er",
        ],
        # TODO to be covered next "mbpa", "metambpa", "icarl", "xdg", "si", "lwf", "gr", "rtf", "er"
        type=str,
        default="vanilla",
    )

    cont_learn_params.add_argument(
        "--cont_comp",
        help="Which component(s) in the model to focus on while learning "
        "during regularization or GEM",
        nargs="+",
        default=["trans_model gclassifier slot_classifier"],
    )

    ### for optimization (ewc, ewc online)
    cont_learn_params.add_argument(
        "--old_task_prop",
        help="The percentage of old tasks used in regularization " "or replay.",
        type=float,
        default=0.1,
    )

    cont_learn_params.add_argument(
        "--ewc_lambda",
        help="If ewc: lambda for regularization in ewc.",
        type=int,
        default=20,
    )

    cont_learn_params.add_argument(
        "--use_online",
        help="If ewc: Whether to use the online version of EWC or not.",
        action="store_true",
    )

    cont_learn_params.add_argument(
        "--gamma_ewc", help="If ewc: The percentage of decay.", type=int, default=0.01
    )

    ## for memory replay (er, mbpa, mbpa++) mbpa++ is automatically the case if sampling_type == random
    cont_learn_params.add_argument(
        "--max_mem_sz",
        help="The maximum size of the memory to be used in replay",
        type=int,
        default=60000,
    )

    cont_learn_params.add_argument(
        "--use_er_only",
        help="Whether to do the backward over both main and memory batch or just the memory batch every 100 steps",
        action="store_true",
    )

    cont_learn_params.add_argument(
        "--storing_type",
        help="The method used to store memory examples.",
        choices=["reservoir", "ring", "k-means", "mof"],
        type=str,
        default="ring",
    )

    cont_learn_params.add_argument(
        "--sampling_type",
        help="The method used to sample memory examples.",
        choices=["random", "near_n"],
        type=str,
        default="random",
    )

    cont_learn_params.add_argument(
        "--sampling_k",
        help="The number of examples to be sampled.",
        type=int,
        default=60000,
    )

    cont_learn_params.add_argument(
        "--adaptive_epochs",
        help="The number of adaptive epochs for which the model is "
        "to be trained on the retrieved batch in case of MbPA",
        type=int,
        default=5,
    )  # between 1 and 20 in the original paper

    cont_learn_params.add_argument(
        "--adaptive_adam_lr",
        help="The learning rate for Adaptive Adam Optimizer"
        "in the case of MbPA/MbPA++.",
        type=float,
        default=1e-03,
    )  # between 0.0 and 1.0 in the original paper

    cont_learn_params.add_argument(
        "--beta",
        help="beta in the regularization of the adaptative loss",
        type=float,
        default=0.001,
    )

    ### for optimization + memory (gem, agem)
    cont_learn_params.add_argument(
        "--use_a_gem", help="If gem: whether to use averaged gem.", action="store_true"
    )

    cont_learn_params.add_argument(
        "--a_gem_n",
        help="If gem: The number of examples in the averaged memory.",
        type=int,
        default=100,
    )


def add_checkpoint_arguments(parser):
    checkpointing_params = parser.add_argument_group("Checkpointing/logging Parameters")
    checkpointing_params.add_argument(
        "--verbose",
        help="If true, return golden labels and predictions to console.",
        action="store_true",
    )

    checkpointing_params.add_argument(
        "--save_dev_pred",
        help="If true, save the dev predictions.",
        action="store_true",
    )

    checkpointing_params.add_argument(
        "--save_test_every_epoch",
        help="If true, save test at the end of each epoch.",
        action="store_true",
    )

    checkpointing_params.add_argument(
        "--save_change_params",
        help="If true, save test at the end of each epoch.",
        action="store_true",
    )

    checkpointing_params.add_argument(
        "--no_debug",
        help="If true, save training and testing logs to disk.",
        action="store_true",
    )

    checkpointing_params.add_argument(
        "--save_model",
        help="Whether to save the model after training.",
        action="store_true",
    )


def add_meta_learning_setup(parser):
    meta_learning_params = parser.add_argument_group("Meta-learning Parameters")
    meta_learning_params.add_argument(
        "--alpha_reptile",
        help="alpha lr in the inner loop in reptile",
        type=float,
        default=0.001,
    )

    meta_learning_params.add_argument(
        "--beta_reptile",
        help="beta lr in the first outer loop reptile",
        type=float,
        default=0.01,
    )

    meta_learning_params.add_argument(
        "--gamma_reptile",
        help="gamma lr in the second outer loop reptile",
        type=float,
        default=0.01,
    )

    meta_learning_params.add_argument(
        "--num_batches_reptile", help="Number of batches per task", type=int, default=10
    )

    meta_learning_params.add_argument(
        "--use_reptile", help="Whether to use reptile or not", action="store_true"
    )

    meta_learning_params.add_argument(
        "--use_batches_reptile",
        help="Whether to use many batches per task or not",
        action="store_true",
    )


def add_spaced_repetition_setup(parser):
    spaced_repetition_params = parser.add_argument_group("Spaced Repetition Parameters")
    spaced_repetition_params.add_argument(
        "--use_processor_sharing",
        help="Whether to use the processor sharing service "
        "discipline or first-in-first-out",
        action="store_true",
    )

    spaced_repetition_params.add_argument(
        "--evaluate_one_batch",
        help="Whether to use one batch to update the leitner queues",
        action="store_true",
    )

    spaced_repetition_params.add_argument(
        "--eval_sched_freq",
        help="How frequently should we evaluate and update the Leitner Scheduler",
        type=int,
        default=10,
    )

    spaced_repetition_params.add_argument(
        "--warm_start_epochs",
        help="How many epochs should the rote training be",
        type=int,
        default=2,
    )

    spaced_repetition_params.add_argument(
        "--nu", help="confidence parameter", type=float, default=0.5
    )

    spaced_repetition_params.add_argument(
        "--kern", help="Kernel function", type=str, default="cos"
    )

    spaced_repetition_params.add_argument(
        "--num_decks", help="The number of decks used", type=int, default=5
    )

    spaced_repetition_params.add_argument(
        "--ltn_promote_thresh",
        help="The score threshold (as a percentage) beyond which we decide to promote to the next queue",
        type=float,
        default=1.0,
    )

    spaced_repetition_params.add_argument(
        "--ltn_scheduler_type",
        help="The type of leitner queue scheduler",
        type=str,
        default="ltn",
        choices=["ltn", "rbf"],
    )

    spaced_repetition_params.add_argument(
        "--use_leitner_queue",
        help="Whether to just use Leitner queues or just a random baseline",
        action="store_true",
    )

    spaced_repetition_params.add_argument(
        "--demote_to_first_deck",
        help="Whether to demote all the way to the first deck or just to the previous one ",
        action="store_true",
    )

    spaced_repetition_params.add_argument(
        "--lt_sampling_mode",
        help="The mode of sampling: if fifo then follow the usual leitner queue order otherwise rand means random",
        choices=["fifo", "rand"],
        type=str,
        default="fifo",
    )

    spaced_repetition_params.add_argument(
        "--use_cont_leitner_queue",
        help="Whether to continually add and reinitialize the leitner queue each time or to keep accumulating",
        action="store_true",
    )

    ## ER with LQ parameters
    spaced_repetition_params.add_argument(
        "--er_lq_scheduler_type",
        help="The mode used for training ER with LQ whether we use LQ just for ER loss or just for main loss or for both",
        choices=["er", "main", "both"],
        type=str,
        default="er",
    )

    spaced_repetition_params.add_argument(
        "--er_lq_scheduler_mode",
        help="The mode used for training ER with LQ whether we use a sampling proportion that we fix or we use a frequency proportion in the interleaving of the two losses",
        choices=["sample_prop", "interleave_prop"],
        type=str,
        default="interleave_prop",
    )

    spaced_repetition_params.add_argument(
        "--er_lq_scheduler_prop",
        help="The proportion of ER memory used in Leitner Queue in next_items.",
        type=float,
        default=1.0,
    )
