

def add_path_arguments(parser):
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

    path_params.add_argument("--param_tune_idx", help="Index of the tuning hyperparameters.", type=str, default="0")


def add_setup_arguments(parser):
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
                                       "cil-ll", "multi",
                                       "cll-er_kd"],
                              type=str, default="cll")

    setup_params.add_argument("--cil_stream_lang", help="Which lang to work on for the CIL setup if it is picked.",
                              default="en")

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

    setup_params.add_argument("--order_lst", help="Specific order for subtasks and languages: list of languages "
                                                  "or subtasks.",
                              type=str, default="")

    setup_params.add_argument("--setup_cillia", help="Different ways of ordering mixture of both cll and cil:"
                                                     "* intents: traversing subtasks horizontally over all intent "
                                                     "           classes first then to languages."
                                                     "* langs: traversing subtasks vertically over all languages first"
                                                     "         then to classes.",
                              type=str, default="intents")

    setup_params.add_argument('--random_pred', help="Whether to predict directly the random initialization of the model"
                                                    "when tested directly on the languages without any fine-tuning.",
                              action="store_true")


def add_dataset_arguments(parser):
    dataset_params = parser.add_argument_group("Dataset Options")
    dataset_params.add_argument("--data_format", help="Whether it is tsv (MTOD), json, or txt (MTOP).",
                                type=str, default="txt")

    dataset_params.add_argument("--languages", help="Train languages list.",
                                nargs="+", default=["de", "en", "es", "fr", "hi", "th"])

    dataset_params.add_argument("--num_intent_tasks", help="The number of intent per task.",
                                type=int, default=10)

    dataset_params.add_argument("--num_lang_tasks", help="The number of lang per task.",
                                type=int, default=2)


def add_base_model_arguments(parser):
    base_model_params = parser.add_argument_group("Base Model Parameters")
    base_model_params.add_argument("--trans_model", help="Name of the Transformer encoder model.",
                                   type=str, default="BertBaseMultilingualCased",
                                   choices=["BertBaseMultilingualCased", "BertLarge", "BertBaseCased",
                                            "Xlnet_base", "Xlnet_large", "XLM", "DistilBert_base",
                                            "DistilBert_large", "Roberta_base", "Roberta_large",
                                            "XLMRoberta_base", "XLMRoberta_large", "ALBERT-base-v1",
                                            "ALBERT-large-v1", "ALBERT-xlarge-v1", "ALBERT-xxlarge-v1",
                                            "ALBERT-base-v2", "ALBERT-large-v2", "ALBERT-xlarge-v2",
                                            "ALBERT-xxlarge-v2"])

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


def add_model_expansion_arguments(parser):
    model_expansion_params = parser.add_argument_group("Model Expansion Options")
    model_expansion_params.add_argument("--multi_head_in", help="Whether to use multiple heads "
                                                                "that would imply multiple subtask/language-specific "
                                                                "heads at the input level.",
                                        action="store_true")

    model_expansion_params.add_argument("--emb_enc_subtask_spec", help="Which layer in the embeddings or the encoder "
                                                                       "to tune for each subtask/language"
                                                                       " independently.",
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

    model_expansion_params.add_argument("--use_pretrained_adapters", help="Whether to use pre-trained adapters",
                                        action="store_true")

    model_expansion_params.add_argument("--adapter_type", help="Which adapter to use.",
                                        type=str, default="MADX", choices=["Houlsby", "MADX"])

    model_expansion_params.add_argument("--adapter_layers", help="List of layers to which adapters are applied.",
                                        type=str, default="0_1_2_3_4_5")


def add_freezing_arguments(parser):
    freezing_params = parser.add_argument_group("Freezing Options")
    freezing_params.add_argument("--freeze_trans", help="Whether to freeze all layers in Transformer encoder/embed.",
                                 action="store_true")

    freezing_params.add_argument("--freeze_first", help="Whether to freeze from the first subtask/language.",
                                 action="store_true")

    freezing_params.add_argument("--freeze_linear", help="Whether to freeze all task-specific layers.",
                                 action="store_true")


def cont_learn_arguments(parser):
    ## CONTINUAL LEARNING ALGORITHMS
    cont_learn_params = parser.add_argument_group("Continuous Learning Options")
    cont_learn_params.add_argument("--cont_learn_alg", help="vanilla fine-tuning or some continuous learning algorithm:"
                                                            "(ewc, gem, mbpa, metambpa, etc) or vanilla if no specific"
                                                            "continuous learning algorithm is used.",
                                   choices=["vanilla", "ewc", "gem", "er", "mbpa", "kd-logits", "kd-rep", "reptile-er"],
                                   # TODO to be covered next "mbpa", "metambpa", "icarl", "xdg", "si", "lwf", "gr", "rtf", "er"
                                   type=str, default="vanilla")

    cont_learn_params.add_argument("--cont_comp", help="Which component(s) in the model to focus on while learning "
                                                       "during regularization or replay",
                                   nargs="+", default=["trans intent slot"])

    ### for optimization (ewc, ewc online)
    cont_learn_params.add_argument("--old_task_prop", help="The percentage of old tasks used in regularization "
                                                           "or replay.",
                                   type=float, default=0.1)

    cont_learn_params.add_argument("--ewc_lambda", help="If ewc: lambda for regularization in ewc.",
                                   type=int, default=20)

    cont_learn_params.add_argument("--use_online", help="If ewc: Whether to use the online version of EWC or not.",
                                   action='store_true')

    cont_learn_params.add_argument("--gamma_ewc", help="If ewc: The percentage of decay.",
                                   type=int, default=0.01)

    ## for memory replay (er, mbpa, mbpa++) mbpa++ is automatically the case if sampling_type == random
    cont_learn_params.add_argument("--max_mem_sz", help="The maximum size of the memory to be used in replay",
                                   type=int, default=60000)

    cont_learn_params.add_argument("--storing_type", help="The method used to store memory examples.",
                                   choices=["reservoir", "ring", "k-means", "mof"],
                                   type=str, default="ring")

    cont_learn_params.add_argument("--sampling_type", help="The method used to sample memory examples.",
                                   choices=["random", "near_n"],
                                   type=str, default="random")

    cont_learn_params.add_argument("--sampling_k", help="The number of examples to be sampled.",
                                   type=int, default=60000)

    cont_learn_params.add_argument("--adaptive_epochs", help="The number of adaptive epochs for which the model is "
                                                             "to be trained on the retrieved batch in case of MbPA",
                                   type=int, default=5)  # between 1 and 20 in the original paper

    cont_learn_params.add_argument("--adaptive_adam_lr", help="The learning rate for Adaptive Adam Optimizer"
                                                              "in the case of MbPA/MbPA++.",
                                   type=float, default=1e-03)  # between 0.0 and 1.0 in the original paper

    cont_learn_params.add_argument("--beta", help="beta in the regularization of the adaptative loss",
                                   type=float, default=0.001)


    ### for optimization + memory (gem, agem)
    cont_learn_params.add_argument("--use_a_gem", help="If gem: whether to use averaged gem.",
                                   action="store_true")

    cont_learn_params.add_argument("--a_gem_n", help="If gem: The number of examples in the averaged memory.",
                                   type=int, default=100)


def add_checkpoint_arguments(parser):
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

def add_meta_learning_setup(parser):
    meta_learning_params = parser.add_argument_group("Meta-learning Parameters")
    meta_learning_params.add_argument("--alpha_reptile", help="alpha lr in the inner loop in reptile",
                                       type=float, default=0.001)

    meta_learning_params.add_argument("--beta_reptile", help="beta lr in the first outer loop reptile",
                                       type=float, default=0.01)

    meta_learning_params.add_argument("--gamma_reptile", help="gamma lr in the second outer loop reptile",
                                       type=float, default=0.01)

    meta_learning_params.add_argument("--num_batches_reptile", help="Number of batches per task",
                                   type=int, default=10)

    meta_learning_params.add_argument("--use_reptile", help="Whether to use reptile or not",
                                   action="store_true")

    meta_learning_params.add_argument("--use_batches_reptile", help="Whether to use many batches per task or not",
                                   action="store_true")
