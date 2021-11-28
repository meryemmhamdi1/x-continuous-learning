import os
import pickle
import logging
from sklearn.metrics import f1_score, precision_score, recall_score
from consts import INTENT_TYPES, SLOT_TYPES
from contlearnalg.memory.ERMemory import Example
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR as SchedulerLR
import configparser
from tqdm import tqdm


def set_optimizer(args, parameters):
    optimizer = Adam(filter(lambda p: p.requires_grad, parameters),
                     betas=(args.beta_1, args.beta_2),
                     eps=args.adam_eps,
                     lr=args.adam_lr)

    scheduler = SchedulerLR(optimizer,
                            step_size=args.step_size,
                            gamma=args.gamma)

    return optimizer, scheduler


def read_saved_pickle(checkpoint_dir,
                      task_i,
                      obj="grads"):

    with open(os.path.join(checkpoint_dir, "pytorch_"+obj+"_"+str(task_i)), "rb") as file:
        read_obj = pickle.load(file)

    return read_obj


def name_in_list(list, name):
    for el in list:
        if el in name:
            return True
    return False


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


def format_store_grads(pp,
                       grad_dims,
                       cont_comp,
                       checkpoint_dir=None,
                       tid=-1,
                       store=True):
    """
        This stores parameter gradients of one task at a time.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads = torch.Tensor(sum(grad_dims))
    grads.fill_(0.0)
    cnt = 0
    for n, p in pp:
        if name_in_list(cont_comp, n):
            if p.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(p.grad.data.view(-1))
            cnt += 1

    if store:
        with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(tid)), "wb") as file:
            pickle.dump(grads, file)

    return grads


def logger(log_file):
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                      datefmt='%d/%m/%Y %H:%M:%S')


    #Setup File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    #Setup Stream Handler (i.e. console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.INFO)

    #Get our logger
    app_log = logging.getLogger('root')
    app_log.setLevel(logging.INFO)

    #Add both Handlers
    app_log.addHandler(file_handler)
    app_log.addHandler(stream_handler)
    return app_log


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def nlu_evaluation(dataset,
                   memory,
                   cont_learn_alg,
                   dataset_test,
                   nb_examples,
                   model,
                   use_slots,
                   train_idx,
                   test_idx,
                   args,
                   app_log,
                   device,
                   name,
                   out_path=None,
                   verbose=False,
                   prior_mbert=None,
                   prior_intents=None,
                   prior_slots=None,
                   prior_adapter=None):

    app_log.info("Evaluating on i_task: %d", test_idx)

    if prior_mbert or prior_intents or prior_slots or prior_adapter:

        model_dict = model.state_dict()

        if prior_mbert:
            app_log.info("Using prior_mbert")
            ### 1. wanted keys, values are in trans_model
            trans_model_dict = {"trans_model."+k: v for k, v in prior_mbert.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(trans_model_dict)

        if prior_intents:
            app_log.info("Using prior_intents")
            # TODO double check the naming with test_idx
            ### 1. wanted keys, values are in trans_model
            if "cil" in args.setup_opt:
                intent_classifier_dict = {"intent_classifier."+str(test_idx)+"."+k: v for k, v in prior_intents.items()}
            else:
                intent_classifier_dict = {"intent_classifier."+k: v for k, v in prior_intents.items()}
            ### 2. overwrite entries in the existing state dict
            model_dict.update(intent_classifier_dict)

        if prior_slots:
            app_log.info("Using prior_slots")
            ### 1. wanted keys, values are in trans_model
            slot_classifier_dict = {"slot_classifier."+k: v for k, v in prior_slots.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(slot_classifier_dict)

        if prior_adapter:
            adapter_norm_before_dict = {"adapter."+k: v for k, v in prior_adapter.items()}

            ### 2. overwrite entries in the existing state dict
            model_dict.update(adapter_norm_before_dict)

        ### 3. load the new state dict
        model.load_state_dict(model_dict)

    intent_corrects = 0
    sents_text = []

    intents_true = []
    intents_pred = []

    slots_true = []
    slots_pred = []

    slots_true_all = []
    slots_pred_all = []

    for _ in tqdm(range(nb_examples)):

        batch_one, text \
            = dataset.next_batch(1, dataset_test)

        input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts, _ = batch_one

        if device != torch.device("cpu"):
            input_ids = input_ids.cuda()
            lengths = lengths.cuda()
            input_masks = input_masks.cuda()
            intent_labels = intent_labels.cuda()
            slot_labels = slot_labels.cuda()

        if train_idx > 0 and name != "dev":
            if args.cont_learn_alg == "mbpa":
                """ Local adaptation of MbPA """
                # q = Example(embed=model.get_embeddings(input_ids, input_masks),
                #             x={"input_ids": input_ids,
                #                "token_type_ids": token_type_ids,
                #                "input_masks": input_masks,
                #                "lengths": lengths,
                #                "input_texts": input_texts},
                #             y_intent=intent_labels,
                #             y_slot=slot_labels,
                #             distance=0.0,
                #             task_id=test_idx)

                q = model.get_embeddings(input_ids, input_masks)[0]

                # eval_model = cont_learn_alg.forward(memory, q, train_idx, model) # Old this is up to train_idx taking into consideration all memory items in previously seen tasks
                if args.use_reptile:
                    if args.use_batches_reptile:
                        eval_model = cont_learn_alg.forward_reptile_many_batches(memory, q, train_idx, model, dataset)
                    else:
                        eval_model = cont_learn_alg.forward_reptile_one_batch(memory, q, train_idx, model, dataset)
                else:
                    eval_model = cont_learn_alg.forward(memory, q, train_idx, model, dataset) # this is taking into consideration only the task we are testing from assuming we know that task.
            else:
                eval_model = model
        else:
            eval_model = model

        eval_model.eval()
        # TODO test this in particular
        # TODO do we change anything at all in the original model just to make sure?
        if use_slots:
            with torch.no_grad():
                intent_logits, slot_logits, _, intent_loss, slot_loss, loss, pooled_output \
                    = eval_model(input_ids=input_ids,
                                 input_masks=input_masks,
                                 train_idx=test_idx,
                                 lengths=lengths,
                                 intent_labels=intent_labels,
                                 slot_labels=slot_labels)

            # Slot Golden Truth/Predictions
            true_slot = slot_labels[0]

            slot_logits = [slot_logits[j, :length].data.numpy() for j, length in enumerate(lengths)]
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
            with torch.no_grad():
                intent_logits, intent_loss, loss, pooled_output = eval_model(input_ids=input_ids,
                                                                             input_masks=input_masks,
                                                                             train_idx=test_idx,
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
                    app_log.info("Sent : %s", sents_text[i][0])
                    app_log.info(" True Intent: ")
                    app_log.info(INTENT_TYPES[intents_true[i]])
                    app_log.info(" Intent Prediction :")
                    app_log.info(INTENT_TYPES[intents_pred[i]])
                    app_log.info(" True Slots: ")
                    app_log.info(" ".join(slots_true[i]))
                    app_log.info(" Slot Prediction:")
                    app_log.info(" ".join(slots_pred[i]))

                text = sents_text[i][0] + "\t" + INTENT_TYPES[intents_true[i]] + "\t" + INTENT_TYPES[intents_pred[i]] \
                    + "\t" + " ".join(slots_true[i]) + "\t" + " ".join(slots_pred[i])
                writer.write(text+"\n")

    if verbose:
        app_log.info(test_idx)
        app_log.info(" -----------intents_true:")
        app_log.info(set(intents_true))
        app_log.info(" -----------intents_pred:")
        app_log.info(set(intents_pred))

    if nb_examples > 0:
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
    else:
        intent_accuracy = 0.0
        intent_prec = 0.0
        intent_rec = 0.0
        intent_f1 = 0.0
        if use_slots:
            slot_prec = 0.0
            slot_rec = 0.0
            slot_f1 = 0.0
            return intent_accuracy, intent_prec, intent_rec, intent_f1, slot_prec, slot_rec, slot_f1

        return intent_accuracy, intent_prec, intent_rec, intent_f1


def evaluate_report(dataset,
                    memory,
                    cont_learn_alg,
                    data_stream,
                    model,
                    train_task,  # lang or subtask
                    train_idx,
                    test_task,  # lang or subtask
                    test_idx,
                    num_steps,
                    writer,
                    args,
                    app_log,
                    device,
                    name,
                    out_path=None,
                    verbose=False,
                    prior_mbert=None,
                    prior_intents=None,
                    prior_slots=None,
                    prior_adapter=None):

    outputs = nlu_evaluation(dataset,
                             memory,
                             cont_learn_alg,
                             data_stream["examples"],
                             data_stream["size"],
                             model,
                             args.use_slots,
                             train_idx,
                             test_idx,
                             args,
                             app_log,
                             device,
                             name,
                             out_path=out_path,
                             verbose=verbose,
                             prior_mbert=prior_mbert,
                             prior_intents=prior_intents,
                             prior_slots=prior_slots,
                             prior_adapter=prior_adapter)

    output_text_format = "----size=%d, test_index=%d, and task=%s" % (data_stream["size"],
                                                                      test_idx,
                                                                      test_task)

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

        metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_slot_prec_'+test_task+'_'+str(test_idx): slot_prec})
        metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_slot_rec_'+test_task+'_'+str(test_idx): slot_rec})
        metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_slot_f1_'+test_task+'_'+str(test_idx): slot_f1})

    metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_intent_acc_'+test_task+'_'+str(test_idx): intent_acc})
    metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_intent_prec_'+test_task+'_'+str(test_idx): intent_prec})
    metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_intent_rec_'+test_task+'_'+str(test_idx): intent_rec})
    metrics.update({train_task+'_'+str(train_idx)+'_'+name+'_intent_f1_'+test_task+'_'+str(test_idx): intent_f1})

    output_text_format += " INTENTS perf: (acc: %f, prec: %f, rec: %f, f1: %f)" % (round(intent_acc*100, 1),
                                                                                   round(intent_prec*100, 1),
                                                                                   round(intent_rec*100, 1),
                                                                                   round(intent_f1*100, 1))

    app_log.info(output_text_format)
    for k, v in metrics.items():
        writer.add_scalar(k, v, num_steps)

    return metrics, avg_perf


def get_config_params(args):
    paths = configparser.ConfigParser()
    paths.read('scripts/paths.ini')

    # location = "ENDEAVOUR"
    location = "LOCAL"

    args.data_root = str(paths.get(location, "DATA_ROOT"))
    args.trans_model = str(paths.get(location, "TRANS_MODEL"))
    args.out_dir = str(paths.get(location, "OUT_DIR"))

    params = configparser.ConfigParser()
    print('scripts/hyperparam'+args.param_tune_idx+'.ini')
    params.read('scripts/hyperparam'+args.param_tune_idx+'.ini')

    args.batch_size = int(params.get("HYPER", "BATCH_SIZE"))
    args.epochs = int(params.get("HYPER", "EPOCHS"))
    args.adam_lr = float(params.get("HYPER", "ADAM_LR"))
    args.adam_eps = float(params.get("HYPER", "ADAM_EPS"))
    args.beta_1 = float(params.get("HYPER", "BETA_1"))
    args.beta_2 = float(params.get("HYPER", "BETA_2"))
    args.epsilon = float(params.get("HYPER", "EPSILON"))
    args.step_size = float(params.get("HYPER", "STEP_SIZE"))
    args.gamma = float(params.get("HYPER", "GAMMA"))
    args.test_steps = int(params.get("HYPER", "TEST_STEPS"))
    args.num_intent_tasks = int(params.get("HYPER", "NUM_INTENT_TASKS"))  # only in case of CILIA setup
    args.num_lang_tasks = int(params.get("HYPER", "NUM_LANG_TASKS"))  # only in case of CILIA setup
    args.test_steps = int(params.get("HYPER", "TEST_STEPS"))

    return args
