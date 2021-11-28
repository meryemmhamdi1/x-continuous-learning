import pickle
import numpy as np
import pandas as pd
import os
from summarize_metrics import acc_avg, fwt_avg, fwt_avg_mono, bwt_avg, forget_avg, final_perf

root_dir = ""


""" CIL """


def cil(mode, class_order, model, cil_mode):
    # cil_mode => cil-other or cil
    # model => "" or ewc
    results_dir = root_dir+cil_mode+"/"
    if mode == "intents":
        results_dir += "Intents_only/"

    results_dir += "BertBaseMultilingualCased/high2lowlang/"+class_order+"/"+model+"SEED_42/"

    for lang in ["de", "en", "fr", "es", "hi", "th"]:
        intents = [[0.0 for _ in range(0, 12)] for _ in range(0, 12)]
        if mode == "nlu":
            slots = [[0.0 for _ in range(0, 12)] for _ in range(0, 12)]
            all_metrics = [[0.0 for _ in range(0, 24)] for _ in range(0, 12)]
        for i in range(0, 12):
            path = results_dir+"final_metrics_"+lang+"_trainsubtask_" + str(i) + ".pickle"
            if os.path.isfile(path):
                with open(path, "rb") as file:
                    data = pickle.load(file)

                for k in range(0, i+1):
                    if data[k]:
                        intents[i][k] = data[k][lang+"_"+str(i)+"_test_intent_acc_"+str(k)+"_"+lang]*100
                        if mode == "nlu":
                            slots[i][k] = data[k][lang+"_"+str(i)+"_test_slot_f1_"+str(k)+"_"+lang]*100
                            all_metrics[i][2*k] = intents[i][k]
                            all_metrics[i][2*k+1] = slots[i][k]

        intents_np = np.array(intents)
        pd.DataFrame(intents_np).to_csv(os.path.join(root_dir, lang + "_results_intents_matrix.csv"))

        if mode == "nlu":
            slots_np = np.array(slots)
            all_metrics_np = np.asarray(all_metrics)
            pd.DataFrame(slots_np).to_csv(os.path.join(root_dir, lang + "_results_slots_matrix.csv"))
            pd.DataFrame(all_metrics_np).to_csv(root_dir+lang+"_results_all_matrix.csv")


""" MULTI INCREMENTAL / MULTI"""


def multi_incremental(lang_order, ):
    results_dir = root_dir+"multi-incremental-lang/BertBaseMultilingualCased/" + \
                  lang_order+"/high2lowclass/SEED_42/metrics/"

    if lang_order == "high2lowlang":
        lang_list = ["en", "de", "fr", "hi", "es", "th"]
    elif lang_order == "low2highlang":
        lang_list = ["th", "es", "hi", "fr", "de", "en"]
    else:
        lang_list = lang_order.split("_")

    incr_lang_list = []
    lang_up = []
    for lang in lang_list:
        lang_up.append(lang)
        incr_lang_list.append("-".join(lang_up))

    intents = [[0.0 for _ in range(0, 6)] for _ in range(0, 6)]
    slots = [[0.0 for _ in range(0, 6)] for _ in range(0, 6)]
    all_metrics = [[0.0 for _ in range(0, 12)] for _ in range(0, 6)]
    for i, train_lang in enumerate(incr_lang_list):
        path = results_dir+"final_metrics_"+train_lang + ".pickle"
        if os.path.isfile(path):
            with open(path, "rb") as file:
                data = pickle.load(file)

                for k, lang in enumerate(lang_list):
                    intents[i][k] = data[lang][train_lang+"_"+str(i)+"_test_intent_acc_"+str(i)+"_"+lang]*100
                    slots[i][k] = data[lang][train_lang+"_"+str(i)+"_test_slot_f1_"+str(i)+"_"+lang]*100
                    all_metrics[i][2*k] = intents[i][k]
                    all_metrics[i][2*k+1] = slots[i][k]

    intents_np = np.array(intents)
    slots_np = np.array(slots)
    all_metrics_np = np.asarray(all_metrics)

    results_out = results_dir + "csv"

    if not os.path.isdir(results_out):
        os.mkdir(results_out)

    pd.DataFrame(intents_np).to_csv(os.path.join(results_out, "results_intents_all_seeds.csv"))
    pd.DataFrame(slots_np).to_csv(os.path.join(results_out, "results_slots_all_seeds.csv"))
    pd.DataFrame(all_metrics_np).to_csv(os.path.join(results_out, "results_matrix.csv"))


""" CLL """


def cll_multi_purpose(model, lang_order, mode="cll"):
    rand_perf = {}
    mono_perf = {}
    if lang_order == "high2lowlang":
        lang_list = ["en", "de", "fr", "hi", "es", "th"]
    elif lang_order == "low2highlang":
        lang_list = ["th", "es", "hi", "fr", "de", "en"]
    else:
        lang_list = lang_order.split("_")

    if mode == "cll":
        test_lang_list = lang_list
    else:
        test_lang_list = ["de", "en", "es", "fr", "hi", "th"]

    seeds = ["42"]#, "40", "35"]

    intents_seeds = [[[0.0 for _ in range(len(seeds))] for _ in range(6)] for _ in range(6)]
    slots_seeds = [[[0.0 for _ in range(len(seeds))] for _ in range(6)] for _ in range(6)]
    all_metrics_seeds = [[[0.0 for _ in range(len(seeds))] for _ in range(12)] for _ in range(6)]

    for k, seed in enumerate(seeds):
        results_dir = root_dir+mode+"/BertBaseMultilingualCased/"+lang_order+"/high2lowclass/"\
                      + model + "SEED_"+seed+"/metrics/"

        results_out = results_dir + "csv"

        if not os.path.isdir(results_out):
            os.mkdir(results_out)

        intents = [[0.0 for _ in range(6)] for _ in range(6)]
        slots = [[0.0 for _ in range(6)] for _ in range(6)]
        all_metrics = [[0.0 for _ in range(12)] for _ in range(6)]
        flag_file_not_found = False
        accumulated_langs = []
        for i, train_lang in enumerate(lang_list):
            accumulated_langs.append(train_lang)
            path = results_dir+"final_metrics_" + str(i) + ".pickle"
            if os.path.isfile(path):
                with open(path, "rb") as file:
                    data = pickle.load(file)

                for j, test_lang in enumerate(lang_list):
                    if mode == "cll":
                        eff_train_lang = train_lang
                    else:
                        eff_train_lang = "-".join(accumulated_langs)

                    intent_alias = [k for k, v in data[test_lang].items() if "intent_acc" in k][0]
                    slot_alias = [k for k, v in data[test_lang].items() if "slot_f1" in k][0]

                    intents[i][j] = data[test_lang][intent_alias] * 100
                    slots[i][j] = data[test_lang][slot_alias] * 100
                    all_metrics[i][2*j] = intents[i][j]
                    all_metrics[i][2*j+1] = slots[i][j]

                intents_np = np.array(intents)
                slots_np = np.array(slots)
                all_metrics_np = np.asarray(all_metrics)

            else:
                print(">>>>>>File ", path, " not found!")
                flag_file_not_found = True
                all_metrics_np = None

        results_array = []

        if not flag_file_not_found:

            acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_mono_all, forget_perf_all, final_perf_all = \
                acc_avg(all_metrics_np, lang_list, lang_list), \
                bwt_avg(all_metrics_np, lang_list, lang_list), \
                fwt_avg(all_metrics_np, lang_list, rand_perf), \
                fwt_avg_mono(all_metrics_np, lang_list, mono_perf), \
                forget_avg(all_metrics_np, lang_list, lang_list), \
                final_perf(all_metrics_np, lang_list)

            # print(" acc_avg: ", acc_avg_all[2], acc_avg_all[3], np.mean([acc_avg_all[2], acc_avg_all[3]]),
            #       " bwt_avg: ", bwt_avg_all[2], bwt_avg_all[3], np.mean([bwt_avg_all[2], bwt_avg_all[3]]),
            #       "forget_perf:", forget_perf_all[2], forget_perf_all[3], np.mean([forget_perf_all[2], forget_perf_all[3]]),
            #       " fwt_avg:", fwt_avg_all[2], fwt_avg_all[3], np.mean([fwt_avg_all[2], fwt_avg_all[3]]),
            #       " final_perf:", final_perf_all[0], final_perf_all[1], np.mean([final_perf_all[0], final_perf_all[1]]))

            # print(acc_avg_all[2], ",",  acc_avg_all[3], ",",  np.mean([acc_avg_all[2], acc_avg_all[3]]),
            #       ",", bwt_avg_all[2], ",",  bwt_avg_all[3], ",",  np.mean([bwt_avg_all[2], bwt_avg_all[3]]),
            #       ",", forget_perf_all[2], ",",  forget_perf_all[3], ",",  np.mean([forget_perf_all[2], forget_perf_all[3]]),
            #       ",", fwt_avg_all[2], ",",  fwt_avg_all[3], ",",  np.mean([fwt_avg_all[2], fwt_avg_all[3]]),
            #       ",", fwt_avg_mono_all[2], ",",  fwt_avg_mono_all[3], ",",  np.mean([fwt_avg_mono_all[2], fwt_avg_mono_all[3]]),
            #       ",", final_perf_all[0], ",",  final_perf_all[1], ",",  np.mean([final_perf_all[0], final_perf_all[1]]))

            # pd.DataFrame(intents_np).to_csv(os.path.join(results_out, "results_intents_seed_"+str(seed)+".csv"))
            # pd.DataFrame(slots_np).to_csv(os.path.join(results_out, "results_slots_seed_"+str(seed)+".csv"))
            # pd.DataFrame(all_metrics_np).to_csv(os.path.join(results_out, "results_matrix_seed_"+str(seed)+".csv"))

            results_array = [acc_avg_all[2], acc_avg_all[3], np.mean([acc_avg_all[2], acc_avg_all[3]]),
                             bwt_avg_all[2], bwt_avg_all[3], np.mean([bwt_avg_all[2], bwt_avg_all[3]]),
                             forget_perf_all[2], forget_perf_all[3], np.mean([forget_perf_all[2], forget_perf_all[3]]),
                             fwt_avg_all[2], fwt_avg_all[3], np.mean([fwt_avg_all[2], fwt_avg_all[3]]),
                             fwt_avg_mono_all[2], fwt_avg_mono_all[3], np.mean([fwt_avg_mono_all[2], fwt_avg_mono_all[3]]),
                             final_perf_all[0], final_perf_all[1], np.mean([final_perf_all[0], final_perf_all[1]])]

            # results_array = [fwt_avg_mono_all[2], fwt_avg_mono_all[3], np.mean([fwt_avg_mono_all[2], fwt_avg_mono_all[3]])]

    # if len(seeds) > 1:
    #     ## ALL SEEDS
    #     intents_np_seeds = np.mean(np.array(intents_seeds), axis=-1)
    #     slots_np_seeds = np.mean(np.array(slots_seeds), axis=-1)
    #     all_metrics_np_seeds = np.mean(np.asarray(all_metrics_seeds), axis=-1)
    #
    #     pd.DataFrame(intents_np_seeds).to_csv(os.path.join(results_out, "results_intents_all_seeds.csv"))
    #     pd.DataFrame(slots_np_seeds).to_csv(os.path.join(results_out, "results_slots_all_seeds.csv"))
    #     pd.DataFrame(all_metrics_np_seeds).to_csv(os.path.join(results_out, "results_matrix_all_seeds.csv"))

    print(model, lang_order, " results_array:", results_array)
    return results_array



def cll_lang_all(model, lang_order, mode="cll"):
    rand_perf = {}

    mono_perf = {}
    if lang_order == "high2lowlang":
        lang_list = ["en", "de", "fr", "hi", "es", "th"]
    elif lang_order == "low2highlang":
        lang_list = ["th", "es", "hi", "fr", "de", "en"]
    else:
        lang_list = lang_order.split("_")

    if mode == "cll":
        test_lang_list = lang_list
    else:
        test_lang_list = ["de", "en", "es", "fr", "hi", "th"]

    seed = "42"#, "40", "35"]

    # results_dir = root_dir+mode+"/BertBaseMultilingualCased/"+lang_order+"/high2lowclass/"\
    #               + model + "SEED_"+seed+"/metrics/"
    if model == "single_head":
        if lang_order == "high2lowlang":
            results_dir = "/home1/mmhamdi/Results/x-continuous-learn/cll/NLU/BertBaseMultilingualCased/"\
                      +lang_order+"/high2lowclass/vanilla/single_head/tune_all_trans/tune_all_linear/SEED_42/metrics/"
        else:
            results_dir = "/home1/mmhamdi/Results/x-continuous-learn/cll/NLU/BertBaseMultilingualCased/" \
                          +lang_order+"/high2lowclass/vanilla/SEED_42/metrics/"

    elif model == "multi_head_in":
        if lang_order != "high2lowlang":
            results_dir = "/home1/mmhamdi/Results/x-continuous-learn/cll/NLU/BertBaseMultilingualCased/"+lang_order\
                          +"/high2lowclass/multi_head_in/SEED_42/metrics/"
        else:
            results_dir = "/home1/mmhamdi/Results/x-continuous-learn/cll/NLU/BertBaseMultilingualCased/"+lang_order \
                          +"/high2lowclass/vanilla/multi_head_in/all/SEED_42/metrics/"

    elif model == "multi_head_out":
        if lang_order != "high2lowlang":
            results_dir = "/home1/mmhamdi/Results/x-continuous-learn/cll/NLU/BertBaseMultilingualCased/"\
                          +lang_order+"/high2lowclass/multi_head_out/SEED_42/metrics/"
        else:
            results_dir = "/home1/mmhamdi/Results/x-continuous-learn/cll/NLU/BertBaseMultilingualCased/" \
                          +lang_order+"/high2lowclass/vanilla/multi_head_out/SEED_42/metrics/"

    elif model == "multi-incr-cll":
        if lang_order == "high2lowlang":
            results_dir = "/project/jonmay_231/meryem/Results/x-continuous-learn/multi-incr-cll/NLU/" \
                          + "BertBaseMultilingualCased/high2lowlang/high2lowclass/SEED_42/metrics/"
        else:
            results_dir = "/home1/mmhamdi/Results/x-continuous-learn/multi-incr-cll/NLU/BertBaseMultilingualCased/" \
                          + lang_order+"/high2lowclass/SEED_42/metrics/"
    elif model == "ewc_online":
        results_dir = "/home1/mmhamdi/Results/x-continuous-learn/cll/NLU/BertBaseMultilingualCased/"+lang_order+ \
                      "/high2lowclass/ewc_1.0_use-online_gamma-0.01/SEED_42/metrics/"

    elif model == "multi_head_embed":
        if lang_order == "high2lowlang":
            results_dir = "/home1/mmhamdi/Results/x-continuous-learn/cll/NLU/BertBaseMultilingualCased/"+lang_order+\
                          "/high2lowclass/vanilla/multi_head_in/embed/SEED_42/metrics/"
        else:
            results_dir = "/home1/mmhamdi/Results/x-continuous-learn/cll/NLU/BertBaseMultilingualCased/"+lang_order+ \
                          "/high2lowclass/multi_head_in_embeddings/SEED_42/metrics"

    results_out = results_dir + "csv"

    if not os.path.isdir(results_out):
        os.mkdir(results_out)

    intents = [[0.0 for _ in range(6)] for _ in range(6)]
    slots = [[0.0 for _ in range(6)] for _ in range(6)]
    all_metrics = [[0.0 for _ in range(12)] for _ in range(6)]
    flag_file_not_found = False
    accumulated_langs = []
    for i, train_lang in enumerate(lang_list):
        accumulated_langs.append(train_lang)
        path = results_dir+"final_metrics_" + str(i) + ".pickle"
        if os.path.isfile(path):
            with open(path, "rb") as file:
                data = pickle.load(file)

            for j, test_lang in enumerate(lang_list):
                if mode == "cll":
                    eff_train_lang = train_lang
                else:
                    eff_train_lang = "-".join(accumulated_langs)
                # if "adapters" in model:
                #     intent_alias = eff_train_lang+"_"+str(i) + "_test_intent_acc_"+test_lang + "_" \
                #                    + str(test_lang_list.index(test_lang))
                #     slot_alias = eff_train_lang+"_" + str(i) + "_test_slot_f1_"+test_lang + "_" \
                #                  + str(test_lang_list.index(test_lang))
                # else:
                #     intent_alias = eff_train_lang+"_"+str(test_lang_list.index(test_lang))\
                #                    + "_test_intent_acc_0_"+test_lang
                #     slot_alias = eff_train_lang+"_"+str(test_lang_list.index(test_lang))\
                #                  + "_test_slot_f1_0_"+test_lang

                intent_alias = [k for k, v in data[test_lang].items() if "intent_acc" in k][0]
                slot_alias = [k for k, v in data[test_lang].items() if "slot_f1" in k][0]

                intents[i][j] = data[test_lang][intent_alias] * 100
                slots[i][j] = data[test_lang][slot_alias] * 100
                all_metrics[i][2*j] = intents[i][j]
                all_metrics[i][2*j+1] = slots[i][j]

            intents_np = np.array(intents)
            slots_np = np.array(slots)
            all_metrics_np = np.asarray(all_metrics)

        else:
            print(">>>>>>File ", path, " not found!")
            flag_file_not_found = True
            all_metrics_np = None

    if not flag_file_not_found:
        # print("all_metrics_np: ", all_metrics_np)

        # summary_metrics = [0.0 for _ in range(15)]

        fwt_avg_mono_all, forget_perf_all = \
            fwt_avg_mono(all_metrics_np, lang_list, mono_perf), \
            forget_avg(all_metrics_np, lang_list, lang_list)

        fwt_array = fwt_avg_mono_all[0], fwt_avg_mono_all[1]
        forget_array = forget_perf_all[0], forget_perf_all[1]

    return fwt_array, forget_array


def cil_rand_init():
    subtask_list = ["subtask_"+str(i) for i in range(0, 117, 10)]
    langs = ["en", "de", "fr", "es", "hi", "th"]

    test_subtask_list = subtask_list
    metrics = {lang: {subtask: [0.0, 0.0] for subtask in subtask_list} for lang in langs}

    for lang in langs:
        results_dir = root_dir + "cil/NLU/BertBaseMultilingualCased/"+lang+"/random_init/metrics/"
        if not os.path.isdir(results_dir):
            print("Experiment not yet done:", results_dir)

            return []
        else:

            intents = [0.0 for _ in range(len(subtask_list))]
            slots = [0.0 for _ in range(len(subtask_list))]
            all_metrics = [0.0 for _ in range(2*len(subtask_list))]
            path = results_dir+"initial_metrics.pickle"
            if os.path.isfile(path):
                with open(path, "rb") as file:
                    data = pickle.load(file)

                for i, test_subtask in enumerate(subtask_list):
                    if len(data[test_subtask]) > 0:
                        intent_alias = lang+"_0_init_intent_acc_"+test_subtask + "_" \
                                       + str(test_subtask_list.index(test_subtask))
                        slot_alias = lang+"_0_init_slot_f1_"+test_subtask + "_" \
                                    + str(test_subtask_list.index(test_subtask))

                        metrics[lang][test_subtask][0] = data[test_subtask][intent_alias] * 100
                        metrics[lang][test_subtask][1] = data[test_subtask][slot_alias] * 100

                        # intents[i] = data[test_subtask][intent_alias] * 100
                        # slots[i] = data[test_subtask][slot_alias] * 100
                        # all_metrics[2*i] = intents[i]
                        # all_metrics[2*i+1] = slots[i]

            else:
                print(">>>>>>File ", path, " NOT FOUND !!!")
    print(metrics)


def cil_mono():
    subtask_list = ["subtask_"+str(i) for i in range(0, 117, 10)]
    langs = ["en", "de", "fr", "es", "hi", "th"]

    metrics = {lang: {subtask: [0.0, 0.0] for subtask in subtask_list} for lang in langs}

    for lang in langs:
        for i, train_subtask in enumerate(subtask_list):
            original_subtask = train_subtask
            if lang == "de" and i == 0:
                train_subtask += "0"
            results_dir = root_dir + "cil/NLU/BertBaseMultilingualCased/MONO/"+lang+"/"+str(i)+"/metrics/"
            if not os.path.isdir(results_dir):
                print("Experiment not yet done:", results_dir)
                return []
            else:
                path = results_dir+"final_metrics_0.pickle"
                if os.path.isfile(path):
                    with open(path, "rb") as file:
                        data = pickle.load(file)
                    print("--------------------------------------------")
                    print("lang:", lang, " i:", i, " train_subtask:", train_subtask, " data:", data)

                    if len(data[train_subtask]) > 0:
                        intent_alias = lang+"_0_test_intent_acc_"+train_subtask+"_0"
                        slot_alias = lang+"_0_test_slot_f1_"+train_subtask+"_0"

                        metrics[lang][original_subtask][0] = data[train_subtask][intent_alias] * 100
                        metrics[lang][original_subtask][1] = data[train_subtask][slot_alias] * 100

                else:
                    print(">>>>>>File ", path, " NOT FOUND !!!")

    print(metrics)


def cil_multi_purpose(result_path, lang):
    rand_perf_all = {}

    mono_perf_all = {}

    rand_perf = rand_perf_all[lang]
    mono_perf = mono_perf_all[lang]

    seed = 42

    subtask_list = ["subtask_"+str(i) for i in range(0, 117, 10)]

    test_subtask_list = subtask_list

    # seeds = ["42", "40", "35"]
    #
    # intents_seeds = [[[0.0 for _ in range(len(seeds))] for _ in range(len(subtask_list))]
    #                  for _ in range(len(subtask_list))]
    #
    # slots_seeds = [[[0.0 for _ in range(len(seeds))] for _ in range(len(subtask_list))]
    #                for _ in range(len(subtask_list))]
    #
    # all_metrics_seeds = [[[0.0 for _ in range(len(seeds))] for _ in range(len(subtask_list))]
    #                      for _ in range(len(subtask_list))]

    if not os.path.isdir(result_path):
        print("Experiment not yet done:", result_path)
        return []

    results_out = result_path + "csv"

    if not os.path.isdir(results_out):
        os.mkdir(results_out)

    intents = [[0.0 for _ in range(len(subtask_list))] for _ in range(len(subtask_list))]
    slots = [[0.0 for _ in range(len(subtask_list))] for _ in range(len(subtask_list))]
    all_metrics = [[0.0 for _ in range(2*len(subtask_list))] for _ in range(len(subtask_list))]
    flag_file_not_found = False
    accumulated_langs = []
    for i, train_subtask in enumerate(subtask_list):
        accumulated_langs.append(train_subtask)
        path = result_path+"final_metrics_" + str(i) + ".pickle"
        if os.path.isfile(path):
            with open(path, "rb") as file:
                data = pickle.load(file)

            for j, test_subtask in enumerate(test_subtask_list):
                if len(data[test_subtask]) > 0:
                    intent_alias = lang+"_"+str(i) + "_test_intent_acc_"+test_subtask + "_" \
                                   + str(test_subtask_list.index(test_subtask))
                    slot_alias = lang+"_" + str(i) + "_test_slot_f1_"+test_subtask + "_" \
                                 + str(test_subtask_list.index(test_subtask))

                    intents[i][j] = data[test_subtask][intent_alias] * 100
                    slots[i][j] = data[test_subtask][slot_alias] * 100
                    all_metrics[i][2*j] = intents[i][j]
                    all_metrics[i][2*j+1] = slots[i][j]

            intents_np = np.array(intents)
            slots_np = np.array(slots)
            all_metrics_np = np.asarray(all_metrics)

        else:
            print(">>>>>>File ", path, " NOT FOUND !!!")
            # flag_file_not_found = True
            # all_metrics_np = None

    results_array = []

    if True:#not flag_file_not_found:
        acc_avg_all, bwt_avg_all, forget_perf_all, fwt_avg_all, fwt_avg_mono_all, final_perf_all = \
            acc_avg(all_metrics_np, subtask_list, subtask_list), \
            bwt_avg(all_metrics_np, subtask_list, subtask_list), \
            forget_avg(all_metrics_np, subtask_list, subtask_list), \
            fwt_avg(all_metrics_np, subtask_list, rand_perf), \
            fwt_avg_mono(all_metrics_np, subtask_list, mono_perf), \
            final_perf(all_metrics_np, subtask_list)

        pd.DataFrame(intents_np).to_csv(os.path.join(results_out, "results_intents_seed_"+str(seed)+".csv"))
        pd.DataFrame(slots_np).to_csv(os.path.join(results_out, "results_slots_seed_"+str(seed)+".csv"))
        pd.DataFrame(all_metrics_np).to_csv(os.path.join(results_out, "results_matrix_seed_"+str(seed)+".csv"))

        results_array = [acc_avg_all[2], acc_avg_all[3], np.mean([acc_avg_all[2], acc_avg_all[3]]),
                         bwt_avg_all[2], bwt_avg_all[3], np.mean([bwt_avg_all[2], bwt_avg_all[3]]),
                         forget_perf_all[2], forget_perf_all[3], np.mean([forget_perf_all[2], forget_perf_all[3]]),
                         fwt_avg_all[2], fwt_avg_all[3], np.mean([fwt_avg_all[2], fwt_avg_all[3]]),
                         fwt_avg_mono_all[2], fwt_avg_mono_all[3], np.mean([fwt_avg_mono_all[2], fwt_avg_mono_all[3]]),
                         final_perf_all[0], final_perf_all[1], np.mean([final_perf_all[0], final_perf_all[1]])]

    return results_array


def fwt_bwt_per_lang(model, lang_orders):
    fwt_array_all = {"en": ([], []), "de": ([], []), "fr": ([], []), "hi": ([], []),
                     "es": ([], []), "th": ([], [])}

    forget_array_all = {"en": ([], []), "de": ([], []), "fr": ([], []), "hi": ([], []),
                        "es": ([], []), "th": ([], [])}

    for lang_order in lang_orders:
        fwt_array, forget_array = cll_lang_all(model, lang_order)
        fwt_intent_array, fwt_slot_array = fwt_array
        forget_intent_array, forget_slot_array = forget_array

        for lang in fwt_intent_array:
            fwt_array_all[lang][0].append(fwt_intent_array[lang])
            fwt_array_all[lang][1].append(fwt_slot_array[lang])
            forget_array_all[lang][0].append(forget_intent_array[lang])
            forget_array_all[lang][1].append(forget_slot_array[lang])

    values = []
    for k, v in fwt_array_all.items():
        values.append(str(round(np.mean(v[0]), 2)))
        values.append(str(round(np.mean(v[1]), 2)))

    print(" & ".join(values))

    values = []
    for k, v in forget_array_all.items():
        values.append(str(round(np.mean(v[0]), 2)))
        values.append(str(round(np.mean(v[1]), 2)))
    print(" & ".join(values))


if __name__ == "__main__":

    lang_orders = ["high2lowlang", "low2highlang", "en_th_de_hi_es_fr", "hi_de_es_th_fr_en"]

    for lang_order in lang_orders:
        # cll_multi_purpose("single_head", lang_order)
        # cll_multi_purpose("multi_head_inall", lang_order)
        # cll_multi_purpose("multi_head_out", lang_order)
        cll_multi_purpose("er_0.10", lang_order)
        cll_multi_purpose("er_0.30", lang_order)
        cll_multi_purpose("er_0.50", lang_order)
        cll_multi_purpose("er_0.70", lang_order)
        cll_multi_purpose("kd-logits", lang_order)
        cll_multi_purpose("kd-rep", lang_order)
        cll_multi_purpose("mbpa_near_n", lang_order)
        cll_multi_purpose("mbpa_reptile", lang_order)
        cll_multi_purpose("mbpa_random", lang_order)





