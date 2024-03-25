from summarize_metrics import acc_avg, fwt_avg, fwt_avg_mono, bwt_avg, forget_avg, final_perf
import pickle, os
import numpy as np

# data_names = ["XNLI"]#, "XNLI"]
# demote_options = ["DemoteFirstDeck", "DemotePreviousDeck"]
# cont_options = ["ContLeitnerQueue", "MultiLeitnerQueue"]
# for data_name in data_names:
#     for demote_option in demote_options:
#         for cont_option in cont_options:
#             print("------------------------------------------------------")
#             print("data_name:", data_name, " demote_option:", demote_option, " cont_option:", cont_option)
#             # results_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/"+data_name+"/"+demote_option+"/"+cont_option+"/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/en_de_fr_hi_es_th/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
#             results_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/"+demote_option+"/"+cont_option+"/cll-er_kd/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/bg_de_el_en_es_fr_hi_ru_ur_tr_th_vi_zh_ar_sw/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
#             # lang_list = ["en", "de", "fr", "hi", "es", "th"]
#             # lang_list = ["en", "de", "es", "fr", "hi", "ja", "pt", "tr", "zh"]
#             lang_list = "bg_de_el_en_es_fr_hi_ru_ur_tr_th_vi_zh_ar_sw".split("_")

#             intents = [[0.0 for _ in range(6)] for _ in range(6)]
#             slots = [[0.0 for _ in range(6)] for _ in range(6)]
#             all_metrics = [[0.0 for _ in range(12)] for _ in range(6)]

#             flag = True
#             for i, train_lang in enumerate(lang_list):
#                 path = results_dir+"final_metrics_" + str(len(lang_list)-1) + ".pickle"
#                 if os.path.isfile(path):
#                     with open(path, "rb") as file:
#                         data = pickle.load(file)

#                     print("data:", data)

#                     for j, test_lang in enumerate(lang_list):
                        
#                         eff_train_lang = train_lang

#                         # print("data[test_lang]:", data[test_lang])

#                         intent_alias = [k for k, v in data[test_lang].items() if "class_acc" in k][0]
#                         # slot_alias = [k for k, v in data[test_lang].items() if "slot_f1" in k][0]

#                         intents[i][j] = data[test_lang][intent_alias] * 100
#                         slots[i][j] = data[test_lang][intent_alias] * 100
#                         # slots[i][j] = data[test_lang][slot_alias] * 100
#                         all_metrics[i][2*j] = intents[i][j]
#                         all_metrics[i][2*j+1] = slots[i][j]

#                     intents_np = np.array(intents)
#                     slots_np = np.array(slots)
#                     all_metrics_np = np.asarray(all_metrics)
#                 else:
#                     print("path:", path, " NOT FOUND")
#                     flag = False

#             if flag:
#                 F = forget_avg(all_metrics_np, lang_list, lang_list)
#                 print("F:", F[4], F[5])
#                 print("FP:", final_perf(all_metrics_np, lang_list))

def cont_perf():
    data_names = ["MTOP"]#, "XNLI"]
    demote_options = ["DemoteFirstDeck", "DemotePreviousDeck"]
    cont_options = ["ContLeitnerQueue", "MultiLeitnerQueue"]
    lang_orders = ["en_de_fr_hi_es_th", "es_hi_en_de_th_fr", "fr_th_de_en_hi_es", "hi_en_es_th_fr_de", "th_es_hi_fr_de_en"]#, "de_fr_th_es_en_hi"]
    for data_name in data_names:
        for demote_option in demote_options:
            for cont_option in cont_options:
                avg_f = [0.0, 0.0]
                avg_fp = [0.0, 0.0]
                for lang_order in lang_orders:
                    # print("------------------------------------------------------")
                    # print("data_name:", data_name, " demote_option:", demote_option, " cont_option:", cont_option)
                    results_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/"+data_name+"/"+demote_option+"/"+cont_option+"/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
                    lang_list = ["en", "de", "fr", "hi", "es", "th"]

                    intents = [[0.0 for _ in range(6)] for _ in range(6)]
                    slots = [[0.0 for _ in range(6)] for _ in range(6)]
                    all_metrics = [[0.0 for _ in range(12)] for _ in range(6)]

                    flag = True
                    for i, train_lang in enumerate(lang_list):
                        path = results_dir+"final_metrics_" + str(i) + ".pickle"
                        if os.path.isfile(path):
                            with open(path, "rb") as file:
                                data = pickle.load(file)

                            for j, test_lang in enumerate(lang_list):
                                
                                eff_train_lang = train_lang

                                # print("data[test_lang]:", data[test_lang])

                                intent_alias = [k for k, v in data[test_lang].items() if "class_acc" in k][0]
                                slot_alias = [k for k, v in data[test_lang].items() if "slot_f1" in k][0]

                                intents[i][j] = data[test_lang][intent_alias] * 100
                                slots[i][j] = data[test_lang][slot_alias] * 100
                                # slots[i][j] = data[test_lang][slot_alias] * 100
                                all_metrics[i][2*j] = intents[i][j]
                                all_metrics[i][2*j+1] = slots[i][j]

                            intents_np = np.array(intents)
                            slots_np = np.array(slots)
                            all_metrics_np = np.asarray(all_metrics)
                        else:
                            print("path:", path, " NOT FOUND")
                            flag = False

                    if flag:
                        # print("all_metrics_np:", all_metrics_np)
                        # exit(0)
                        F = forget_avg(all_metrics_np, lang_list, lang_list)
                        avg_f[0] += F[4]
                        avg_f[1] += F[5]
                        print("F:", F[4], F[5])

                        FP = final_perf(all_metrics_np, lang_list)
                        avg_fp[0] += FP[0]
                        avg_fp[1] += FP[1]

                        print("FP:", FP[0], FP[1])

                print(demote_option, cont_option, " F:", round(avg_f[0]/len(lang_orders), 2), round(avg_f[1]/len(lang_orders), 2), " FP:", round(avg_fp[0]/len(lang_orders), 2), round(avg_fp[1]/len(lang_orders), 2))

def cont_perf_path(results_dir):
    avg_f = [0.0, 0.0]
    avg_fp = [0.0, 0.0]
    avg_fwt = [0.0, 0.0]
    lang_orders = ["en_de_fr_hi_es_th"]#, "es_hi_en_de_th_fr", "fr_th_de_en_hi_es", "hi_en_es_th_fr_de", "th_es_hi_fr_de_en"]
    for lang_order in lang_orders:
        lang_list = lang_order.split("_")

        intents = [[0.0 for _ in range(6)] for _ in range(6)]
        slots = [[0.0 for _ in range(6)] for _ in range(6)]
        all_metrics = [[0.0 for _ in range(12)] for _ in range(6)]
        
        # results_dir =  "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/rand/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"

        # results_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/fifo/DemotePreviousDeck/ContLeitnerQueue/er/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans-intent-slot/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
        # results_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemotePreviousDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
        # results_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemotePreviousDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
        # results_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/rbf/fifo/DemoteFirstDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
        # results_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
        results_dir = "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/BASELINE/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"

        if "BASELINE" in results_dir:
            mono_perf = {"en": [95.3, 77.97], "de": [93.8, 66.29], "fr": [93.17, 70.6], "hi": [92.43, 62.54], "es": [94.0, 70.36], "th": [90.71, 64.08]}
        else:
            if "Previous" in results_dir:
                mono_perf = {"en": [94.19, 72.89], "de": [88.42, 59.7], "fr": [91.2, 65.43], "hi": [88.78, 58.08], "es": [90.76, 63.14], "th": [82.31, 57.13]}
            elif "First" in results_dir:
                mono_perf = {"en": [92.59, 72.07], "de": [87.94, 62.86], "fr": [88.1, 65.94], "hi": [85.73, 56.12], "es": [86.66, 61.24], "th": [83.8, 58.51]}
        #mono_perf = {"en": [95.3, 77.97], "de": [93.8, 66.29], "fr": [93.17, 70.6], "hi": [92.43, 62.54], "es": [94.0, 70.36], "th": [90.71, 64.08]}
        flag = True
        for i, train_lang in enumerate(lang_list):
            path = results_dir+"final_metrics_" + str(i) + ".pickle"
            if os.path.isfile(path):
                with open(path, "rb") as file:
                    data = pickle.load(file)

                for j, test_lang in enumerate(lang_list):
                    
                    eff_train_lang = train_lang

                    # print("data[test_lang]:", data[test_lang])

                    intent_alias = [k for k, v in data[test_lang].items() if "intent_acc" in k][0]
                    slot_alias = [k for k, v in data[test_lang].items() if "slot_f1" in k][0]

                    intents[i][j] = data[test_lang][intent_alias] * 100
                    slots[i][j] = data[test_lang][slot_alias] * 100
                    # slots[i][j] = data[test_lang][slot_alias] * 100
                    all_metrics[i][2*j] = intents[i][j]
                    all_metrics[i][2*j+1] = slots[i][j]

                intents_np = np.array(intents)
                slots_np = np.array(slots)
                all_metrics_np = np.asarray(all_metrics)
            else:
                print("path:", path, " NOT FOUND")
                flag = False

        if flag:
            #print("intents_np:", intents_np)
            # exit(0)
            F = forget_avg(all_metrics_np, lang_list, lang_list)
            avg_f[0] += F[4]
            avg_f[1] += F[5]
            #print(lang_order, " F:", F[4], F[5])

            FP = final_perf(all_metrics_np, lang_list)
            avg_fp[0] += FP[0]
            avg_fp[1] += FP[1]

            FWT = fwt_avg_mono(all_metrics_np, lang_list, mono_perf)
            avg_fwt[0] += FWT[2]
            avg_fwt[1] += FWT[3]
            print(lang_order, " F:", F[4], F[5], " FP:", FP[0], FP[1], " FWT:", FWT[2], FWT[3])

    print(avg_f[0]/len(lang_orders), avg_f[1]/len(lang_orders), avg_fp[0]/len(lang_orders), avg_fp[1]/len(lang_orders), avg_fwt[0]/len(lang_orders), avg_fwt[1]/len(lang_orders))



def multi_perf():
    avg_f = [0.0, 0.0]
    avg_fp = [0.0, 0.0]

    data_path = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/rand/DemoteFirstDeck/MultiLeitnerQueue/multi/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/tune_all_trans/tune_all_linear/SEED_42/metrics/final_metrics_0.pickle"
    lang_list = ["en", "de", "fr", "hi", "es", "th"]

    intents = [0.0 for _ in range(6)]
    slots = [0.0 for _ in range(6)]
    all_metrics = [0.0 for _ in range(12)]

    flag = True
    
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    for j, test_lang in enumerate(lang_list):

        # print("data[test_lang]:", data[test_lang])

        intent_alias = [k for k, v in data[test_lang].items() if "class_acc" in k][0]
        slot_alias = [k for k, v in data[test_lang].items() if "slot_f1" in k][0]

        intents[j] = data[test_lang][intent_alias] * 100
        slots[j] = data[test_lang][slot_alias] * 100
        # slots[i][j] = data[test_lang][slot_alias] * 100
        all_metrics[2*j] = intents[j]
        all_metrics[2*j+1] = slots[j]

    intents_np = np.array(intents)
    slots_np = np.array(slots)
    all_metrics_np = np.asarray(all_metrics)

    print("intents_np:", np.mean(intents_np))
    print("slots_np:", np.mean(slots_np))
    FP = final_perf(all_metrics_np, lang_list)
    avg_fp[0] += FP[0]
    avg_fp[1] += FP[1]

    print("FP:", FP[0], FP[1])

# multi_perf()

# path = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/rand/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/en_de_fr_hi_es_th/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
# path = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/rbf/fifo/DemoteFirstDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/en_de_fr_hi_es_th/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
# path = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemoteFirstDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/en_de_fr_hi_es_th/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
# path = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/en_de_fr_hi_es_th/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
#path = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/fifo/DemotePreviousDeck/ContLeitnerQueue/er/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/en_de_fr_hi_es_th/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans-intent-slot/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
path = "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/BASELINE/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/en_de_fr_hi_es_th/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/metrics/"
cont_perf_path(path)
