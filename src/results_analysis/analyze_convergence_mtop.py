import pickle, random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from summarize_metrics import acc_avg, fwt_avg, fwt_avg_mono, bwt_avg, forget_avg, final_perf
import matplotlib.pyplot as plt

eval_type = "Test" # "Dev"
def load_multi_purpose(out_dir, model):
    with open(out_dir+model+".pickle", "rb") as file:
        data_pickle = pickle.load(file)

    return data_pickle[model + "_perf"], data_pickle[model + "_conf"]

def compute_bootstrap(root_dir, train_lang, test_lang, epoch_num):
    all_metrics = [[0.0 for _ in range(6)] for _ in range(2)]
    lang_list = lang_order.split("_")[:1]

    # Approach 1 : Compute the metrics for all
    for i_train, train_lang in enumerate(lang_list):
        file_name = root_dir+eval_type+"_perf-Epoch_"+str(epoch_num)+"-train_"+str(i_train) # Dev
        # file_name = root_dir+eval_type+"_perf-Epoch_"+str(epoch_num)+"-train_"+train_lang+"-test_"+test_lang

        with open(file_name, "r") as file:
            results = file.read().splitlines()

        intent_corrects = 0

        intents_true = []
        intents_pred = []

        slots_true = []
        slots_pred = []

        for i, sent in enumerate(results):
            parts = sent.split("\t")
            assert len(parts) == 5
            sent_text, intent_true, intent_pred, slot_true, slot_pred = parts

            intents_true.append(intent_true)
            intents_pred.append(intent_pred)

            #####
            slot_true_list = slot_true.split(" ")
            slot_pred_list = slot_pred.split(" ")

            if len(slot_true_list) != len(slot_pred_list):
                print("len(slot_true): ", len(slot_true), "len(slot_pred): ", len(slot_pred),
                        " slot_true:", slot_true, " slot_pred:", slot_pred)

            assert len(slot_true_list) == len(slot_pred_list)

            slots_true.extend(slot_true_list)
            slots_pred.extend(slot_pred_list)

            intent_corrects += int(intent_pred == intent_true)

        intent_acc = float(intent_corrects)*100 / len(results)
        slot_f1 = f1_score(slots_true, slots_pred, average="macro") * 100

        all_metrics[0][i_train] = intent_acc
        all_metrics[1][i_train] = slot_f1

    all_metrics_np = np.asarray(all_metrics)
    # acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_mono_all, forget_perf_all, final_perf_all = \
    #     acc_avg(all_metrics_np, lang_list, lang_list), \
    #     bwt_avg(all_metrics_np, lang_list, lang_list), \
    #     fwt_avg(all_metrics_np, lang_list, rand_perf), \
    #     fwt_avg_mono(all_metrics_np, lang_list, mono_perf), \
    #     forget_avg(all_metrics_np, lang_list, lang_list), \
    #     final_perf(all_metrics_np, lang_list)

    # return acc_avg_all, bwt_avg_all, fwt_avg_all, fwt_avg_mono_all, forget_perf_all, final_perf_all
    return all_metrics_np

def compute_Test_bootstrap(root_dir, train_lang, test_lang, epoch_num):    
    file_name = root_dir+eval_type+"_perf-Epoch_"+str(epoch_num)+"-train_"+train_lang+"-test_"+test_lang

    with open(file_name, "r") as file:
        results = file.read().splitlines()

    intent_corrects = 0

    intents_true = []
    intents_pred = []

    slots_true = []
    slots_pred = []

    for i, sent in enumerate(results):
        parts = sent.split("\t")
        assert len(parts) == 5
        sent_text, intent_true, intent_pred, slot_true, slot_pred = parts

        intents_true.append(intent_true)
        intents_pred.append(intent_pred)

        #####
        slot_true_list = slot_true.split(" ")
        slot_pred_list = slot_pred.split(" ")

        if len(slot_true_list) != len(slot_pred_list):
            print("len(slot_true): ", len(slot_true), "len(slot_pred): ", len(slot_pred),
                    " slot_true:", slot_true, " slot_pred:", slot_pred)

        assert len(slot_true_list) == len(slot_pred_list)

        slots_true.extend(slot_true_list)
        slots_pred.extend(slot_pred_list)

        intent_corrects += int(intent_pred == intent_true)

    intent_acc = float(intent_corrects)*100 / len(results)
    slot_f1 = f1_score(slots_true, slots_pred, average="macro") * 100

    return intent_acc, slot_f1

if __name__ == "__main__":
    #### CONSTANTS
    out_dir = "outputs/metrics/"
    plot_save_dir = "/home1/mmhamdi/x-continuous-learning/outputs/Plots/spacedrepetition/Test/MTOP/cont/"
    lang_order = "en_de_fr_hi_es_th"
    languages = lang_order.split("_")
    # lang_orders = ["en_de_fr_hi_es_th",
    #                "th_es_hi_fr_de_en",
    #                "fr_th_de_en_hi_es",
    #                "hi_en_es_th_fr_de",
    #                "es_hi_en_de_th_fr",
    #                "de_fr_th_es_en_hi"]

    #### RANDOM
    rand_perf, rand_conf = load_multi_purpose(out_dir, "random")
    print("--------------------------------------------------------------------------------------------")

    multi_perf, multi_conf = load_multi_purpose(out_dir, "multi")
    print("--------------------------------------------------------------------------------------------")

    mono_orig_perf, mono_orig_conf = load_multi_purpose(out_dir, "mono_orig")
    print("--------------------------------------------------------------------------------------------")

    # root_dirs = {#"vanilla": "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/BASELINE/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "joint_plus": "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/BASELINE/multi-incr-cll/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "er": "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/BASELINE/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans-intent-slot/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "rand_ltn_demotefirst_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/rand/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #              "rand_ltn_demoteprevious_contqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/rand/DemotePreviousDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",}
    #             #  "ltn_demotefirst_contqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemoteFirstDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "ltn_demotefirst_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "ltn_demoteprevious_contqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemotePreviousDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "ltn_demoteprevious_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemotePreviousDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "erltn_demoteprevious_contqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/fifo/DemotePreviousDeck/ContLeitnerQueue/er/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans-intent-slot/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/"}
    
    # multi_root_dirs = {"vanilla": "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/BASELINE/multi/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "joint_plus": "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/BASELINE/multi-incr-cll/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "er": "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/BASELINE/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans-intent-slot/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "rand_ltn_demotefirst_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/rand/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #              "ltn_demotefirst": "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/DemoteFirstDeck/multi/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/tune_all_trans/tune_all_linear/SEED_42/",
    #              "ltn_demoteprevious": "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/DemotePreviousDeck/multi/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/tune_all_trans/tune_all_linear/SEED_42/",
    #              "rand_demotefirst": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/rand/DemoteFirstDeck/MultiLeitnerQueue/multi/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/tune_all_trans/tune_all_linear/SEED_42/"}
    #             #  "ltn_demotefirst_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "ltn_demoteprevious_contqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemotePreviousDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "ltn_demoteprevious_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemotePreviousDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
    #             #  "erltn_demoteprevious_contqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/fifo/DemotePreviousDeck/ContLeitnerQueue/er/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans-intent-slot/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/"}
    
    root_dirs = {"vanilla": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/BASELINE/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                 "er": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/BASELINE/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans_model-gclassifier-slot_classifier/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                 "rand_ltn_demoteprevious_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/rand/DemotePreviousDeck/MultiLeitnerQueue/main/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                 "ltn_demoteprevious_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/fifo/DemotePreviousDeck/MultiLeitnerQueue/main/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/"}
                #  "ermainltn_demoteprevious_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/fifo/DemotePreviousDeck/MultiLeitnerQueue/main/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans_model-gclassifier-slot_classifier/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/"}#,
                #  "erltn_demoteprevious_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/fifo/DemotePreviousDeck/MultiLeitnerQueue/er/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans_model-gclassifier-slot_classifier/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/"}


    for train_lang in languages:
        for test_lang in languages:
            intents_results_dict = {}
            slots_results_dict = {}
            for model_name, root_dir in root_dirs.items():
                print(model_name)
                for epoch_num in range(10):
                    # all_metrics_np = compute_bootstrap(root_dir, mono_orig_perf, rand_perf, lang_order, model_name, epoch_num)
                    intent_acc, slot_f1 = compute_Test_bootstrap(root_dir, train_lang, test_lang, epoch_num)
                    if model_name not in intents_results_dict:
                        intents_results_dict.update({model_name: []})
                    intents_results_dict[model_name].append(intent_acc)

                    if model_name not in slots_results_dict:
                        slots_results_dict.update({model_name: []})
                    slots_results_dict[model_name].append(slot_f1)

                print("intents_results_dict:", intents_results_dict, " slots_results_dict:", slots_results_dict)

                for model_name, values in intents_results_dict.items():
                    intents_results_dict.update({model_name: np.array(values).T.tolist()})

                for model_name, values in slots_results_dict.items():
                    slots_results_dict.update({model_name: np.array(values).T.tolist()})

                #create dataset
                df = pd.DataFrame({model_name: values for model_name, values in intents_results_dict.items()})

                #plot individual lines
                plt.clf()
                for model_name in intents_results_dict:
                    plt.plot(df[model_name], label=model_name)

                plt.xlabel("# Epochs")
                plt.ylabel("Intent Accuracy Score")
                # plt.title(eval_type+' on '+languages[0]+' after training on '+lang + ' in language_order: '+lang_order)
                plt.title('Multi '+eval_type)
                plt.legend(loc="lower right")

                #display plot
                plt.savefig(plot_save_dir+lang_order+'/intenton_'+test_lang+'_aftertrain-'+train_lang+'_'+eval_type+'-perf-epoch-curves.png')
                # plt.savefig(plot_save_dir+lang_order+'/intent_'+eval_type+'-perf-epoch-curves.png')

                ### SLOTS

                #create dataset
                df = pd.DataFrame({model_name: values for model_name, values in slots_results_dict.items()})
                # df = pd.DataFrame(slots_results_dict)

                #plot individual lines
                plt.clf()
                for model_name in slots_results_dict:
                    plt.plot(df[model_name], label=model_name)

                plt.xlabel("# Epochs")
                plt.ylabel("Slot F1 Score")
                # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
                plt.title('Multi '+eval_type)
                plt.legend(loc="lower right")

                #display plot
                plt.savefig(plot_save_dir+lang_order+'/sloton_'+test_lang+'_aftertrain-'+train_lang+'_'+eval_type+'-perf-epoch-curves.png')
                # plt.savefig(plot_save_dir+lang_order+'/slot_'+eval_type+'-perf-epoch-curves.png')