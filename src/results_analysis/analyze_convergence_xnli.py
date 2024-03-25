import pickle, random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from summarize_metrics import acc_avg, fwt_avg, fwt_avg_mono, bwt_avg, forget_avg, final_perf
import matplotlib.pyplot as plt

def load_multi_purpose(out_dir, model):
    with open(out_dir+model+".pickle", "rb") as file:
        data_pickle = pickle.load(file)

    return data_pickle[model + "_perf"], data_pickle[model + "_conf"]

def compute_bootstrap(root_dir, lang_order, model_name, epoch_num):
    all_metrics = [0.0 for _ in range(6)]
    lang_list = lang_order.split("_")[:1]

    # Approach 1 : Compute the metrics for all
    for i_train, train_lang in enumerate(lang_list):
        file_name = root_dir+"Dev_perf-Epoch_"+str(epoch_num)+"-train_"+str(i_train)

        with open(file_name, "r") as file:
            results = file.read().splitlines()

        intent_corrects = 0

        intents_true = []
        intents_pred = []

        for i, sent in enumerate(results):
            parts = sent.split("\t")
            assert len(parts) == 3
            sent_text, intent_true, intent_pred = parts

            intents_true.append(intent_true)
            intents_pred.append(intent_pred)

            #####

            intent_corrects += int(intent_pred == intent_true)

        intent_acc = float(intent_corrects)*100 / len(results)

        all_metrics[i_train] = intent_acc

    all_metrics_np = np.asarray(all_metrics)
    return all_metrics_np

if __name__ == "__main__":
    #### CONSTANTS
    out_dir = "metrics/"
    lang_order = "en_zh_vi_ar_tr_bg_el_ur"
    plot_save_dir = "/project/jonmay_231/meryem/SpacedRepetitionFigures/XNLI/multi"

    languages = lang_order.split("_")

    root_dirs = {"vanilla": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/BASELINE/cll-er_kd/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                #  "joint_plus": "/project/jonmay_231/meryem/OtherReproduction_Results_Debug/x-continuous-learn/MTOP/BASELINE/multi-incr-cll/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                "erltn_demotefirst_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/ltn/fifo/DemoteFirstDeck/MultiLeitnerQueue/er/cll-er_kd/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans-class/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                #  "rand_ltn_demotefirst_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/rand/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                 "er": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/BASELINE/cll-er_kd/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans-class/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                "ltn_demotefirst_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/ltn/fifo/DemoteFirstDeck/MultiLeitnerQueue/main/cll-er_kd/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/"}
                #  "ltn_demotefirst_contqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemoteFirstDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                #  "ltn_demotefirst_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemoteFirstDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                #  "ltn_demoteprevious_contqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemotePreviousDeck/ContLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                #  "ltn_demoteprevious_multiqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/DemotePreviousDeck/MultiLeitnerQueue/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/vanilla/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/",
                #  "erltn_demoteprevious_contqueue": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/MTOP/ltn/fifo/DemotePreviousDeck/ContLeitnerQueue/er/cll-er_kd/NLU/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/"+lang_order+"/high2lowclass/er_memsz-6000_type-reservoir_sample-random_k-16/trans-intent-slot/single_head/no_adapters/tune_all_trans/tune_all_linear/SEED_42/"}
    
    multi_root_dirs = {
        "ltn_demotefirst": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/ltn/fifo/DemoteFirstDeck/MultiLeitnerQueue/main/multi/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/tune_all_trans/tune_all_linear/SEED_42/",
        "ltn_demoteprevious": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/ltn/fifo/DemotePreviousDeck/MultiLeitnerQueue/main/multi/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/tune_all_trans/tune_all_linear/SEED_42/",
        "rand_demotefirst": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/ltn/rand/DemoteFirstDeck/MultiLeitnerQueue/main/multi/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/tune_all_trans/tune_all_linear/SEED_42/",
        "rand_demoteprevious": "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/ltn/rand/DemotePreviousDeck/MultiLeitnerQueue/main/multi/BertBaseMultilingualCased/lr-3e-05_eps-1e-08_beta-0.9-0.99/tune_all_trans/tune_all_linear/SEED_42/"
    }
    intents_results_dict = {}
    for model_name, root_dir in multi_root_dirs.items():
        print(model_name)
        for epoch_num in range(10):
            all_metrics_np = compute_bootstrap(root_dir, lang_order, model_name, epoch_num)
            if model_name not in intents_results_dict:
                intents_results_dict.update({model_name: []})
            intents_results_dict[model_name].append(all_metrics_np[0])

    print("intents_results_dict:", intents_results_dict)

    for model_name, values in intents_results_dict.items():
        intents_results_dict.update({model_name: np.array(values).T.tolist()})

    for i, lang in enumerate(languages[:1]):
        #create dataset
        # df = pd.DataFrame({model_name: values[i] for model_name, values in intents_results_dict.items()})
        df = pd.DataFrame(intents_results_dict)

        #plot individual lines
        plt.clf()
        for model_name in intents_results_dict:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy Score")
        # plt.title('Dev on '+languages[0]+' after training on '+lang + ' in language_order: '+lang_order)
        plt.title('Dev after training on '+lang + ' in language_order: '+lang_order)
        plt.legend(loc="lower right")

        #display plot
        plt.savefig(plot_save_dir+'classscore_'+languages[0]+'_aftertrain-'+lang+'_dev-perf-epoch-curves.png')