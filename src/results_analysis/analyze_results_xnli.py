import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import glob
from summarize_metrics import acc_avg, fwt_avg, fwt_avg_mono, bwt_avg, forget_avg, final_perf

def plot_mono():
    plot_save_dir = "/home1/mmhamdi/x-continuous-learning/outputs/Plots/spacedrepetition/Test/XNLI/mono/"

    root_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/HyperparamSearch/"

    results_baseline = [0.6506, 0.6460, 0.6592, 0.6558, 0.6381, 0.6428, 0.6538, 0.6432, 0.6486, 0.6209]

    # Leitner Queues number of decks 3
    with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-mono_en_ndecks-3_sample-epoch_update-batch/metrics.json") as file:
        results_ltn_deck_3 = json.load(file)

    # Leitner Queues number of decks 4
    with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-mono_en_ndecks-4_sample-epoch_update-batch/metrics.json") as file:
        results_ltn_deck_4 = json.load(file)

    # Leitner Queues number of decks 4
    with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-mono_en_ndecks-5_sample-epoch_update-epoch/metrics.json") as file:
        results_ltn_deck_5 = json.load(file)

    # Leitner Queues number of decks 6
    with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-mono_en_ndecks-6_sample-epoch_update-batch/metrics.json") as file:
        results_ltn_deck_6 = json.load(file)

    # RBF
    with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-rbf_fifo_schedtype-main_ltmode-mono_en_ndecks-5_sample-epoch_update-batch/metrics.json") as file:
        results_rbf = json.load(file)

    results_update_epoch = [0.6506, 0.6460, 0.2967, 0.6544, 0.3364, 0.4823, 0.5963, 0.6500, 0.6554, 0.3639]
    
    results_df = {"vanilla": results_baseline,
                  "ltn_deck_3": results_ltn_deck_3[0]["test"]["acc"]["en"],
                  "ltn_deck_4": results_ltn_deck_4[0]["test"]["acc"]["en"],
                  "ltn_deck_5": results_ltn_deck_5[0]["test"]["acc"]["en"],
                  "ltn_deck_6": results_ltn_deck_6[0]["test"]["acc"]["en"],
                  "rbf": results_rbf[0]["test"]["acc"]["en"],
                  "update_epoch": results_update_epoch}
    #create dataset
    df = pd.DataFrame(results_df)
    # df = pd.DataFrame(slots_results_dict)

    #plot individual lines
    plt.clf()
    for model_name in results_df:
        plt.plot(df[model_name], label=model_name)

    plt.xlabel("# Epochs")
    plt.ylabel("Test Performance on ")
    # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
    plt.title("Testing on EN")
    plt.legend(loc="lower right")
    plt.savefig(plot_save_dir+'en/accuracy_mono_en.png')

def plot_cont():
    plot_save_dir = "/home1/mmhamdi/x-continuous-learning/outputs/Plots/spacedrepetition/Test/XNLI/cont/"
    lang_order = "en_zh_vi_ar_tr_bg_el_ur"
    langs = lang_order.split("_")

    root_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/HyperparamSearch/"

    # Baseline
    with open(root_dir+"Baseline/ltmode-cont-mono_"+lang_order+"/metrics_without_er.json") as file:
        results_baseline = json.load(file)

    with open(root_dir+"Baseline/ltmode-cont-multi/metrics.json") as file:
        results_baseline_multi = json.load(file)

    # Leitner Queues Demote Previous
    with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-cont-multi_"+lang_order+"_ndecks-5_sample-epoch_update-batch/metrics.json") as file:
        results_ltn_demprev = json.load(file)

    # Leitner Queues Demote First
    with open(root_dir+"LtnScheduler/DemoteFirst/ltnmodel-ltn_fifo_schedtype-main_ltmode-cont-multi_"+lang_order+"_ndecks-5_sample-epoch_update-batch/metrics.json") as file:
        results_ltn_demfirst = json.load(file)

    # Rand
    with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_rand_schedtype-main_ltmode-cont-multi_"+lang_order+"_ndecks-5_sample-epoch_update-batch/metrics.json") as file:
        results_ltn_rand = json.load(file)
    

    for i, train_lang in enumerate(langs):
        for j, test_lang in enumerate(langs):
            results_df = {"vanilla": results_baseline_multi[i]["test"]["acc"][test_lang],
                          "rand": results_ltn_rand[i]["test"]["acc"][test_lang],
                          "ltn_dem_prev": results_ltn_demprev[i]["test"]["acc"][test_lang],
                          "ltn_dem_first": results_ltn_demfirst[i]["test"]["acc"][test_lang]}
            #create dataset
            df = pd.DataFrame(results_df)
            # df = pd.DataFrame(slots_results_dict)

            #plot individual lines
            plt.clf()
            for model_name in results_df:
                plt.plot(df[model_name], label=model_name)

            plt.xlabel("# Epochs")
            plt.ylabel("Test Performance on ")
            # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
            plt.title("Testing on "+ test_lang+" after training on "+train_lang)
            plt.legend(loc="lower right")
            plt.savefig(plot_save_dir+lang_order+'/accuracy_teston-'+test_lang+'_aftertrain-'+train_lang+'.png')

    # print(results_ltn_demprev)
    # print(results_ltn_demfirst)

    # root_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/HyperparamSearch/"
    # with open(root_dir+"Baseline/ltmode-cont-mono_en_zh_vi_ar_tr_bg_el_ur/metrics.json") as file:
    #     results = json.load(file)

    # print("BASELINE results:", max(results[0]["test"]["acc"]["en"]))

    # with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-rbf_fifo_schedtype-main_ltmode-mono_en_ndecks-5_sample-epoch_update-batch/metrics.json") as file:
    #     results = json.load(file)

    # print("RBF results:", max(results[0]["test"]["acc"]["en"]))


    # with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-mono_en_ndecks-3_sample-epoch_update-batch/metrics.json") as file:
    #     results = json.load(file)

    # print("LTN NUMBER OF DECKS 3 results:", max(results[0]["test"]["acc"]["en"]))

    # with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-mono_en_ndecks-4_sample-epoch_update-batch/metrics.json") as file:
    #     results = json.load(file)

    # print("LTN NUMBER OF DECKS 4 results:", max(results[0]["test"]["acc"]["en"]))

    # with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-mono_en_ndecks-6_sample-epoch_update-batch/metrics.json") as file:
    #     results = json.load(file)

    # print("LTN NUMBER OF DECKS 6 results:", max(results[0]["test"]["acc"]["en"]))

    # with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-mono_en_ndecks-5_sample-epoch_update-epoch/metrics.json") as file:
    #     results = json.load(file)

    # print("LTN NUMBER OF DECKS 5 UPDATE EPOCH results:", max(results[0]["test"]["acc"]["en"]))

    # ### 
    # with open(root_dir+"LtnScheduler/DemotePrevious/ltnmodel-ltn_fifo_schedtype-main_ltmode-cont-multi_en_zh_vi_ar_tr_bg_el_ur_ndecks-5_sample-epoch_update-batch/metrics.json") as file:
    #     results = json.load(file)

    # print("LTN en_zh_vi_ar_tr_bg_el_ur cont-multi results:", results[0]["test"]["acc"])

random_perf = {"xnli":
                    {"en": [0.3333998802156119, 0.0],
                     "vi": [0.3333998802156119, 0.0], 
                     "ar": [0.3333998802156119, 0.0],
                     "tr": [0.3333998802156119, 0.0],
                     "bg": [0.3333998802156119, 0.0], 
                     "el": [0.3333998802156119, 0.0], 
                     "ur": [0.3333998802156119, 0.0]},
               "mtop":
                    {},
               "tydiqa":
                    {}}

def plot_cont_new(task_name):
    lang_order = "en_vi_ar_tr_bg_el_ur"
    models = {"Vanilla":"Baseline/ltmode-cont-multi/"+lang_order+"/W_SCHEDULER_0_WARMUP/",
            #   "baseline_cont-mono":"Baseline/ltmode-cont-mono/"+lang_order+"/W_SCHEDULER_0_WARMUP/",
            #   "er_baseline_cont-mono": "Baseline/ER/ltmode-cont-mono/"+lang_order+"/W_SCHEDULER_0_WARMUP/",
              "er_baseline": "Baseline/ER/ltmode-cont-multi/"+lang_order+"/W_SCHEDULER_0_WARMUP/",
            #   "er_only_ltn_previous_cont-multi": "ER/LtnScheduler/DemotePrevious/ltnmodel-ltn/fifo/schedtype-er/ltmode-cont-multi/"+lang_order+"/ndecks-5/sample-epoch/update-batch/W_SCHEDULER_0_WARMUP/",
            #   "er_main_ltn_previous_cont-multi": "ER/LtnScheduler/DemotePrevious/ltnmodel-ltn/fifo/schedtype-main/ltmode-cont-multi/"+lang_order+"/ndecks-5/sample-epoch/update-batch/W_SCHEDULER_0_WARMUP/",
              "ltn_previous_cont-multi": "LtnScheduler/DemotePrevious/ltnmodel-ltn/fifo/schedtype-main/ltmode-cont-multi/"+lang_order+"/ndecks-5/sample-epoch/update-batch/W_SCHEDULER_0_WARMUP/",
              "ltn_previous_rand_cont-multi": "LtnScheduler/DemotePrevious/ltnmodel-ltn/rand/schedtype-main/ltmode-cont-multi/"+lang_order+"/ndecks-5/sample-epoch/update-batch/W_SCHEDULER_0_WARMUP/"}

    root_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/XNLI/HyperparamSearch/BertBaseMultilingualCased/"
    plot_save_dir = "/home1/mmhamdi/x-continuous-learning/outputs/Plots/spacedrepetition/Test/XNLI/cont_new/"

    langs = lang_order.split("_")
    results_json = {}
    all_metrics = {}

    avg_forgetting_epochs = {}
    avg_final_perf_epochs = {}
    avg_fwt_perf_epochs = {}
    avg_acc_perf_epochs = {}

    avg_forgetting_epochs_lang = {lang: {model_name: [] for model_name in models.keys()} for lang in langs}
    avg_final_perf_epochs_lang = {lang: {model_name: [] for model_name in models.keys()} for lang in langs}
    avg_acc_perf_epochs_lang = {lang: {model_name: [] for model_name in models.keys()} for lang in langs}
    avg_fwt_perf_epochs_lang = {lang: {model_name: [] for model_name in models.keys()} for lang in langs}
    
    for model_name, model_path in models.items():
        avg_forgetting_epochs.update({model_name: []})
        avg_final_perf_epochs.update({model_name: []})
        avg_fwt_perf_epochs.update({model_name: []})
        avg_acc_perf_epochs.update({model_name: []})

        for epoch in range(10):
            root_results_dir = root_dir+model_path+"metrics.json"

            all_metrics.update({model_name: [[0.0 for _ in range(14)] for _ in range(7)]})
            with open(root_results_dir) as file:
                results_json.update({model_name: json.load(file)})

            for i, train_lang in enumerate(langs):
                for j, test_lang in enumerate(langs):
                    all_metrics[model_name][i][2*j] = results_json[model_name][i]["test"]["acc"][test_lang][epoch]*100

            all_metrics_np = np.asarray(all_metrics[model_name])
            avg_forgetting_epochs[model_name].append(forget_avg(all_metrics_np, langs, langs)[4])
            avg_final_perf_epochs[model_name].append(final_perf(all_metrics_np, langs)[0])
            avg_fwt_perf_epochs[model_name].append(fwt_avg(all_metrics_np, langs, random_perf[task_name])[2])
            avg_acc_perf_epochs[model_name].append(acc_avg(all_metrics_np, langs, langs)[2])

            for k, lang in enumerate(langs):
                print(fwt_avg(all_metrics_np, langs, random_perf[task_name])[0])
                avg_forgetting_epochs_lang[lang][model_name].append(forget_avg(all_metrics_np, langs, langs)[0][lang])
                avg_final_perf_epochs_lang[lang][model_name].append(final_perf(all_metrics_np, langs)[2][k])
                avg_fwt_perf_epochs_lang[lang][model_name].append(fwt_avg(all_metrics_np, langs, random_perf[task_name])[0][lang])
                avg_acc_perf_epochs_lang[lang][model_name].append(acc_avg(all_metrics_np, langs, langs)[0][lang])

    print("avg_forgetting_epochs:", avg_forgetting_epochs)
    print("avg_final_perf_epochs:", avg_final_perf_epochs)
    print("avg_acc_perf_epochs:", avg_acc_perf_epochs)

    # Average forgetting
    df = pd.DataFrame(avg_forgetting_epochs)

    plt.clf()
    for model_name in avg_forgetting_epochs:
        plt.plot(df[model_name], label=model_name)

    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy")
    # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
    plt.title("Average Forgetting over epochs and test languages")
    plt.legend(loc="lower right")
    plt.savefig(plot_save_dir+lang_order+'/average_forgetting.png')

    # Average final performance
    df = pd.DataFrame(avg_final_perf_epochs)

    plt.clf()
    for model_name in avg_final_perf_epochs:
        plt.plot(df[model_name], label=model_name)

    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy")
    # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
    plt.title("Average Final Performance over epochs and test languages")
    plt.legend(loc="lower right")
    plt.savefig(plot_save_dir+lang_order+'/average_final_perf.png')

    # Average accuracy
    df = pd.DataFrame(avg_acc_perf_epochs)

    plt.clf()
    for model_name in avg_acc_perf_epochs:
        plt.plot(df[model_name], label=model_name)

    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy")
    # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
    plt.title("Average Accuracy over epochs and test languages")
    plt.legend(loc="lower right")
    plt.savefig(plot_save_dir+lang_order+'/average_acc.png')

    # Forward Transfer
    df = pd.DataFrame(avg_fwt_perf_epochs)

    plt.clf()
    for model_name in avg_fwt_perf_epochs:
        plt.plot(df[model_name], label=model_name)

    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy")
    # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
    plt.title("Average FWT over epochs and test languages")
    plt.legend(loc="lower right")
    plt.savefig(plot_save_dir+lang_order+'/average_fwt.png')

    for lang in langs:
        # Average forgetting
        df = pd.DataFrame(avg_forgetting_epochs_lang[lang])

        plt.clf()
        for model_name in avg_forgetting_epochs_lang[lang]:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
        plt.title("Average Forgetting over epochs and "+lang)
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+lang_order+'/average_forgetting_'+lang+'.png')

        # Average final performance
        df = pd.DataFrame(avg_final_perf_epochs_lang[lang])

        plt.clf()
        for model_name in avg_final_perf_epochs_lang[lang]:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
        plt.title("Average Final Performance over epochs and "+lang)
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+lang_order+'/average_final_perf_'+lang+'.png')

        # Average accuracy
        df = pd.DataFrame(avg_acc_perf_epochs_lang[lang])

        plt.clf()
        for model_name in avg_acc_perf_epochs_lang[lang]:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
        plt.title("Average Accuracy over epochs and "+lang)
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+lang_order+'/average_acc_'+lang+'.png')

        # FWT
        df = pd.DataFrame(avg_fwt_perf_epochs_lang[lang])

        plt.clf()
        for model_name in avg_fwt_perf_epochs_lang[lang]:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
        plt.title("Average FWT over epochs and "+lang)
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+lang_order+'/average_fwt_'+lang+'.png')

    ## Accuracy Per test and train language
    for i, train_lang in enumerate(langs):
        for j, test_lang in enumerate(langs):
            results_df = {model_name: model_json_val[i]["test"]["acc"][test_lang] for model_name, model_json_val in results_json.items()}
            # results_df = {"vanilla": results_baseline_multi[i]["test"]["acc"][test_lang],
            #               "rand": results_ltn_rand[i]["test"]["acc"][test_lang],
            #               "ltn_dem_prev": results_ltn_demprev[i]["test"]["acc"][test_lang],
            #               "ltn_dem_first": results_ltn_demfirst[i]["test"]["acc"][test_lang]}
            #create dataset
            df = pd.DataFrame(results_df)
            # df = pd.DataFrame(slots_results_dict)

            #plot individual lines
            plt.clf()
            for model_name in results_df:
                plt.plot(df[model_name], label=model_name)

            plt.xlabel("# Epochs")
            plt.ylabel("Accuracy")
            # plt.title('Test Performance on '+ test_lang+ ' on after training on '+train_lang + ' in language_order: '+lang_order)
            plt.title("Testing on "+ test_lang+" after training on "+train_lang)
            plt.legend(loc="lower right")
            plt.savefig(plot_save_dir+lang_order+'/accuracy_teston-'+test_lang+'_aftertrain-'+train_lang+'.png')

    ## DEV Accuracy Per train language
    plot_save_dir = "/home1/mmhamdi/x-continuous-learning/outputs/Plots/spacedrepetition/Dev/XNLI/cont_new/"
    for i, train_lang in enumerate(langs):
        results_df = {model_name: model_json_val[i]["val"]["acc"] for model_name, model_json_val in results_json.items()}
        # results_df = {"vanilla": results_baseline_multi[i]["test"]["acc"][test_lang],
        #               "rand": results_ltn_rand[i]["test"]["acc"][test_lang],
        #               "ltn_dem_prev": results_ltn_demprev[i]["test"]["acc"][test_lang],
        #               "ltn_dem_first": results_ltn_demfirst[i]["test"]["acc"][test_lang]}
        #create dataset
        df = pd.DataFrame(results_df)
        # df = pd.DataFrame(slots_results_dict)

        #plot individual lines
        plt.clf()
        for model_name in results_df:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        # plt.title('Test Performance on '+ test_lang+ ' on after training on '+train_lang + ' in language_order: '+lang_order)
        plt.title("Dev on "+ train_lang+" after training on "+train_lang)
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+lang_order+'/accuracy_teston-'+train_lang+'_aftertrain-'+train_lang+'.png')

     ## Train Accuracy Per train language
    print("***************************")
    plot_save_dir = "/home1/mmhamdi/x-continuous-learning/outputs/Plots/spacedrepetition/Train/XNLI/cont_new/"
    for i, train_lang in enumerate(langs):
        for model_name, model_json_val in results_json.items():
            print(train_lang, model_name, model_json_val[i]["train"]["acc"])
        results_df = {model_name: model_json_val[i]["train"]["acc"] for model_name, model_json_val in results_json.items()}
        # results_df = {"vanilla": results_baseline_multi[i]["test"]["acc"][test_lang],
        #               "rand": results_ltn_rand[i]["test"]["acc"][test_lang],
        #               "ltn_dem_prev": results_ltn_demprev[i]["test"]["acc"][test_lang],
        #               "ltn_dem_first": results_ltn_demfirst[i]["test"]["acc"][test_lang]}
        #create dataset
        df = pd.DataFrame(results_df)
        # df = pd.DataFrame(slots_results_dict)

        #plot individual lines
        plt.clf()
        for model_name in results_df:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        plt.title("Train Performance on "+ train_lang+" after training on "+train_lang)
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+lang_order+'/accuracy_teston-'+train_lang+'_aftertrain-'+train_lang+'.png')

def plot_er_results(task_name):
    # models = {"Vanilla":"Baseline/ltmode-cont-multi/"+lang_order+"/W_SCHEDULER_0_WARMUP/",
    #         #   "baseline_cont-mono":"Baseline/ltmode-cont-mono/"+lang_order+"/W_SCHEDULER_0_WARMUP/",
    #         #   "er_baseline_cont-mono": "Baseline/ER/ltmode-cont-mono/"+lang_order+"/W_SCHEDULER_0_WARMUP/",
    #           "er_baseline": "Baseline/ER/ltmode-cont-multi/"+lang_order+"/W_SCHEDULER_0_WARMUP/",
    #         #   "er_only_ltn_previous_cont-multi": "ER/LtnScheduler/DemotePrevious/ltnmodel-ltn/fifo/schedtype-er/ltmode-cont-multi/"+lang_order+"/ndecks-5/sample-epoch/update-batch/W_SCHEDULER_0_WARMUP/",
    #         #   "er_main_ltn_previous_cont-multi": "ER/LtnScheduler/DemotePrevious/ltnmodel-ltn/fifo/schedtype-main/ltmode-cont-multi/"+lang_order+"/ndecks-5/sample-epoch/update-batch/W_SCHEDULER_0_WARMUP/",
    #           "ltn_previous_cont-multi": "LtnScheduler/DemotePrevious/ltnmodel-ltn/fifo/schedtype-main/ltmode-cont-multi/"+lang_order+"/ndecks-5/sample-epoch/update-batch/W_SCHEDULER_0_WARMUP/",
    #           "ltn_previous_rand_cont-multi": "LtnScheduler/DemotePrevious/ltnmodel-ltn/rand/schedtype-main/ltmode-cont-multi/"+lang_order+"/ndecks-5/sample-epoch/update-batch/W_SCHEDULER_0_WARMUP/"}
    
    root_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/"+task_name.upper()+"/HyperparamSearch/BertBaseMultilingualCased/"
    lang_orders = ["en_vi_ar_tr"] #["en_vi_ar_tr", "vi_en_tr_ar", "ar_tr_en_vi", "tr_ar_vi_en"]
    ltmodes = ["cont-mono"] #["cont-mono", "cont-multi"]
    er_strategies = ["extreme", "random"]#, "extreme", "random"] #["easy", "hard", "extreme", "random"]
    modes = ["fifo"] #["fifo", "rand"]

    for lang_order in lang_orders:
        all_metrics = {}
        results_json = {}
        langs = lang_order.split("_")
        avg_forgetting_epochs = {}
        avg_final_perf_epochs = {}
        avg_fwt_perf_epochs = {}
        avg_acc_perf_epochs = {}

        # ## Vanilla
        # model_dir = "Baseline/ltmode-cont-mono/en_vi_ar_tr_bg_el_ur/W_SCHEDULER_0_WARMUP/"
        # root_results_dir = root_dir+model_dir+"metrics.json"

        # model_name = "vanilla"
        # avg_forgetting_epochs.update({model_name: []})
        # avg_final_perf_epochs.update({model_name: []})
        # avg_fwt_perf_epochs.update({model_name: []})
        # avg_acc_perf_epochs.update({model_name: []})
        # with open(root_results_dir) as file:
        #     results_json.update({model_name: json.load(file)})

        # for epoch in range(10):
        #     all_metrics.update({model_name: [[0.0 for _ in range(14)] for _ in range(7)]})
        
        #     for i, train_lang in enumerate(langs):
        #         for j, test_lang in enumerate(langs):
        #             all_metrics[model_name][i][2*j] = results_json[model_name][i]["test"]["acc"][test_lang][epoch]*100

        #     all_metrics_np = np.asarray(all_metrics[model_name])
        #     avg_forgetting_epochs[model_name].append(forget_avg(all_metrics_np, langs, langs)[4])
        #     avg_final_perf_epochs[model_name].append(final_perf(all_metrics_np, langs)[0])
        #     avg_fwt_perf_epochs[model_name].append(fwt_avg(all_metrics_np, langs, random_perf[task_name])[2])
        #     avg_acc_perf_epochs[model_name].append(acc_avg(all_metrics_np, langs, langs)[2])
        
        for ltmode in ltmodes:
            for er_strategy in er_strategies:
                for mode in modes:
                    model_dir = "LtnScheduler/ER/ER_PROP_0.0-use_er_only/DemotePrevious/ltnmodel-ltn/"+mode+"/schedtype-both_erstrategy-"+er_strategy+"/ltmode-"+ltmode+"/"+lang_order+"/ndecks-5/sample-epoch/update-epoch/W_SCHEDULER_0_WARMUP/"
                    root_results_dir = root_dir+model_dir+"metrics.json"
                    model_name = mode + "-" + ltmode + "-ER_" + er_strategy

                    avg_forgetting_epochs.update({model_name: []})
                    avg_final_perf_epochs.update({model_name: []})
                    avg_fwt_perf_epochs.update({model_name: []})
                    avg_acc_perf_epochs.update({model_name: []})

                    with open(root_results_dir) as file:
                        results_json.update({model_name: json.load(file)})
                    
                    for epoch in range(10):
                        all_metrics.update({model_name: [[0.0 for _ in range(14)] for _ in range(7)]})
                        
                        for i, train_lang in enumerate(langs):
                            for j, test_lang in enumerate(langs):
                                all_metrics[model_name][i][2*j] = results_json[model_name][i]["test"]["acc"][test_lang][epoch]*100

                        all_metrics_np = np.asarray(all_metrics[model_name])
                        avg_forgetting_epochs[model_name].append(forget_avg(all_metrics_np, langs, langs)[4])
                        avg_final_perf_epochs[model_name].append(final_perf(all_metrics_np, langs)[0])
                        avg_fwt_perf_epochs[model_name].append(fwt_avg(all_metrics_np, langs, random_perf[task_name])[2])
                        avg_acc_perf_epochs[model_name].append(acc_avg(all_metrics_np, langs, langs)[2])

        # Average forgetting
        df = pd.DataFrame(avg_forgetting_epochs)
        plot_save_dir = "/home1/mmhamdi/x-continuous-learning/outputs/Plots/spacedrepetition/Test/XNLI/cont_ER/" + lang_order + "/"
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)

        plt.clf()
        for model_name in avg_forgetting_epochs:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
        plt.title("Average Forgetting over epochs and test languages")
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+'average_forgetting.png')

        # Average final performance
        df = pd.DataFrame(avg_final_perf_epochs)

        plt.clf()
        for model_name in avg_final_perf_epochs:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
        plt.title("Average Final Performance over epochs and test languages")
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+'average_final_perf.png')

        # Average accuracy
        df = pd.DataFrame(avg_acc_perf_epochs)

        plt.clf()
        for model_name in avg_acc_perf_epochs:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
        plt.title("Average Accuracy over epochs and test languages")
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+'average_acc.png')

        # Forward Transfer
        df = pd.DataFrame(avg_fwt_perf_epochs)

        plt.clf()
        for model_name in avg_fwt_perf_epochs:
            plt.plot(df[model_name], label=model_name)

        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy")
        # plt.title(eval_type+' on after training on '+lang + ' in language_order: '+lang_order)
        plt.title("Average FWT over epochs and test languages")
        plt.legend(loc="lower right")
        plt.savefig(plot_save_dir+'average_fwt.png')

def clean_results_dir(task_name):
    root_dir = "/project/jonmay_231/meryem/ResultsSpacedRepetition/x-continuous-learn/"+task_name.upper()+"/HyperparamSearch/BertBaseMultilingualCased/"
    lang_orders = ["en_vi_ar_tr", "vi_en_tr_ar", "ar_tr_en_vi", "tr_ar_vi_en"]
    ltmodes = ["cont-mono", "cont-multi"]
    er_strategies = ["easy", "hard", "extreme", "random"]
    modes = ["fifo", "rand"]

    for lang_order in lang_orders:
        for ltmode in ltmodes:
            for er_strategy in er_strategies:
                for mode in modes:
                    model_dir = root_dir+"LtnScheduler/ER/ER_PROP_0.0-use_er_only/DemotePrevious/ltnmodel-ltn/"+mode+"/schedtype-both_erstrategy-"+er_strategy+"/ltmode-"+ltmode+"/"+lang_order+"/ndecks-5/sample-epoch/update-epoch/W_SCHEDULER_0_WARMUP/"

                    if os.path.exists(model_dir):
                        print(lang_order, ltmode, er_strategy, mode, " EXISTS")
                        ltn_scheduler_dir_decks = model_dir+"/lt_scheduler/decks/"
                        if not os.path.exists(ltn_scheduler_dir_decks):
                            os.makedirs(ltn_scheduler_dir_decks)

                        source_files = model_dir + "lt_scheduler_decks_*.json"
                        target_folder = ltn_scheduler_dir_decks

                        # retrieve file list
                        filelist=glob.glob(source_files)
                        for single_file in filelist:
                            # move file with full paths as shutil.move() parameters
                            shutil.move(single_file,target_folder) 

                        ## 
                        ltn_scheduler_dir_idmovements = model_dir+"/lt_scheduler/idmovements/"
                        if not os.path.exists(ltn_scheduler_dir_idmovements):
                            os.makedirs(ltn_scheduler_dir_idmovements)

                        source_files = model_dir + "lt_scheduler_idmovements_*.json"
                        target_folder = ltn_scheduler_dir_idmovements

                        # retrieve file list
                        filelist=glob.glob(source_files)
                        for single_file in filelist:
                            # move file with full paths as shutil.move() parameters
                            shutil.move(single_file,target_folder) 

                        

# plot_cont()
# plot_mono()
task_name="xnli"
# plot_cont_new(task_name)
plot_er_results(task_name)
# clean_results_dir(task_name)
    