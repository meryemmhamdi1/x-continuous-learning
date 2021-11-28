import pickle
import numpy as np
from summarize_metrics import acc_avg, fwt_avg, fwt_avg_mono, bwt_avg, forget_avg, final_perf

def compute_adapter_perf(root_dir, orders):
    languages = ["en", "de", "fr", "hi", "es"]
    mono_perf = {lang: (0.0, 0.0) for lang in languages}
    for lang_order in orders:
        for mode_freeze_bert in ["TUNED_BERT", "FROZEN_BERT"]:
            for mode_task_adapters in ["NO_TASK_ADAPTERS", "USE_TASK_ADAPTERS"]:
                # MONOLINGUAL PERFORMANCE
                for lang in languages:
                    file_name = root_dir+lang + "/NLU/BertBaseMultilingualCased/" \
                                + mode_freeze_bert + "/" + mode_task_adapters + \
                                "/TRAINING_ADAPTERS/USE_MONO_"+lang+"/all_metrics_10_epochs.pickle"

                    with open(file_name, "rb") as file:
                        results = pickle.load(file)

                    mono_perf[lang] = (results["train_"+lang]["test_"+lang][lang+"_0_test_intent_acc_"+lang+"_0"]*100,
                                       results["train_"+lang]["test_"+lang][lang+"_0_test_slot_f1_"+lang+"_0"]*100)

                # MAIN PERFORMANCE
                file_name = root_dir+lang_order + "/NLU/BertBaseMultilingualCased/"\
                            + mode_freeze_bert + "/" + mode_task_adapters + \
                            "/TRAINING_ADAPTERS/all_metrics_10_epochs.pickle"

                all_metrics = [[0.0 for _ in range(10)] for _ in range(5)]

                with open(file_name, "rb") as file:
                    data = pickle.load(file)

                for i, train_lang in enumerate(languages):
                    for j, test_lang in enumerate(languages):
                        intent_alias = [k for k, v in data["train_"+train_lang]["test_"+test_lang].items() if "intent_acc" in k][0]
                        slot_alias = [k for k, v in data["train_"+train_lang]["test_"+test_lang].items() if "slot_f1" in k][0]

                        all_metrics[i][2*j] = data["train_"+train_lang]["test_"+test_lang][intent_alias] * 100
                        all_metrics[i][2*j+1] = data["train_"+train_lang]["test_"+test_lang][slot_alias] * 100

                all_metrics_np = np.asarray(all_metrics)

                acc_avg_all, fwt_avg_mono_all, forget_perf_all, final_perf_all = \
                    acc_avg(all_metrics_np, languages, languages), \
                    fwt_avg_mono(all_metrics_np, languages, mono_perf), \
                    forget_avg(all_metrics_np, languages, languages), \
                    final_perf(all_metrics_np, languages)

                results_array = [acc_avg_all[2], acc_avg_all[3], np.mean([acc_avg_all[2], acc_avg_all[3]]),
                                 bwt_avg_all[2], bwt_avg_all[3], np.mean([bwt_avg_all[2], bwt_avg_all[3]]),
                                 forget_perf_all[2], forget_perf_all[3], np.mean([forget_perf_all[2], forget_perf_all[3]]),
                                 fwt_avg_mono_all[2], fwt_avg_mono_all[3], np.mean([fwt_avg_mono_all[2], fwt_avg_mono_all[3]]),
                                 final_perf_all[0], final_perf_all[1], np.mean([final_perf_all[0], final_perf_all[1]])]

                results_array = list(map(lambda x: round(x, 2), results_array))

                print(lang_order, mode_freeze_bert, mode_task_adapters, results_array)

if __name__ == "__main__":
    root_dir = ""
    orders = []
    compute_adapter_perf(root_dir, orders)
