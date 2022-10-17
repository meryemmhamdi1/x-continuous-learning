import pickle
import numpy as np

models_dict = {"vanilla": repr("\naiveft{}"),
               "mono_orig": "\langspec{}",
               "mono_ada_tuned": "\langspecadatuned{}",
               "mono_ada_frozen": "\langspecadafrozen{}",
               "joint_plus": "\jointinc{}",
               "multi": "\multi{}",
               "multi_head_inembed": "\specemb{}",
               "multi_head_inenc.0-enc.1-enc.2": "\specencaa{}",
               "multi_head_inenc.3-enc.4-enc.5": "\specencbb{}",
               "multi_head_inenc.6-enc.7-enc.8": "\specenccc{}",
               "multi_head_inenc.9-enc.10-enc.11": "\specencb{}",
               "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8": "\specenca{}",
               "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool": "\spectrans{}",
               "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11": "\specencall{}",
               "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5": "\specencaaa{}",
               "multi_head_inenc.6-enc.7-enc.8-enc.9-enc.10-enc.11": "\specencbbb{}",
               "multi_head_out": "\spechead{}",
               "adapters/TUNED_BERT": "\\adatuned{}",
               "adapters/FROZEN_BERT": "\\adafrozen{}",
               "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01": "\ewc{}",
               "er_memsz-750_type-reservoir_sample-random_k-16": "\er{}-750",
               "er_memsz-1500_type-reservoir_sample-random_k-16": "\er{}-1500",
               "er_memsz-3000_type-reservoir_sample-random_k-16": "\er{}-3000",
               "er_memsz-4500_type-reservoir_sample-random_k-16": "\er{}-4500",
               "er_memsz-6000_type-reservoir_sample-random_k-16": "\er{}-6000",
               "kd-logits_memsz-6000_type-reservoir_sample-random_k-16": "\kdlogit{}",
               "kd-rep_memsz-6000_type-reservoir_sample-random_k-16": "\kdrep{}"
               }

lang_orders_dict = {
    "en_de_fr_hi_es_th": "EN->DE->FR->HI->ES->TH",
    "de_fr_th_es_en_hi": "DE->FR->TH->ES->EN->HI",
    "fr_th_de_en_hi_es": "FR->TH->DE->EN->HI->ES",
    "hi_en_es_th_fr_de": "HI->EN->ES->TH->FR->DE",
    "es_hi_en_de_th_fr": "ES->HI->EN->DE->TH->FR",
    "th_es_hi_fr_de_en": "TH->ES->HI->FR->DE->EN"
}

def table2_3(all_avg_metrics, mono_orig_line_intent, mono_orig_line_slot, multi_line_intent, multi_line_slot):
    # TODO seperate them
    # TODO "%.2f" % round(num, 2)
    avg_metrics_dict = {}
    for avg_metric in all_avg_metrics:
        method = avg_metric[0]
        avg_metrics_dict.update({method: avg_metric})

    method = "vanilla"
    avg_metric = avg_metrics_dict[method]
    line = models_dict[method] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
           str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
           str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
           str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
           str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
           str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
    print(line.replace("'", ''))

    mono_str = models_dict["mono_orig"] + " &  & &  &  & " + mono_orig_line_intent + " & "+ mono_orig_line_slot
    print(mono_str.replace("'", ''))

    method = "joint_plus"
    avg_metric = avg_metrics_dict[method]
    line = models_dict[method] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ &" + \
           str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ &" + \
           str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ &" + \
           str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ &" + \
           str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ &" + \
           str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
    print(line.replace("'", ''))

    multi_str = models_dict["multi"] + " &  & &  &  & " + multi_line_intent + " & " + multi_line_slot
    print(multi_str.replace("'", ''))

def table2(all_avg_metrics, mono_orig_line_intent, mono_orig_line_slot, multi_line_intent, multi_line_slot):
    print("TABLE 2")
    avg_metrics_dict = {}
    for avg_metric in all_avg_metrics:
        method = avg_metric[0]
        avg_metrics_dict.update({method: avg_metric})

    method = "vanilla"
    avg_metric = avg_metrics_dict[method]
    line = models_dict[method] + " & " + str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ &" + \
           str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ " + repr("\\")
    print(line.replace("'", ''))

    mono_str = models_dict["mono_orig"] + " & " + mono_orig_line_intent + " & " + mono_orig_line_slot + repr("\\")
    print(mono_str.replace("'", ''))

    method = "joint_plus"
    avg_metric = avg_metrics_dict[method]
    line = models_dict[method] + " & " + str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ &" + \
           str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ " + repr("\\")
    print(line.replace("'", ''))

    multi_str = models_dict["multi"] + " & " + multi_line_intent + " & " + multi_line_slot + repr("\\")
    print(multi_str.replace("'", ''))

def table3(all_avg_metrics):
    print("TABLE 3")
    avg_metrics_dict = {}
    for avg_metric in all_avg_metrics:
        method = avg_metric[0]
        avg_metrics_dict.update({method: avg_metric})

    method = "vanilla"
    avg_metric = avg_metrics_dict[method]
    line = models_dict[method] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
           str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
           str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
           str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$" + repr("\\")
    print(line.replace("'", ''))

    method = "joint_plus"
    avg_metric = avg_metrics_dict[method]
    line = models_dict[method] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ &" + \
           str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ &" + \
           str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ &" + \
           str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) +  "$" + repr("\\")
    print(line.replace("'", ''))

def table4(all_per_order_metrics):
    print("TABLE 4")
    per_order_metrics_dict = {}
    for per_orders_metrics in all_per_order_metrics:
        for per_order_metric in per_orders_metrics:
            model = per_order_metric[0]
            lang_order = per_order_metric[1]
            if model not in per_order_metrics_dict:
                per_order_metrics_dict.update({model: {}})
            per_order_metrics_dict[model].update({lang_order: per_order_metric})

    for method in ["vanilla",
                   "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                   # "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                   "multi_head_out",
                   "adapters/TUNED_BERT",
                   # "adapters/FROZEN_BERT",
                   "ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                   "er_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-logits_memsz-6000_type-reservoir_sample-random_k-16",
                   "kd-rep_memsz-6000_type-reservoir_sample-random_k-16"]:
        # line = models_dict[method] + " & "
        line_w_stds = models_dict[method] + " & "
        high2low_metric = per_order_metrics_dict[method]["en_de_fr_hi_es_th"]
        low2high_metric = per_order_metrics_dict[method]["th_es_hi_fr_de_en"]

        # line += str(high2low_metric[2]) + " & " + str(low2high_metric[2]) + " & " + \
        #         str(high2low_metric[10]) + " & " + str(low2high_metric[10]) + " & " + \
        #         str(high2low_metric[14]) + " & " + str(low2high_metric[14])  +  " " + repr("\\")

        line_w_stds += str(high2low_metric[2]) + " $\pm " + str(high2low_metric[3]) + "$ & " + \
                       str(low2high_metric[2]) + " $\pm " + str(low2high_metric[3]) + "$ & " + \
                       str(high2low_metric[10]) + " $\pm " + str(high2low_metric[11]) + "$ & " + \
                       str(low2high_metric[10]) + " $\pm " + str(low2high_metric[11]) + "$ & " + \
                       str(high2low_metric[14]) + " $\pm " + str(high2low_metric[15]) + "$ & " + \
                       str(low2high_metric[14]) + " $\pm " + str(low2high_metric[15]) + "$ "+repr("\\")

        print(line_w_stds.replace("'", ''))

def table5(root_dir):
    print("TABLE 5")
    with open(root_dir+"cll-equals/all_per_order_metrics_bootstrap.pickle", "rb") as file:
        all_per_order_metrics = pickle.load(file)

    per_order_metrics_dict = {}
    for per_orders_metrics in all_per_order_metrics:
        for per_order_metric in per_orders_metrics:
            model = per_order_metric[0]
            lang_order = per_order_metric[1]
            if model not in per_order_metrics_dict:
                per_order_metrics_dict.update({model: {}})
            per_order_metrics_dict[model].update({lang_order: per_order_metric})


    for method in per_order_metrics_dict:
        for lang_order in lang_orders_dict:
            metric = per_order_metrics_dict[method][lang_order]
            line = lang_orders_dict[lang_order] + " & " + str(metric[2]) + " $\pm " + str(metric[3]) + "$ & " + \
                   str(metric[4]) + " $\pm " + str(metric[5]) + "$ & " + \
                   str(metric[10]) + " $\pm " + str(metric[11]) + "$ & " + \
                   str(metric[12]) + " $\pm " + str(metric[13]) + "$ & " + \
                   str(metric[14]) + " $\pm " + str(metric[15]) + "$ & " + \
                   str(metric[16]) + " $\pm " + str(metric[17]) + "$ "+repr("\\")

            print(line.replace("'", ''))


def table8(all_avg_metrics, mono_orig_line_intent, mono_orig_line_slot, mono_ada_tuned_line_intent,
           mono_ada_tuned_line_slot, mono_ada_frozen_line_intent, mono_ada_frozen_line_slot, multi_line_intent,
           multi_line_slot):
    print("TABLE 8")
    avg_metrics_dict = {}
    for avg_metric in all_avg_metrics:
        avg_metrics_dict.update({avg_metric[0]: avg_metric})

    lines = []

    avg_metric = avg_metrics_dict["vanilla"]
    line = models_dict["vanilla"] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
           str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
           str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
           str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
           str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
           str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
           str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
           str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
    lines.append(line.replace("'", ''))

    mono_str = models_dict["mono_orig"] + " &  & &  &  &  & & " + mono_orig_line_intent + " & " + mono_orig_line_slot+" "+repr("\\")
    mono_ada_tuned_str = models_dict["mono_ada_tuned"] + " &  & &  &  &  & & " + mono_ada_tuned_line_intent + " & " + mono_ada_tuned_line_slot+" "+repr("\\")
    mono_ada_frozen_str = models_dict["mono_ada_frozen"] + " &  & &  &  &  & & " + mono_ada_frozen_line_intent + " & " + mono_ada_frozen_line_slot+" "+repr("\\")

    lines.append(mono_str.replace("'", ''))
    lines.append(mono_ada_tuned_str.replace("'", ''))
    lines.append(mono_ada_frozen_str.replace("'", ''))

    avg_metric = avg_metrics_dict["joint_plus"]
    line = models_dict["joint_plus"] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
           str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
           str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
           str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
           str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
           str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
           str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
           str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
    lines.append(line.replace("'", ''))

    multi_line = models_dict["multi"] + " &  & &  &  &  & & " + multi_line_intent + " & " + multi_line_slot+" "+repr("\\")

    lines.append(multi_line.replace("'", ''))
    line_ = repr("\r")+"owcolor{lightgray} \multicolumn{9}{c}{Model Expansion Baselines}"+repr("\\")+ "\midrule"
    lines.append(line_.replace("'", ''))
    for model in ["multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                  "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                  "multi_head_out",
                  "adapters/TUNED_BERT",
                  "adapters/FROZEN_BERT"]:
        avg_metric = avg_metrics_dict[model]
        line = repr("\textbf{") + models_dict[model] + "} & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
               str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
               str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
               str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
               str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
               str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
               str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
               str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
        lines.append(line.replace("'", ''))

    line_ = "\midrule "+repr("\r")+"owcolor{lightgray}  \multicolumn{9}{c}{Other Continuous Learning Algorithms }"+repr("\\")+"\midrule"
    lines.append(line_.replace("'", ''))
    for model in ["ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                  "er_memsz-6000_type-reservoir_sample-random_k-16",
                  "kd-logits_memsz-6000_type-reservoir_sample-random_k-16",
                  "kd-rep_memsz-6000_type-reservoir_sample-random_k-16"]:
        avg_metric = avg_metrics_dict[model]
        line = repr("\textbf{")+ models_dict[model] + "} & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
               str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
               str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
               str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
               str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
               str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
               str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
               str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ " +repr("\\")
        if model not in ["kd-logits_memsz-6000_type-reservoir_sample-random_k-16", "kd-rep_memsz-6000_type-reservoir_sample-random_k-16"]:
            line += " \midrule "
        lines.append(line.replace("'", ''))

    for line in lines:
        print(line)


def table_10_11_12(all_per_order_metrics, mono_orig_line_intent, mono_orig_line_slot, mono_ada_tuned_line_intent,
           mono_ada_tuned_line_slot, mono_ada_frozen_line_intent, mono_ada_frozen_line_slot, multi_line_intent,
           multi_line_slot, lang_order_1, lang_order_2):
    print("TABLE ORDERS")
    per_order_metrics_dict = {}
    for per_orders_metrics in all_per_order_metrics:
        for per_order_metric in per_orders_metrics:
            model = per_order_metric[0]
            lang_order = per_order_metric[1]
            if model not in per_order_metrics_dict:
                per_order_metrics_dict.update({model: [[], []]})
            if lang_order == lang_order_1:
                per_order_metrics_dict[model][0] = per_order_metric
            elif lang_order == lang_order_2:
                per_order_metrics_dict[model][1] = per_order_metric

    intent_lines = []
    slot_lines = []

    method = "vanilla"
    order_0_metric = per_order_metrics_dict[method][0]
    order_1_metric = per_order_metrics_dict[method][1]
    intent_line = models_dict[method] + " & " + str(order_0_metric[2]) + " $\pm " + str(order_0_metric[3]) + "$ & " + \
                str(order_0_metric[10]) + " $\pm " + str(order_0_metric[11]) + "$ & " + \
                str(order_0_metric[6]) + " $\pm " + str(order_0_metric[7]) + "$ & " + \
                "\multicolumn{}{c||}{" +str(order_0_metric[14]) + " $\pm " + str(order_0_metric[15]) + "$ } & " +\
                str(order_1_metric[2]) + " $\pm " + str(order_1_metric[3]) + "$ & " + \
                str(order_1_metric[10]) + " $\pm " + str(order_1_metric[11]) + "$ & " + \
                str(order_1_metric[6]) + " $\pm " + str(order_1_metric[7]) + "$ & " + \
                str(order_1_metric[14]) + " $\pm " + str(order_1_metric[15]) + "$ "+repr("\\")

    slot_line = models_dict[method] + " & " + str(order_0_metric[4]) + " $\pm " + str(order_0_metric[5]) + "$ & " + \
              str(order_0_metric[12]) + " $\pm " + str(order_0_metric[13]) + "$ & " + \
              str(order_0_metric[8]) + " $\pm " + str(order_0_metric[9]) + "$ & " + \
              "\multicolumn{}{c||}{" + str(order_0_metric[16]) + " $\pm " + str(order_0_metric[17]) + "$ } & " + \
              str(order_1_metric[4]) + " $\pm " + str(order_1_metric[5]) + "$ & " + \
              str(order_1_metric[12]) + " $\pm " + str(order_1_metric[13]) + "$ & " + \
              str(order_1_metric[8]) + " $\pm " + str(order_1_metric[9]) + "$ & " + \
              str(order_1_metric[16]) + " $\pm " + str(order_1_metric[17]) + "$ "+repr("\\")

    intent_lines.append(intent_line.replace("'", ''))
    slot_lines.append(slot_line.replace("'", ''))

    intent_lines.append(models_dict["mono_orig"] + " &  & & & \multicolumn{}{c||}{" + mono_orig_line_intent+ "}  &  & & & "+mono_orig_line_intent+repr("\\"))
    intent_lines.append(models_dict["mono_ada_tuned"] + " &  & & & \multicolumn{}{c||}{" + mono_ada_tuned_line_intent + "}  &  & & & " + mono_ada_tuned_line_intent+repr("\\"))
    intent_lines.append(models_dict["mono_ada_frozen"] + " &  & & & \multicolumn{}{c||}{" + mono_ada_frozen_line_intent+ "}  &  & & & " + mono_ada_frozen_line_intent+repr("\\"))

    slot_lines.append(models_dict["mono_orig"] + " &  & & & \multicolumn{}{c||}{" + mono_orig_line_slot + "}  &  & & & " +mono_orig_line_slot+repr("\\"))
    slot_lines.append(models_dict["mono_ada_tuned"] + " &  & & & \multicolumn{}{c||}{" + mono_ada_tuned_line_slot+ "}  &  & & & " +mono_ada_tuned_line_slot+repr("\\"))
    slot_lines.append(models_dict["mono_ada_frozen"] + " &  & & & \multicolumn{}{c||}{" + mono_ada_frozen_line_slot+ "}  &  & & & " +mono_ada_frozen_line_slot+repr("\\"))

    method = "joint_plus"
    order_0_metric = per_order_metrics_dict[method][0]
    order_1_metric = per_order_metrics_dict[method][1]
    intent_line = models_dict[method] + " & " + str(order_0_metric[2]) + " $\pm " + str(order_0_metric[3]) + "$ & " + \
                  str(order_0_metric[10]) + " $\pm " + str(order_0_metric[11]) + "$ & " + \
                  str(order_0_metric[6]) + " $\pm " + str(order_0_metric[7]) + "$ & " + \
                  "\multicolumn{}{c||}{" + str(order_0_metric[14]) + " $\pm " + str(order_0_metric[15]) + "$ } & " + \
                  str(order_1_metric[2]) + " $\pm " + str(order_1_metric[3]) + "$ & " + \
                  str(order_1_metric[10]) + " $\pm " + str(order_1_metric[11]) + "$ & " + \
                  str(order_1_metric[6]) + " $\pm " + str(order_1_metric[7]) + "$ & " + \
                  str(order_1_metric[14]) + " $\pm " + str(order_1_metric[15]) + "$ "+repr("\\")

    slot_line = models_dict[method] + " & " + str(order_0_metric[4]) + " $\pm " + str(order_0_metric[5]) + "$ &" + \
                str(order_0_metric[12]) + " $\pm " + str(order_0_metric[13]) + "$ & " + \
                str(order_0_metric[8]) + " $\pm " + str(order_0_metric[9]) + "$ & " + \
                "\multicolumn{}{c||}{" + str(order_0_metric[16]) + " $\pm " + str(order_0_metric[17]) + "$ } & " + \
                str(order_1_metric[4]) + " $\pm " + str(order_1_metric[5]) + "$ & " + \
                str(order_1_metric[12]) + " $\pm " + str(order_1_metric[13]) + "$ & " + \
                str(order_1_metric[8]) + " $\pm " + str(order_1_metric[9]) + "$ & " + \
                str(order_1_metric[16]) + " $\pm " + str(order_1_metric[17]) + "$ "+repr("\\")
    intent_lines.append(intent_line.replace("'", ''))
    slot_lines.append(slot_line.replace("'", ''))

    intent_lines.append(models_dict["multi"] + " &  & & & \multicolumn{}{c||}{" + multi_line_intent+ "}  &  & & & "+multi_line_intent+repr("\\"))
    slot_lines.append(models_dict["multi"] + " &  & & & \multicolumn{}{c||}{" + multi_line_slot+ "}  &  & & & "+multi_line_slot+repr("\\"))

    intent_lines.append(repr("\r")+"owcolor{lightgray} \multicolumn{9}{c}{Model Expansion Baselines}"+repr("\\")+" \midrule")
    slot_lines.append(repr("\r")+"owcolor{lightgray} \multicolumn{9}{c}{Model Expansion Baselines}"+repr("\\")+" \midrule")
    for model in ["multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                  "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                  "multi_head_out",
                  "adapters/TUNED_BERT",
                  "adapters/FROZEN_BERT"]:
        order_0_metric = per_order_metrics_dict[model][0]
        order_1_metric = per_order_metrics_dict[model][1]
        intent_line = repr("\t")+"extbf{" + models_dict[model] + "} & " + str(order_0_metric[2]) + " $\pm " + str(order_0_metric[3]) + "$ & " + \
               str(order_0_metric[10]) + " $\pm " + str(order_0_metric[11]) + "$ & " + \
               str(order_0_metric[6]) + " $\pm " + str(order_0_metric[7]) + "$ & " + \
               "\multicolumn{}{c||}{" + str(order_0_metric[14]) + " $\pm " + str(order_0_metric[15]) + "$ } & " + \
               str(order_1_metric[2]) + " $\pm " + str(order_1_metric[3]) + "$ & " + \
               str(order_1_metric[10]) + " $\pm " + str(order_1_metric[11]) + "$ & " + \
               str(order_1_metric[6]) + " $\pm " + str(order_1_metric[7]) + "$ & " + \
               str(order_1_metric[14]) + " $\pm " + str(order_1_metric[15]) + "$ "+repr("\\")

        slot_line = repr("\t")+"extbf{" + models_dict[model] + "} & " + str(order_0_metric[4]) + " $\pm " + str(order_0_metric[5]) + "$ & " + \
                    str(order_0_metric[12]) + " $\pm " + str(order_0_metric[13]) + "$ & " + \
                    str(order_0_metric[8]) + " $\pm " + str(order_0_metric[9]) + "$ & " + \
                    "\multicolumn{}{c||}{" + str(order_0_metric[16]) + " $\pm " + str(order_0_metric[17]) + "$ } & " + \
                    str(order_1_metric[4]) + " $\pm " + str(order_1_metric[5]) + "$ & " + \
                    str(order_1_metric[12]) + " $\pm " + str(order_1_metric[13]) + "$ & " + \
                    str(order_1_metric[8]) + " $\pm " + str(order_1_metric[9]) + "$ & " + \
                    str(order_1_metric[16]) + " $\pm " + str(order_1_metric[17]) + "$ "+repr("\\")
        intent_lines.append(intent_line.replace("'", ''))
        slot_lines.append(slot_line.replace("'", ''))

    intent_lines.append("\midrule "+repr("\r")+"owcolor{lightgray}  \multicolumn{9}{c}{Other Continuous Learning Algorithms }"+repr("\\")+" \midrule")
    slot_lines.append("\midrule "+repr("\r")+"owcolor{lightgray}  \multicolumn{9}{c}{Other Continuous Learning Algorithms }"+repr("\\")+" \midrule")
    for model in ["ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                  "er_memsz-6000_type-reservoir_sample-random_k-16",
                  "kd-logits_memsz-6000_type-reservoir_sample-random_k-16",
                  "kd-rep_memsz-6000_type-reservoir_sample-random_k-16"]:
        order_0_metric = per_order_metrics_dict[model][0]
        order_1_metric = per_order_metrics_dict[model][1]
        intent_line = repr("\t")+"extbf{" + models_dict[model] + "} & " + str(order_0_metric[2]) + " $\pm " + str(order_0_metric[3]) + "$ & " + \
                      str(order_0_metric[10]) + " $\pm " + str(order_0_metric[11]) + "$ & " + \
                      str(order_0_metric[6]) + " $\pm " + str(order_0_metric[7]) + "$ & " + \
                      "\multicolumn{}{c||}{" + str(order_0_metric[14]) + " $\pm " + str(order_0_metric[15]) + "$ } & " + \
                      str(order_1_metric[2]) + " $\pm " + str(order_1_metric[3]) + "$ & " + \
                      str(order_1_metric[10]) + " $\pm " + str(order_1_metric[11]) + "$ & " + \
                      str(order_1_metric[6]) + " $\pm " + str(order_1_metric[7]) + "$ & " + \
                      str(order_1_metric[14]) + " $\pm " + str(order_1_metric[15]) + "$ "+repr("\\")

        slot_line = repr("\t")+"extbf{" + models_dict[model] + "} & " + str(order_0_metric[4]) + " $\pm " + str(order_0_metric[5]) + "$ & " + \
                    str(order_0_metric[12]) + " $\pm " + str(order_0_metric[13]) + "$ & " + \
                    str(order_0_metric[8]) + " $\pm " + str(order_0_metric[9]) + "$ & " + \
                    "\multicolumn{}{c||}{" + str(order_0_metric[16]) + " $\pm " + str(order_0_metric[17]) + "$ } &" + \
                    str(order_1_metric[4]) + " $\pm " + str(order_1_metric[5]) + "$ & " + \
                    str(order_1_metric[12]) + " $\pm " + str(order_1_metric[13]) + "$ & " + \
                    str(order_1_metric[8]) + " $\pm " + str(order_1_metric[9]) + "$ & " + \
                    str(order_1_metric[16]) + " $\pm " + str(order_1_metric[17]) + "$ "+repr("\\")

        if model != "kd-logits_memsz-6000_type-reservoir_sample-random_k-16":
            intent_line += " \midrule "
            slot_line += " \midrule "

        intent_lines.append(intent_line.replace("'", ''))
        slot_lines.append(slot_line.replace("'", ''))

    for line in intent_lines:
        print(line.replace("'", ''))

    sep_line = "& \multicolumn{8}{c}{"+repr("\t")+"extbf{Test Slot Filling On}}"+repr("\\") + repr("\r")\
               +"owcolor{lightgray} \multicolumn{9}{c}{Shared \{Trans, Task\} Baselines } "+repr("\\")+"\midrule"
    print(sep_line.replace("'", ''))

    for line in slot_lines:
        print(line.replace("'", ''))


def table9(all_avg_metrics, mono_orig_line_intent, mono_orig_line_slot):
    print("TABLE 9")
    avg_metrics_dict = {}
    for avg_metric in all_avg_metrics:
        avg_metrics_dict.update({avg_metric[0]: avg_metric})

    model = "vanilla"
    avg_metric = avg_metrics_dict[model]
    line = models_dict[model] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
           str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
           str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
           str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
           str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
           str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
           str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
           str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
    print(line.replace("'", '') + " \midrule")

    for model in ["er_memsz-750_type-reservoir_sample-random_k-16",
                   "er_memsz-1500_type-reservoir_sample-random_k-16",
                   "er_memsz-3000_type-reservoir_sample-random_k-16",
                   "er_memsz-4500_type-reservoir_sample-random_k-16",
                   "er_memsz-6000_type-reservoir_sample-random_k-16"]:
        avg_metric = avg_metrics_dict[model]
        line = models_dict[model] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
               str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
               str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
               str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
               str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
               str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
               str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
               str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ " + repr("\\")
        print(line.replace("'", '') + " \midrule")

    mono_str = models_dict["mono_orig"] + " &  & &  &  &  & & " + mono_orig_line_intent + " & " + mono_orig_line_slot + repr("\\")
    print(mono_str.replace("'", ''))

    model = "multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool"
    avg_metric = avg_metrics_dict[model]
    line = models_dict[model] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
           str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
           str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
           str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
           str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
           str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
           str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
           str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
    print(line.replace("'", '') + " \midrule")

    for model in ["multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11", "multi_head_inembed"]:
        avg_metric = avg_metrics_dict[model]
        line = models_dict[model] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
               str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
               str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
               str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
               str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
               str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
               str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
               str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
        print(line.replace("'", ''))
    print("\midrule")

    for model in ["multi_head_inenc.0-enc.1-enc.2",
                  "multi_head_inenc.3-enc.4-enc.5",
                  "multi_head_inenc.6-enc.7-enc.8",
                  "multi_head_inenc.9-enc.10-enc.11"]:
        avg_metric = avg_metrics_dict[model]
        line = models_dict[model] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
               str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
               str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
               str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
               str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
               str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
               str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
               str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
        print(line.replace("'", ''))
    print("\midrule")

    for model in ["multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5",
                  "multi_head_inenc.6-enc.7-enc.8-enc.9-enc.10-enc.11"]:
        avg_metric = avg_metrics_dict[model]
        line = models_dict[model] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
               str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
               str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
               str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
               str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
               str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
               str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
               str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
        print(line.replace("'", ''))
    print("\midrule")

    for model in ["multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                  "multi_head_inenc.9-enc.10-enc.11"]:
        avg_metric = avg_metrics_dict[model]
        line = models_dict[model] + " & " + str(avg_metric[2]) + " $\pm " + str(avg_metric[3]) + "$ & " + \
               str(avg_metric[4]) + " $\pm " + str(avg_metric[5]) + "$ & " + \
               str(avg_metric[10]) + " $\pm " + str(avg_metric[11]) + "$ & " + \
               str(avg_metric[12]) + " $\pm " + str(avg_metric[13]) + "$ & " + \
               str(avg_metric[6]) + " $\pm " + str(avg_metric[7]) + "$ & " + \
               str(avg_metric[8]) + " $\pm " + str(avg_metric[9]) + "$ & " + \
               str(avg_metric[14]) + " $\pm " + str(avg_metric[15]) + "$ & " + \
               str(avg_metric[16]) + " $\pm " + str(avg_metric[17]) + "$ "+repr("\\")
        print(line.replace("'", ''))


def table_14_15_16(all_lang_metrics, all_avg_metrics, fwt_mono=False):
    print("TABLE ")
    model_names_lang = [avg_metric[0] for avg_metric in all_avg_metrics]
    lang_metrics_dict = {}
    for i, lang_metric in enumerate(all_lang_metrics):
        lang_metrics_dict.update({model_names_lang[i]: lang_metric})

    intent_lines = []
    slot_lines = []
    for model in ["vanilla", "joint_plus"]:
        lang_metric_i = lang_metrics_dict[model]
        if not fwt_mono:
            intent_line = models_dict[model] + " & " + \
                          str(lang_metric_i[2]) + " $\pm " + str(lang_metric_i[3]) + "$ & " + \
                          str(lang_metric_i[10]) + " $\pm " + str(lang_metric_i[11]) + "$ & " + \
                          str(lang_metric_i[18]) + " $\pm " + str(lang_metric_i[19]) + "$ & " + \
                          str(lang_metric_i[26]) + " $\pm " + str(lang_metric_i[27]) + "$ & " + \
                          str(lang_metric_i[34]) + " $\pm " + str(lang_metric_i[35]) + "$ & " + \
                          str(lang_metric_i[42]) + " $\pm " + str(lang_metric_i[43]) + "$ "+repr("\\")

            slot_line = models_dict[model] + " & " + \
                        str(lang_metric_i[5]) + " $\pm " + str(lang_metric_i[6]) + "$ & " + \
                        str(lang_metric_i[13]) + " $\pm " + str(lang_metric_i[14]) + "$ & " + \
                        str(lang_metric_i[21]) + " $\pm " + str(lang_metric_i[22]) + "$ & " + \
                        str(lang_metric_i[29]) + " $\pm " + str(lang_metric_i[30]) + "$ & " + \
                        str(lang_metric_i[37]) + " $\pm " + str(lang_metric_i[38]) + "$ & " + \
                        str(lang_metric_i[45]) + " $\pm " + str(lang_metric_i[46]) + "$ "+repr("\\")
        else:
            intent_line = models_dict[model] + " & " + \
                          str(lang_metric_i[2]) + " $\pm " + str(lang_metric_i[3]) + "$ & " + \
                          str(lang_metric_i[9]) + " $\pm " + str(lang_metric_i[10]) + "$ & " + \
                          str(lang_metric_i[16]) + " $\pm " + str(lang_metric_i[17]) + "$ & " + \
                          str(lang_metric_i[23]) + " $\pm " + str(lang_metric_i[24]) + "$ & " + \
                          str(lang_metric_i[30]) + " $\pm " + str(lang_metric_i[31]) + "$ & " + \
                          str(lang_metric_i[37]) + " $\pm " + str(lang_metric_i[38]) + "$ "+repr("\\")

            slot_line = models_dict[model] + " & " + \
                        str(lang_metric_i[5]) + " $\pm " + str(lang_metric_i[6]) + "$ & " + \
                        str(lang_metric_i[12]) + " $\pm " + str(lang_metric_i[13]) + "$ & " + \
                        str(lang_metric_i[19]) + " $\pm " + str(lang_metric_i[20]) + "$ & " + \
                        str(lang_metric_i[26]) + " $\pm " + str(lang_metric_i[27]) + "$ & " + \
                        str(lang_metric_i[33]) + " $\pm " + str(lang_metric_i[34]) + "$ & " + \
                        str(lang_metric_i[40]) + " $\pm " + str(lang_metric_i[41]) + "$ "+repr("\\")

        if model == "joint_plus":
            intent_line += "\midrule"
            slot_line += "\midrule"

        intent_lines.append(intent_line)
        slot_lines.append(slot_line)

    intent_lines.append(repr("\r")+"owcolor{lightgray} \multicolumn{7}{c}{Model Expansion Baselines}"+repr("\\")+" \midrule")
    slot_lines.append(repr("\r")+"owcolor{lightgray} \multicolumn{7}{c}{Model Expansion Baselines}"+repr("\\")+" \midrule")
    for model in ["multi_head_inembed-enc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8-enc.9-enc.10-enc.11-pool",
                  "multi_head_inenc.0-enc.1-enc.2-enc.3-enc.4-enc.5-enc.6-enc.7-enc.8",
                  "multi_head_out",
                  "adapters/TUNED_BERT",
                  "adapters/FROZEN_BERT"]:

        lang_metric_i = lang_metrics_dict[model]
        if not fwt_mono:
            intent_line = repr("\t")+"extbf{" + models_dict[model] + "} & "+ \
                          str(lang_metric_i[2]) + " $\pm " + str(lang_metric_i[3]) + "$ & " + \
                          str(lang_metric_i[10]) + " $\pm " + str(lang_metric_i[11]) + "$ & " + \
                          str(lang_metric_i[18]) + " $\pm " + str(lang_metric_i[19]) + "$ & " + \
                          str(lang_metric_i[26]) + " $\pm " + str(lang_metric_i[27]) + "$ & " + \
                          str(lang_metric_i[34]) + " $\pm " + str(lang_metric_i[35]) + "$ & " + \
                          str(lang_metric_i[42]) + " $\pm " + str(lang_metric_i[43]) + "$ "+repr("\\")

            slot_line = repr("\t")+"extbf{" + models_dict[model] + "} & "+ \
                        str(lang_metric_i[5]) + " $\pm " + str(lang_metric_i[6]) + "$ & " + \
                        str(lang_metric_i[13]) + " $\pm " + str(lang_metric_i[14]) + "$ & " + \
                        str(lang_metric_i[21]) + " $\pm " + str(lang_metric_i[22]) + "$ & " + \
                        str(lang_metric_i[29]) + " $\pm " + str(lang_metric_i[30]) + "$ & " + \
                        str(lang_metric_i[37]) + " $\pm " + str(lang_metric_i[38]) + "$ & " + \
                        str(lang_metric_i[45]) + " $\pm " + str(lang_metric_i[46]) + "$ "+repr("\\")
        else:
            intent_line = repr("\t")+"extbf{" + models_dict[model] + "} & "+ \
                          str(lang_metric_i[2]) + " $\pm " + str(lang_metric_i[3]) + "$ & " + \
                          str(lang_metric_i[9]) + " $\pm " + str(lang_metric_i[10]) + "$ & " + \
                          str(lang_metric_i[16]) + " $\pm " + str(lang_metric_i[17]) + "$ & " + \
                          str(lang_metric_i[23]) + " $\pm " + str(lang_metric_i[24]) + "$ & " + \
                          str(lang_metric_i[30]) + " $\pm " + str(lang_metric_i[31]) + "$ & " + \
                          str(lang_metric_i[37]) + " $\pm " + str(lang_metric_i[38]) + "$ "+repr("\\")

            slot_line = repr("\t")+"extbf{" + models_dict[model] + "} & "+ \
                        str(lang_metric_i[5]) + " $\pm " + str(lang_metric_i[6]) + "$ & " + \
                        str(lang_metric_i[12]) + " $\pm " + str(lang_metric_i[13]) + "$ & " + \
                        str(lang_metric_i[19]) + " $\pm " + str(lang_metric_i[20]) + "$ & " + \
                        str(lang_metric_i[26]) + " $\pm " + str(lang_metric_i[27]) + "$ & " + \
                        str(lang_metric_i[33]) + " $\pm " + str(lang_metric_i[34]) + "$ & " + \
                        str(lang_metric_i[40]) + " $\pm " + str(lang_metric_i[41]) + "$ "+repr("\\")

        if model == "adapters/FROZEN_BERT":
            intent_line += "\midrule"
            slot_line += "\midrule"

        intent_lines.append(intent_line)
        slot_lines.append(slot_line)

    intent_lines.append(repr("\r")+"owcolor{lightgray}  \multicolumn{7}{c}{Other Continuous Learning Algorithms }"+repr("\\")+" \midrule")
    slot_lines.append(repr("\r")+"owcolor{lightgray}  \multicolumn{7}{c}{Other Continuous Learning Algorithms }"+repr("\\")+" \midrule")
    for model in ["ewc_memsz-60000_type-ring_sample-random_k-60000_use-online_gamma-0.01",
                  "er_memsz-6000_type-reservoir_sample-random_k-16",
                  "kd-logits_memsz-6000_type-reservoir_sample-random_k-16",
                  "kd-rep_memsz-6000_type-reservoir_sample-random_k-16"]:
        lang_metric_i = lang_metrics_dict[model]
        if not fwt_mono:
            intent_line = repr("\t")+"extbf{" + models_dict[model] + "} & "+ \
                          str(lang_metric_i[2]) + " $\pm " + str(lang_metric_i[3]) + "$ & " + \
                          str(lang_metric_i[10]) + " $\pm " + str(lang_metric_i[11]) + "$ & " + \
                          str(lang_metric_i[18]) + " $\pm " + str(lang_metric_i[19]) + "$ & " + \
                          str(lang_metric_i[26]) + " $\pm " + str(lang_metric_i[27]) + "$ & " + \
                          str(lang_metric_i[34]) + " $\pm " + str(lang_metric_i[35]) + "$ & " + \
                          str(lang_metric_i[42]) + " $\pm " + str(lang_metric_i[43]) + "$ "+repr("\\")

            slot_line = repr("\t")+"extbf{" + models_dict[model] + "} & "+ \
                        str(lang_metric_i[5]) + " $\pm " + str(lang_metric_i[6]) + "$ & " + \
                        str(lang_metric_i[13]) + " $\pm " + str(lang_metric_i[14]) + "$ & " + \
                        str(lang_metric_i[21]) + " $\pm " + str(lang_metric_i[22]) + "$ & " + \
                        str(lang_metric_i[29]) + " $\pm " + str(lang_metric_i[30]) + "$ & " + \
                        str(lang_metric_i[37]) + " $\pm " + str(lang_metric_i[38]) + "$ & " + \
                        str(lang_metric_i[45]) + " $\pm " + str(lang_metric_i[46]) + "$ "+repr("\\")
        else:
            intent_line = repr("\t")+"extbf{" + models_dict[model] + "} & "+ \
                          str(lang_metric_i[2]) + " $\pm " + str(lang_metric_i[3]) + "$ & " + \
                          str(lang_metric_i[9]) + " $\pm " + str(lang_metric_i[10]) + "$ & " + \
                          str(lang_metric_i[16]) + " $\pm " + str(lang_metric_i[17]) + "$ & " + \
                          str(lang_metric_i[23]) + " $\pm " + str(lang_metric_i[24]) + "$ & " + \
                          str(lang_metric_i[30]) + " $\pm " + str(lang_metric_i[31]) + "$ & " + \
                          str(lang_metric_i[37]) + " $\pm " + str(lang_metric_i[38]) + "$ "+repr("\\")

            slot_line = repr("\t")+"extbf{" + models_dict[model] + "} & "+ \
                        str(lang_metric_i[5]) + " $\pm " + str(lang_metric_i[6]) + "$ & " + \
                        str(lang_metric_i[12]) + " $\pm " + str(lang_metric_i[13]) + "$ & " + \
                        str(lang_metric_i[19]) + " $\pm " + str(lang_metric_i[20]) + "$ & " + \
                        str(lang_metric_i[26]) + " $\pm " + str(lang_metric_i[27]) + "$ & " + \
                        str(lang_metric_i[33]) + " $\pm " + str(lang_metric_i[34]) + "$ & " + \
                        str(lang_metric_i[40]) + " $\pm " + str(lang_metric_i[41]) + "$ "+repr("\\")

        if model == "kd-rep_memsz-6000_type-reservoir_sample-random_k-16":
            intent_line += "\midrule"
            slot_line += "\midrule"

        intent_lines.append(intent_line)
        slot_lines.append(slot_line)

    for line in intent_lines:
        print(line.replace("'", ''))

    sep_line = "& \multicolumn{6}{c}{"+repr("\t")+"extbf{Test Slot Filling On}}"+repr("\\") + repr("\r") \
               +"owcolor{lightgray} \multicolumn{7}{c}{Shared \{Trans, Task\} Baselines } "+repr("\\")+"\midrule"
    print(sep_line.replace("'", ''))

    for line in slot_lines:
        print(line.replace("'", ''))


if __name__ == "__main__":
    languages = ["de", "en", "fr", "es", "hi", "th"]
    root_dir = "/Users/meryem/Desktop/x-continuous-learning-repro-backup/metrics/seeds/"

    """ MONO and MULTI Performances """
    with open(root_dir+"mono_orig.pickle", "rb") as file:
        mono_orig = pickle.load(file)
        mono_orig_perf = mono_orig["mono_orig_perf"]
        mono_orig_conf = mono_orig["mono_orig_conf"]

    with open(root_dir+"mono_ada_tuned.pickle", "rb") as file:
        mono_ada_tuned = pickle.load(file)
        mono_ada_tuned_perf = mono_ada_tuned["mono_ada_tuned_perf"]
        mono_ada_tuned_conf = mono_ada_tuned["mono_ada_tuned_conf"]

    with open(root_dir+"mono_ada_frozen.pickle", "rb") as file:
        mono_ada_frozen = pickle.load(file)
        mono_ada_frozen_perf = mono_ada_frozen["mono_ada_frozen_perf"]
        mono_ada_frozen_conf = mono_ada_frozen["mono_ada_frozen_conf"]

    with open(root_dir+"multi.pickle", "rb") as file:
        multi = pickle.load(file)
        multi_perf = multi["multi_perf"]
        multi_conf = multi["multi_conf"]

    #### AVG
    alias = "all_except_adaptersfrozen-multienc8"
    with open(root_dir+alias+"/all_avg_metrics_bootstrap.pickle", "rb") as file:
        all_avg_metrics = pickle.load(file)

    #### Per order
    with open(root_dir+alias+"/all_per_order_metrics_bootstrap.pickle", "rb") as file:
        all_per_order_metrics = pickle.load(file)

    # #### Per lang Forgetting
    # with open(root_dir+alias+"/all_forget_lang_strings_bootstrap.pickle", "rb") as file:
    #     all_forget_lang_metrics = pickle.load(file)
    #
    # #### Per lang FWT
    # with open(root_dir+alias+"/all_fwt_lang_strings_bootstrap.pickle", "rb") as file:
    #     all_fwt_lang_metrics = pickle.load(file)
    #
    # #### Per lang FWT mono
    # with open(root_dir+alias+"/all_fwt_mono_lang_strings_bootstrap.pickle", "rb") as file:
    #     all_fwt_mono_lang_metrics = pickle.load(file)

    mono_orig_line_intent = str(round(np.mean([mono_orig_perf[lang][0] for lang in languages]), 2)) + " $\pm " \
                           + str(round(np.mean([mono_orig_conf[lang][0] for lang in languages]), 2)) + "$"

    mono_orig_line_slot = str(round(np.mean([mono_orig_perf[lang][1] for lang in languages]), 2)) + " $\pm " \
                          + str(round(np.mean([mono_orig_conf[lang][1] for lang in languages]), 2)) + "$ "

    mono_ada_tuned_line_intent = str(round(np.mean([mono_ada_tuned_perf[lang][0] for lang in languages]), 2)) + " $\pm " \
                                + str(round(np.mean([mono_ada_tuned_conf[lang][0] for lang in languages]), 2)) + "$"

    mono_ada_tuned_line_slot = str(round(np.mean([mono_ada_tuned_perf[lang][1] for lang in languages]), 2)) + " $\pm " \
                              + str(round(np.mean([mono_ada_tuned_conf[lang][1] for lang in languages]), 2)) + "$ "

    mono_ada_frozen_line_intent = str(round(np.mean([mono_ada_frozen_perf[lang][0] for lang in languages]), 2)) + " $\pm " \
                                 + str(round(np.mean([mono_ada_frozen_conf[lang][0] for lang in languages]), 2)) + "$"

    mono_ada_frozen_line_slot = str(round(np.mean([mono_ada_frozen_perf[lang][1] for lang in languages]), 2)) + " $\pm " \
                                + str(round(np.mean([mono_ada_frozen_conf[lang][1] for lang in languages]), 2)) + "$ "

    multi_line_intent = str(round(np.mean([multi_perf[lang][0] for lang in languages]), 2)) + " $\pm " \
                       + str(round(np.mean([multi_conf[lang][0] for lang in languages]), 2)) + "$"

    multi_line_slot = str(round(np.mean([multi_perf[lang][1] for lang in languages]), 2)) + " $\pm " \
                      + str(round(np.mean([multi_conf[lang][1] for lang in languages]), 2)) + "$ "

    # table2_3(all_avg_metrics, mono_orig_line_intent, mono_orig_line_slot, multi_line_intent, multi_line_slot)
    table2(all_avg_metrics,
           mono_orig_line_intent,
           mono_orig_line_slot,
           multi_line_intent,
           multi_line_slot)
    print("------------------------------------------------------------------------------------------------")

    table3(all_avg_metrics)
    print("------------------------------------------------------------------------------------------------")

    table4(all_per_order_metrics)
    print("------------------------------------------------------------------------------------------------")

    exit(0)

    table5(root_dir)
    print("------------------------------------------------------------------------------------------------")

    table8(all_avg_metrics,
           mono_orig_line_intent,
           mono_orig_line_slot,
           mono_ada_tuned_line_intent,
           mono_ada_tuned_line_slot,
           mono_ada_frozen_line_intent,
           mono_ada_frozen_line_slot,
           multi_line_intent,
           multi_line_slot)
    print("------------------------------------------------------------------------------------------------")

    table9(all_avg_metrics,
           mono_orig_line_intent,
           mono_orig_line_slot)

    exit(0)
    print("------------------------------------------------------------------------------------------------")

    # table_10_11_12(all_per_order_metrics,
    #                mono_orig_line_intent,
    #                mono_orig_line_slot,
    #                mono_ada_tuned_line_intent,
    #                mono_ada_tuned_line_slot,
    #                mono_ada_frozen_line_intent,
    #                mono_ada_frozen_line_slot,
    #                multi_line_intent,
    #                multi_line_slot,
    #                "en_de_fr_hi_es_th",
    #                "th_es_hi_fr_de_en")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    #
    # table_10_11_12(all_per_order_metrics,
    #                mono_orig_line_intent,
    #                mono_orig_line_slot,
    #                mono_ada_tuned_line_intent,
    #                mono_ada_tuned_line_slot,
    #                mono_ada_frozen_line_intent,
    #                mono_ada_frozen_line_slot,
    #                multi_line_intent,
    #                multi_line_slot,
    #                "es_hi_en_de_th_fr",
    #                "fr_th_de_en_hi_es")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    # print("------------------------------------------------------------------------------------------------")
    #
    # table_10_11_12(all_per_order_metrics,
    #                mono_orig_line_intent,
    #                mono_orig_line_slot,
    #                mono_ada_tuned_line_intent,
    #                mono_ada_tuned_line_slot,
    #                mono_ada_frozen_line_intent,
    #                mono_ada_frozen_line_slot,
    #                multi_line_intent,
    #                multi_line_slot,
    #                "hi_en_es_th_fr_de",
    #                "de_fr_th_es_en_hi")
    # print("------------------------------------------------------------------------------------------------")
    # print("FORGETTING")
    # table_14_15_16(all_forget_lang_metrics, all_avg_metrics, fwt_mono=False)
    print("------------------------------------------------------------------------------------------------")

    print("TRANSFER")
    table_14_15_16(all_fwt_mono_lang_metrics, all_avg_metrics, fwt_mono=True)

    print("------------------------------------------------------------------------------------------------")
    # print("ZERO-SHOT GENERALIZATION")
    # table_14_15_16(all_fwt_lang_metrics, all_avg_metrics, fwt_mono=False)
