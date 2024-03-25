import pickle, random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from summarize_metrics import acc_avg, fwt_avg, fwt_avg_mono, bwt_avg, forget_avg, final_perf
import matplotlib.pyplot as plt

intents_results_dict_en = {"er_both":[67.58, 66.64, 66.34, 67.76, 67.14, 66.94], 
                           "er_only": [68.08, 67.78, 66.98, 67.38, 66.80, 67.34],
                           "er_main": [67.72, 67.64, 65.28, 67.54, 67.16, 67.00]}

intents_results_dict_vi = {"er_both":[63.35, 62.67, 60.41, 64.88, 62.23, 64.34], 
                           "er_only": [64.20, 64.82, 62.89, 63.96, 63.75, 64.52],
                           "er_main": [65.16, 64.06, 59.93, 64.48, 62.89, 62.15]}



df = pd.DataFrame(intents_results_dict_en)

#plot individual lines
plt.clf()
for model_name in intents_results_dict_en:
    plt.plot(df[model_name], label=model_name)

plt.xlabel("# Epochs")
plt.ylabel("Accuracy Score")
# plt.title('Dev on '+languages[0]+' after training on '+lang + ' in language_order: '+lang_order)
plt.title('Test after training on '+"vi" + ' in language_order: '+"en_vi_ar_tr_bg_el_ur")
plt.legend(loc="lower right")

# plot_save_dir = "/project/jonmay_231/meryem/SpacedRepetitionFigures/XNLI/cont-new"
plot_save_dir = ""
#display plot
plt.savefig(plot_save_dir+'ERclassscore_en_aftertrain-vi_dev-perf-epoch-curves.png')


df = pd.DataFrame(intents_results_dict_vi)

#plot individual lines
plt.clf()
for model_name in intents_results_dict_vi:
    plt.plot(df[model_name], label=model_name)

plt.xlabel("# Epochs")
plt.ylabel("Accuracy Score")
# plt.title('Dev on '+languages[0]+' after training on '+lang + ' in language_order: '+lang_order)
plt.title('Test after training on '+"vi" + ' in language_order: '+"en_vi_ar_tr_bg_el_ur")
plt.legend(loc="lower right")

# plot_save_dir = "/project/jonmay_231/meryem/SpacedRepetitionFigures/XNLI/cont-new/"
#display plot
plt.savefig(plot_save_dir+'ERclassscore_vi_aftertrain-vi_dev-perf-epoch-curves.png')