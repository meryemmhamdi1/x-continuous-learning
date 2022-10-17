import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from bioinfokit.analys import stat
from plot_bar_charts import read_all_lang_order_results, read_all_lang_order_results_seeds, model_names_dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_data(df):
    # load data file
    # reshape the d dataframe suitable for statsmodels package
    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['A', 'B', 'C', 'D'])
    # replace column names

    # generate a boxplot to see the data distribution by treatments. Using boxplot, we can
    # easily detect the differences between different treatments
    ax = sns.boxplot(x='treatments', y='value', data=df_melt, color='#99c2a2')
    ax = sns.swarmplot(x="treatments", y="value", data=df_melt, color='#7d0013')
    plt.show()


def compute_sig_test(df, models):
    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    # methods_df = []
    # for method in list(df.columns):
    #     methods_df.append(df[method])
    fvalue, pvalue = stats.f_oneway(df['Naive Seq FT'],
                                    df['Inc Joint'],
                                    # df['Lang-Spec Enc[0-8]'],
                                    df['Lang-Spec Trans'],
                                    df['Lang-Spec Ada(T)'],
                                    # df['Lang-Spec Ada(F)'],
                                    df['EWC-Online'],
                                    df['ER'],
                                    df['KD-Logit'],
                                    df['KD-Rep'])

    print(fvalue, pvalue)
    # perform multiple pairwise comparison (Tukey's HSD)
    # unequal sample size data, tukey_hsd uses Tukey-Kramer test
    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=models)
    df_melt.columns = ['index', 'treatments', 'value']
    res = stat()
    res.tukey_hsd(df=df_melt, res_var='value', xfac_var='treatments', anova_model='value ~ C(treatments)')
    # print(res.tukey_summary[['group1', 'group2', 'p-value']])

    results_all = res.tukey_summary[res.tukey_summary['p-value'] > 0][['group1', 'group2', 'p-value']]

    group1_list = list(results_all['group1'])
    group2_list = list(results_all['group2'])
    pvalue_list = list(results_all['p-value'])

    # conf_matrix = {model: {model: 0.0} for model in models}
    conf_matrix = [len(models)*[0.0] for _ in models]
    for group1, group2, pvalue in zip(group1_list, group2_list, pvalue_list):
        conf_matrix[models.index(group2)][models.index(group1)] = float(pvalue)

    return conf_matrix
    # print(list(results_all['group1']))
    # print(list(results_all['group2']))
    # print(list(results_all['p-value']))

    # conf_matrix[results_all['group1']][results_all['group2']] = results_all['p-value']]

plot_save_path = "Plots/significancetesting/"

def plot_conf_matrix(conf_matrix, models, name):
    # array = [[33,2,0,0,0,0,0,0,0,1,3],
    #          [3,31,0,0,0,0,0,0,0,0,0],
    #          [0,4,41,0,0,0,0,0,0,0,1],
    #          [0,1,0,30,0,6,0,0,0,0,1],
    #          [0,0,0,0,38,10,0,0,0,0,0],
    #          [0,0,0,3,1,39,0,0,0,0,4],
    #          [0,2,2,0,4,1,31,0,0,0,2],
    #          [0,1,0,0,0,0,0,36,0,2,0],
    #          [0,0,0,0,0,0,1,5,37,5,1],
    #          [3,0,0,0,0,0,0,0,0,39,0],
    #          [0,0,0,0,0,0,0,0,0,0,38]]
    df_cm = pd.DataFrame(conf_matrix, index=models,
                         columns=models)

    dropSelf = np.zeros_like(conf_matrix)
    dropSelf[np.triu_indices_from(dropSelf)] = True
    plt.subplots(figsize=(11, 8))
    # plt.imshow(A2, cmap='Blues', interpolation='nearest')
    sns.heatmap(df_cm, mask=(df_cm <= 0), annot=True, cmap="Blues",
                linewidths=2, vmax=1, vmin=0, square=True, cbar_kws={"shrink": .5})#, mask=dropSelf) (df_cm > 0.05) |

    plt.xticks(rotation=90, size=13)
    plt.yticks(rotation=0, size=13)
    plt.subplots_adjust(bottom=0.27, top=1, left=0.26)
    # plt.show()
    filename = plot_save_path+name

    plt.savefig(filename+".png", format="png")

import pickle
if __name__ == "__main__":
    # df = pd.read_csv("https://reneshbedre.github.io/assets/posts/anova/onewayanova.txt", sep="\t")
    # plot_data(df)
    # intents, slots = read_all_lang_order_results()
    intents, slots = read_all_lang_order_results_seeds()

    print("intents:", intents["vanilla"]["forgetting"])
    print("intents:", intents["er_memsz-6000_type-reservoir_sample-random_k-16"]["forgetting"])
    print("intents:", intents["vanilla"]["fwt"])
    print("intents:", intents["vanilla"]["fp"])

    forgetting_intent_df = pd.DataFrame()
    forgetting_slot_df = pd.DataFrame()

    fwt_intent_df = pd.DataFrame()
    fwt_slot_df = pd.DataFrame()

    fwt_0_intent_df = pd.DataFrame()
    fwt_0_slot_df = pd.DataFrame()

    fp_intent_df = pd.DataFrame()
    fp_slot_df = pd.DataFrame()

    models = []
    for model in intents:
        model_s = model_names_dict[model]
        print("model_s:", model_s)
        if model_s not in ["Lang-Spec Enc[0-8]", "Lang-Spec Ada(F)"] :
            print("[el[1] for el in intents[model]['forgetting']]:", [el[1] for el in intents[model]["forgetting"]])
            forgetting_intent_df[model_s] = [el[1] for el in intents[model]["forgetting"]]
            forgetting_slot_df[model_s] = [el[1] for el in slots[model]["forgetting"]]

            fwt_intent_df[model_s] = [el[1] for el in intents[model]["fwt"]]
            fwt_slot_df[model_s] = [el[1] for el in slots[model]["fwt"]]

            fwt_0_intent_df[model_s] = [el[1] for el in intents[model]["fwt_0"]]
            fwt_0_slot_df[model_s] = [el[1] for el in slots[model]["fwt_0"]]

            fp_intent_df[model_s] = [el[1] for el in intents[model]["fp"]]
            fp_slot_df[model_s] = [el[1] for el in slots[model]["fp"]]

            models.append(model_s)

    conf_matrix = compute_sig_test(forgetting_intent_df, models)
    plot_conf_matrix(conf_matrix, models, name="IntentForget")

    conf_matrix = compute_sig_test(forgetting_slot_df, models)
    plot_conf_matrix(conf_matrix, models, name="SlotForget")

    conf_matrix = compute_sig_test(fwt_intent_df, models)
    plot_conf_matrix(conf_matrix, models, name="FWTIntent")

    conf_matrix = compute_sig_test(fwt_slot_df, models)
    plot_conf_matrix(conf_matrix, models, name="FWTSlot")

    conf_matrix = compute_sig_test(fp_intent_df, models)
    plot_conf_matrix(conf_matrix, models, name="FPIntent")

    conf_matrix = compute_sig_test(fp_slot_df, models)
    plot_conf_matrix(conf_matrix, models, name="FPSlot")

    #

    conf_matrix = compute_sig_test(fwt_0_intent_df, models)
    plot_conf_matrix(conf_matrix, models, name="FWT0Intent")

    conf_matrix = compute_sig_test(fwt_0_slot_df, models)
    plot_conf_matrix(conf_matrix, models, name="FWT0Slot")
