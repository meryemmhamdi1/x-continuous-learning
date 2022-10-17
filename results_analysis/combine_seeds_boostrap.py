import pickle
import math

def compute_a_b(std1, std2):

    a = (1/std1) / ((1/std1)+(1/std2))
    b = (1/std2) / ((1/std1)+(1/std2))
    return a, b

def combine_weighted_avg_conf(mean_1, mean_2, std_1, std_2):
    # Taken from https://www.reddit.com/r/AskStatistics/comments/nptl5z/how_to_combine_two_confidence_intervals/h07vvyw/?context=3
    a, b = compute_a_b(std_1, std_2)
    new_mean = a * mean_1 + b * mean_2
    new_conf = math.sqrt((a*std_1)**2 + (b*std_2)**2)

    return new_mean, new_conf

if __name__ == "__main__":
    out_dir = "metrics/weighted/"
    model = "mono_orig"
    languages = ["de", "en", "fr", "es", "hi", "th"]
    ## Load metrics for all language orders, languages, and over the average from seeds
    in_dir = "metrics/seeds/all_except_ewc-joint-kdrep-vanilla-adapters/"
    with open(in_dir+"all_avg_metrics_bootstrap.pickle", "rb") as file:
        seeds_data = pickle.load(file)

    print("SEED:", seeds_data)

    print("------------------------------------------------------------------------------")

    ## Load metrics for all language orders, languages, and over the average from bootstrap
    in_dir = "metrics/bootstrap/multi_head_out/"
    with open(in_dir + "all_avg_metrics_bootstrap.pickle", "rb") as file:
        boot_data = pickle.load(file)

    print("BOOTSTRAP:", boot_data)

    exit(0)

    ## Get averages
    new_data = {model+"_perf": {lang: [0.0, 0.0] for lang in languages}, model+"_conf": {lang: [0.0, 0.0] for lang in languages}}
    for lang in languages:
        for i in range(2):
            mean_1, mean_2, std_1, std_2 = seeds_data[model+"_perf"][lang][i], boot_data[model+"_perf"][lang][i], \
                                           seeds_data[model+"_perf"][lang][i], boot_data[model+"_conf"][lang][i]

            new_data[model+"_perf"][lang][i], new_data[model+"_conf"][lang][i] \
                = combine_weighted_avg_conf(mean_1, mean_2, std_1, std_2)

    print("new_data:", new_data)