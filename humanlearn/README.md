# Leitner-Guided Memory Replay for Cross-lingual Continual Learning:

This is a Pytorch implementation to reproduce results for our leitner-guided memory replay cross-lingual continual learning approach. This folder is an endpoint for running different experiments along with qualitative analysis. The main source code for base models, continual learning approaches and leitner queues schedulers can be found under [src](https://github.com/meryemmhamdi1/x-continuous-learning/tree/main/src)

N.B.: The repository is still being organized/refactored. If you have questions or comments, feel free to send an email to mmhamdi at usc.edu or meryemmhamdi1 at gmail.com.


## Table of Contents:

1. [Abstract](#abstract)
2. [Requirements](#requirements)
3. [Running Scripts](#scripts)
4. [Citation](#citation)

## 1. Abstract <a name="abstract"></a>:
Cross-lingual continual learning aims to continuously fine-tune a downstream model on emerging data from new languages. One major challenge in cross-lingual continual learning is catastrophic forgetting: a stability-plasticity dilemma, where performance on previously seen languages decreases as the model learns to transfer to new languages. Experience replay, which revisits data from a fixed-size memory of old languages while training on new ones, is among the most successful approaches for solving this dilemma. Faced with the challenge of dynamically storing the memory with high-quality examples while complying with its fixed size limitations, we consider Leitner queuing, a human-inspired spaced-repetition technique, to determine what should be replayed at each phase of learning. Via a controlled set of quantitative and qualitative analyses across different memory strategies, we show that, just like humans, carefully picking informative examples to be prioritized in cross-lingual memory replay helps tame the plasticity-stability dilemma. Compared to vanilla and strong memory replay baselines, our Leitner-guided approach significantly and consistently decreases forgetting while maintaining accuracy across natural language understanding tasks, language orders, and languages.

## 2. Requirements <a name="requirements"></a>:
* Refer to [requirements.txt](https://github.com/meryemmhamdi1/x-continuous-learning/blob/main/requirements.txt) for a list of required packages to install.

## 3. Running Scripts <a name="scripts"></a>:

Please refer to scripts folder for bash script [main.sh](scripts/main.sh) to run different approaches and options.
The hyperparameters used for each base model can be found under [hyperparameters](https://github.com/meryemmhamdi1/x-continuous-learning/tree/main/src/scripts) folder in scripts in root directory. 

After adding a paths.ini where you include DATA_ROOT and TRANS_MODEL and OUT_DIR paths under ENDEAVOUR attribute, run different scripts prefixed with "run" with different command line options.

## 4. Citation <a name="citation"></a>:

If you find our work or code useful, please cite our paper (just accepted to NAACL 2024):

CITATION TO BE PROVIDED SOON
<pre>
@misc{
      author = {M'hamdi, Meryem and May, Jonathan},
      title = {Leitner-Guided Memory Replay for Cross-lingual Continual Learning},
      publisher = {To appear in Proceedings of 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)}
      year = {2024},
     }
</pre>