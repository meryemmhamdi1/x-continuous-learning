# Cross-lingual Continuous Learning:

This is a Pytorch implementation to reproduce reproduce results for our cross-lingual continuous fine-tuning benchmark. This serves as testbed to systematically study the performance of different approaches to continuous learning across different metrics to investigate forward and backward transfer trends, as well as narrow down which components and orders are responsible for catastrophic forgetting. All this to help with coming up with a recipe for deploying continuous learning cross-lingually.

N.B.: The repository is still being organized/refactored. If you have questions or comments, feel free to send an email to mmhamdi at usc.edu.

## Table of Contents:

1. [Abstract](#abstract)
2. [Requirements](#requirements)
3. [Continuous Data Streams](#datastreams)
4. [Running Scripts](#scripts)
5. [Results Visualization](#results)
6. [Citation](#citation)
7. [Credits](#credits)

## 1. Abstract <a name="abstract"></a>:

In this paper, we present the cross-lingual continuous learning challenge, where a model is continually fine-tuned so as to adapt to emerging data from different languages. Over continuous streams of training examples from different languages, we incrementally fine-tune NLU model and extensively analyze the performance of the model in terms of its forward and backward transfer. We compare the changes in performance throughout the stream with respect to different balanced orderings, languages, and over two state of the art transformer encoders M-BERT and XLM-Roberta. We also perform further analysis to shed the light on the impact of resourcefulness and language similarity separately by carefully curating adequate data streams for that.

Our experimental analysis shows that continual learning algorithms, namely model expansion approaches, improve over naive fine-tuning and regularization-based approaches narrowing the gap with joint learning baselines. We perform extensive ablation studies to shed the light on which components are responsible for catastrophic forgetting and need further attention when it comes to continual learning. Based on that, we show that the encoder module is the culprit of catastrophic forgetting and dig deeper to check which algorithm help with with addressing that. We believe our problem setup, evaluation protocols, and analysis can inspire future studies towards continual learning for more or across downstream tasks covering different languages.

![image](xcontlearndiagram.pdf)

## 2. Requirements <a name="requirements"></a>:
* Python 3.6 or higher.
* Run pip install -r requirements.txt to install all required packages.

## 3. Continuous Data Streams <a name="datasets"></a>:
* The original MTOP dataset can be directly downloaded from https://fb.me/mtop_dataset. For more details regarding the processing of the dataset, refer to functions in data_utils.py.
* For the high-level analysis, we use the dataset splits as is. This is referred to as CLL with fixed label set where the stream consists of data from different languages preserving the original MTOP label space distribution following a specific permutation of languages.
* For further analysis, we provide the data streams used for approximating the impact of resourcefulness and language similarity as follows:
  - The impact of resourcefulness: For this purpose, we design different N-way and K-Shot data stream splits:
      - N-Way Analysis: We gradually increase or decrease the coverage of classes (with the same number of shots per class) across one single language. [LINK HERE either gdrive or within repo]
      - K-Shot Analysis: We gradually increase or decrease the number of shots per class where the classes are fixed.
  - The impact of resourcefulness: We come up with a datastream with an equal coverage over the different classes per language. [LINK HERE either gdrive or within repo]

## 4. Running Scripts <a name="scripts"></a>:

Please refer to scripts folder for bash script (train_trans_nlu_generic.sh) to run different approaches and options and the hyperparameters (hyperparam.ini) used throughout all experiments.

After providing paths.ini where you include DATA_ROOT and TRANS_MODEL and OUT_DIR paths under ENDEAVOUR attribute, run sh scripts/train_trans_nlu_generic.sh

[TODO] add some examples of scripts

## 5. Results Visualization <a name="results"></a>:

COMING SOON

## 6. Citation <a name="citation"></a>:

If you find our work or code useful, please cite our paper:

<pre>
@misc{https://doi.org/10.48550/arxiv.2205.11152,
      doi = {10.48550/ARXIV.2205.11152},
      url = {https://arxiv.org/abs/2205.11152},
      author = {M'hamdi, Meryem and Ren, Xiang and May, Jonathan},
      keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Cross-lingual Lifelong Learning},
      publisher = {arXiv},
      year = {2022},
      abstract="The longstanding goal of multi-lingual learning has been to develop a universal cross-lingual model that can withstand the changes in multi-lingual data distributions. However, most existing models assume full access to the target languages in advance, whereas in realistic scenarios this is not often the case, as new languages can be incorporated later on. In this paper, we present the Cross-lingual Lifelong Learning (CLL) challenge, where a model is continually fine-tuned to adapt to emerging data from different languages. We provide insights into what makes multilingual sequential learning particularly challenging. To surmount such challenges, we benchmark a representative set of cross-lingual continual learning algorithms and analyze their knowledge preservation, accumulation, and generalization capabilities compared to baselines on carefully curated datastreams. The implications of this analysis include a recipe for how to measure and balance between different cross-lingual continual learning desiderata, which goes beyond conventional transfer learning."
      copyright = {Creative Commons Attribution 4.0 International},
     }
</pre>



## 7. Credits <a name="credits"></a>:

The code in this repository is partially based on [meta_cross_nlu_qa](https://github.com/meryemmhamdi1/meta_cross_nlu_qa) for the data preprocessing and downstream models, on [adapter-hub](https://github.com/Adapter-Hub/adapter-transformers) for the adapter-based approaches, [three-scenarios-for-cont-learn](https://github.com/GMvandeVen/continual-learning) for some algorithms.
