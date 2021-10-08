from transformers import BertTokenizer, BertModel, BertConfig, \
                         XLNetTokenizer, XLNetModel, XLNetConfig, \
                         XLMTokenizer, XLMModel, XLMConfig, \
                         DistilBertTokenizer, DistilBertModel, DistilBertConfig, \
                         RobertaTokenizer, RobertaModel, RobertaConfig, \
                         XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig, \
                         AlbertTokenizer, AlbertModel, AlbertConfig

MODELS_dict = {"BertLarge": ('bert-large-uncased', BertTokenizer, BertModel, BertConfig),
               "BertBaseCased": ('bert-base-cased', BertTokenizer, BertModel, BertConfig),
               "BertBaseMultilingualCased": ('bert-base-multilingual-cased', BertTokenizer, BertModel, BertConfig),
               "Xlnet_base": ('xlnet-base-cased', XLNetTokenizer, XLNetModel, XLNetConfig),
               "Xlnet_large": ('xlnet-large-cased', XLNetTokenizer, XLNetModel, XLNetConfig),
               "XLM": ('xlm-mlm-enfr-1024', XLMTokenizer, XLMModel, XLMConfig),
               "DistilBert_base": ('distilbert-base-uncased', DistilBertTokenizer, DistilBertModel, DistilBertConfig),
               "DistilBert_large": ('distilbert-large-cased', DistilBertTokenizer, DistilBertModel, DistilBertConfig),
               "Roberta_base": ('roberta-base', RobertaTokenizer, RobertaModel, RobertaConfig),
               "Roberta_large": ('roberta-large', RobertaTokenizer, RobertaModel, RobertaConfig),
               "XLMRoberta_base": ('xlm-roberta-base', XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig),
               "XLMRoberta_large": ('xlm-roberta-large', XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig),
               "ALBERT-base-v1": ('albert-base-v1', AlbertTokenizer, AlbertModel, AlbertConfig),
               "ALBERT-large-v1": ('albert-large-v1', AlbertTokenizer, AlbertModel, AlbertConfig),
               "ALBERT-xlarge-v1": ('albert-xlarge-v1', AlbertTokenizer, AlbertModel, AlbertConfig),
               "ALBERT-xxlarge-v1": ('albert-xxlarge-v1', AlbertTokenizer, AlbertModel, AlbertConfig),
               "ALBERT-base-v2": ( 'albert-base-v2', AlbertTokenizer, AlbertModel, AlbertConfig),
               "ALBERT-large-v2": ('albert-large-v2', AlbertTokenizer, AlbertModel, AlbertConfig),
               "ALBERT-xlarge-v2": ('albert-xlarge-v2', AlbertTokenizer, AlbertModel, AlbertConfig),
               "ALBERT-xxlarge-v2": ('albert-xxlarge-v2', AlbertTokenizer, AlbertModel, AlbertConfig),
               }
