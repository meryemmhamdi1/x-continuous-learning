import logging
from consts import NLI_TYPES
import json, os, copy
from torch import LongTensor
from processors.utils import clean_text
from processors.utils import _read_tsv
from transformers import XLMTokenizer

logger = logging.getLogger(__name__)

class NLIExample(object):
  """
  A single training/test example for simple sequence classification.
  Args:
    unique_id: Unique id for the example.
    text_a: string. The untokenized text of the first sequence. For single
    sequence tasks, only this sequence must be specified.
    text_b: (Optional) string. The untokenized text of the second sequence.
    Only must be specified for sequence pair tasks.
    label: (Optional) string. The label of the example. This should be
    specified for train and dev examples, but not for test examples.
  """

  def __init__(self, unique_id, text_a, text_b=None, label=None, language=None, input_ids=None, attention_mask=None, token_type_ids=None):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.language = language
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids

  def __repr__(self):
    return str(self.to_json_string())

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def _parse_xnli(data_path,
                tokenizer,
                args,
                split,
                lang):

    process_egs = []
    process_egs_dict = {}
    lines = _read_tsv(data_path)
    for i, line in enumerate(lines):
        if i == 0:
            continue

        text_a = line[1]
        text_b = line[2]
        label = line[3]

        if isinstance(tokenizer, XLMTokenizer):
            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=args.xnli_max_length, lang=lang)
        else:
            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=args.xnli_max_length)

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        attention_mask = [1 if args.mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.xnli_max_length - len(input_ids)

        input_ids = input_ids + ([args.pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if args.mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([args.pad_token_segment_id] * padding_length)

        assert len(input_ids) == args.xnli_max_length, "Error with input length {} vs {}".format(len(input_ids), args.xnli_max_length)
        assert len(attention_mask) == args.xnli_max_length, "Error with input length {} vs {}".format(
            len(attention_mask), args.xnli_max_length)
        assert len(token_type_ids) == args.xnli_max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), args.xnli_max_length)

        id_ = split+"_"+lang+"_"+str(i)
        example = NLIExample(unique_id=id_, 
                             text_a=text_a, 
                             text_b=text_b, 
                             label=label, 
                             language=lang, 
                             input_ids=input_ids, 
                             attention_mask=attention_mask, 
                             token_type_ids=token_type_ids)
        process_egs.append(example)
        process_egs_dict.update({id_: example})

    return process_egs, process_egs_dict

class Processor(object):
    """
    Processor of natural language inference
    """

    def __init__(self, args, tokenizer):
        self.data_name = args.data_name
        self.data_format = args.data_format
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.class_types = NLI_TYPES

    def read_split(self, lang, split_name):
        file_path = os.path.join(os.path.join(self.data_root, lang), split_name)


        process_egs, process_egs_dict = _parse_xnli(file_path + "." + self.data_format,
                                                    self.tokenizer,
                                                    self.args,
                                                    split_name,
                                                    lang)

        return process_egs, process_egs_dict


    def next_batch(self, batch_size, data_split=None, identifier=None):
        if identifier:
            examples = [self.get_item_by_id(identifier)]
        else:
            examples = data_split.next_items(batch_size)

        # max_sent_len = 0
        input_ids, lengths, labels, token_type_ids, input_masks, \
            input_texts_a, input_texts_b = [], [], [], [], [], [], []


        for example in examples:
            input_texts_a.append(example.text_a)
            input_texts_b.append(example.text_b)
            labels.append(example.label)
            cur_input_ids = example.input_ids
            input_ids.append(cur_input_ids)

            cur_length = len(cur_input_ids)
            # max_sent_len = max(max_sent_len, cur_length)
            lengths.append(cur_length)
            input_masks.append(example.attention_mask)
            token_type_ids.append(example.token_type_ids)


        # Padding
        # for i in range(batch_size):
        #     # input_masks.append([1] * len(input_ids[i]) + [0] * (max_sent_len - len(input_ids[i])))
        #     # input_ids[i] += [0] * (max_sent_len - len(input_ids[i]))

        # Convert to LongTensors
        input_ids = LongTensor(input_ids)
        lengths = LongTensor(lengths)
        labels = LongTensor(labels)
        token_type_ids = LongTensor(token_type_ids)
        input_masks = LongTensor(input_masks)

        # Batch tensors, text tuples (input_texts_a, input_texts_b), identifiers (input_identifiers), examples
        return {"input_ids": input_ids, "input_masks": input_masks, "token_type_ids": token_type_ids, "labels": labels},  examples
