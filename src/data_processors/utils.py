import re
import csv

# detect pattern
# detect <TIME>
pattern_time1 = re.compile(r"[0-9]+[ap]")
pattern_time2 = re.compile(r"[0-9]+[;.h][0-9]+")
pattern_time3 = re.compile(r"[ap][.][am]")
pattern_time4 = range(2000, 2020)
# pattern_time5: token.isdigit() and len(token) == 3

pattern_time_th1 = re.compile(r"[\u0E00-\u0E7F]+[0-9]+")
pattern_time_th2 = re.compile(r"[0-9]+[.]*[0-9]*[\u0E00-\u0E7F]+")
pattern_time_th3 = re.compile(r"[0-9]+[.][0-9]+")

# detect <LAST>
pattern_last1 = re.compile(r"[0-9]+min")
pattern_last2 = re.compile(r"[0-9]+h")
pattern_last3 = re.compile(r"[0-9]+sec")

# detect <DATE>
pattern_date1 = re.compile(r"[0-9]+st")
pattern_date2 = re.compile(r"[0-9]+nd")
pattern_date3 = re.compile(r"[0-9]+rd")
pattern_date4 = re.compile(r"[0-9]+th")

remove_list = ["'s", "'ll", "'ve", "'d", "'m"]


def clean_text(token_list, lang):
    """
    Applying the same pre-processing on NLU as in the latest AAAI 2020 publication
    taken from https://github.com/zliucr/mixed-language-training
    :param token_list:
    :param lang:
    :return:
    """
    token_list_clean = []
    for token in token_list:
        new_token = token
        # detect <TIME>
        if lang != "th" and (
            bool(re.match(pattern_time1, token))
            or bool(re.match(pattern_time2, token))
            or bool(re.match(pattern_time3, token))
            or token in pattern_time4
            or (token.isdigit() and len(token) == 3)
        ):
            new_token = "<TIME>"
            token_list_clean.append(new_token)
            continue
        if lang == "th" and (
            bool(re.match(pattern_time_th1, token))
            or bool(re.match(pattern_time_th2, token))
            or bool(re.match(pattern_time_th3, token))
        ):
            new_token = "<TIME>"
            token_list_clean.append(new_token)
            continue
        # detect <LAST>
        if lang == "en" and (
            bool(re.match(pattern_last1, token))
            or bool(re.match(pattern_last2, token))
            or bool(re.match(pattern_last3, token))
        ):
            new_token = "<LAST>"
            token_list_clean.append(new_token)
            continue
        # detect <DATE>
        if lang == "en" and (
            bool(re.match(pattern_date1, token))
            or bool(re.match(pattern_date2, token))
            or bool(re.match(pattern_date3, token))
            or bool(re.match(pattern_date4, token))
        ):
            new_token = "<DATE>"
            token_list_clean.append(new_token)
            continue
        # detect <LOCATION>
        if lang != "th" and (token.isdigit() and len(token) == 5):
            new_token = "<LOCATION>"
            token_list_clean.append(new_token)
            continue
        # detect <NUMBER>
        if token.isdigit():
            new_token = "<NUMBER>"
            token_list_clean.append(new_token)
            continue
        if lang == "en" and ("n't" in token):
            new_token = "not"
            token_list_clean.append(new_token)
            continue
        if lang == "en":
            for item in remove_list:
                if item in token:
                    new_token = token.replace(item, "")
                    break

        token_list_clean.append(new_token)

    assert len(token_list_clean) == len(token_list)

    return token_list_clean


def convert2ids(tokenizer, tokens):
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_masks = [1] * len(input_ids)
    token_type_ids = [1] * len(input_ids)

    return input_ids, attention_masks, token_type_ids


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _improve_answer_span(
    doc_tokens,
    input_start,
    input_end,
    tokenizer,
    orig_answer_text,
    lang="en",
    lang2id=None,
):
    """Returns tokenized answer spans that better match the annotated answer."""
    if lang2id is None:
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    else:
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text, lang=lang))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
