from data_utils import _parse_mtop_simplified


if __name__ == "__main__":
    data_path = "/Users/d22admin/USCGDrive/Spring21/Research/XContLearn/Datasets/NLU/mtop-v1/"
    intent_set_total = []
    slot_set_total = []
    for lang in ["en", "de", "fr", "es", "hi", "th"]:
        total = 0
        for split in ["train", "eval", "test"]:
            process_egs, intent_set, slot_set = \
                _parse_mtop_simplified(data_path + lang + "/" + split + ".txt")

            intent_set_total.extend([intent for intent in intent_set if intent not in intent_set_total])
            slot_set_total.extend([slot for slot in slot_set if slot not in slot_set_total])

            total += len(process_egs)

            print("lang: ", lang, " split: ", split, " len(intent_set): ", len(intent_set),
                  " len(slot_set): ", len(slot_set))

        print("total: ", total)
        print("-------------------------------------------")

    domains = {}
    for intent in sorted(intent_set_total):
        domain = intent.split(":")[0]
        if domain not in domains:
            domains.update({domain: []})
        domains[domain].append(intent)
        print("\"" + intent + "\",")
    print("len(intent_set_total):", len(intent_set_total))

    for domain in domains:
        print("domain:", domain, " len(domains[domain]):", len(domains[domain]))
    print("-------------------------------")
    print("O")
    print("X")
    for slot in sorted(slot_set_total):
        print("\"B-" + slot + "\",")
        print("\"I-" + slot + "\",")

    print("len(slot_set_total):", len(slot_set_total))
