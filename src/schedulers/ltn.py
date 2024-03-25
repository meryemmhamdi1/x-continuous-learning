from tqdm import tqdm
import random, math


class Element(object):
    def __init__(self, id, current_queue=0):
        self.id = id
        self.movements = []
        self.current_queue = current_queue

    def add_movement(self, queue_num):
        self.movements.append(queue_num)

    def set_current_queue(self, queue_num):
        self.current_queue = queue_num

    def stats_deck(self, deck_num):
        return self.movements.count(deck_num)

    def repr(self):
        return self.id


class LeitnerQueue(object):
    def __init__(self, args):
        """
        LeitnerQueues using Static Scheduler:
            * Deck View:
                - Deck 0: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                - Deck 1: 2, 4, 6, 8, 10
                - Deck 2: 4, 8
                - Deck 3: 8
                - Deck 4: none => convergence

            * Epoch view:
                - Epoch 1: deck 0
                - Epoch 2: deck 0, deck 1
                - Epoch 3: deck 0
                - Epoch 4: deck 0, deck 1, deck 2
                - Epoch 5: deck 0
                - Epoch 6: deck 0, deck 1
                - Epoch 7: deck 0
                - Epoch 8: deck 0, deck 1, deck 2, deck 3
                - Epoch 9: deck 0
                - Epoch 10: deck 0

        Initializes some useful data structures to keep track of the contents of the different decks and their movements
        """

        self.args = args
        self.num_decks = args.num_decks
        self.demote_to_first = args.demote_to_first_deck
        self.lt_sampling_mode = args.lt_sampling_mode
        self.er_strategy = args.er_strategy
        self.er_strategy_prop = args.er_strategy_prop
        self.thresh = args.ltn_promote_thresh
        self.max_mem_sz = args.max_mem_sz
        self.languages_len = len(args.order_lst.split("_"))
        self.num_er_langs = self.languages_len - 1

        self.decks = [[] for _ in range(self.num_decks)]
        self.deck_of_items = {}
        self.all_items = {}
        self.ids_movements = {}

    def init_first_deck(self, dataset, train_examples, nb_examples):
        """
        Initializes the Leitner Queues by placing everything in the first queue
        :param dataset: dataset instance of MultiPurposeDataset class
        :param train_examples: examples instance of AugmentedList
            Training examples used to initialize the first deck
        :param nb_examples: int
            The number of examples used to initialize the first deck
        """
        _, examples, _ = dataset.next_batch(
            dataset=dataset, batch_size=nb_examples, data_split=train_examples
        )

        input_identifiers = [example.unique_id for example in examples]

        self.decks[0].extend(input_identifiers)
        for i, id_ in enumerate(input_identifiers):
            self.all_items.update({id_: i})
            self.deck_of_items.update({id_: 0})
            self.ids_movements.update({id_: [0]})

    def update_dicts(self):
        self.all_items = {}
        self.deck_of_items = {}
        count = 0
        for i, deck in enumerate(self.decks):
            for id_ in deck:
                self.all_items.update({id_: count})
                self.deck_of_items.update({id_: i})
                count += 1

    def get_all_items(self):
        all_items = []
        for deck in self.decks:
            all_items.extend(deck)
        return all_items

    def wipe(self, i_task):
        ids_to_wipe = []
        num_items_to_wipe = self.max_mem_sz // i_task
        total_num_to_wipe = num_items_to_wipe

        if self.args.wipe_strategy in ["easy", "hard"]:
            lower_deck, upper_deck = 0, self.num_decks - 1

            # wipe hard stuff
            while (
                len(ids_to_wipe) < total_num_to_wipe and 0 <= lower_deck <= upper_deck
            ):
                if self.args.wipe_strategy == "hard":
                    # wiping easy stuff
                    deck_items = self.decks[upper_deck]
                    upper_deck -= 1
                else:
                    # wiping hard stuff
                    deck_items = self.decks[lower_deck]
                    lower_deck += 1

                ids_to_wipe.extend(deck_items[:num_items_to_wipe])

                num_items_to_wipe = total_num_to_wipe - len(ids_to_wipe)

        elif self.args.wipe_strategy == "random":
            # wipe stuff randomly
            # ids_to_wipe = random.choices(
            #     list(self.all_items.keys()), k=total_num_to_wipe
            # )

            # ids_to_wipe = random.sample(list(self.all_items.keys()), total_num_to_wipe)
            ids_to_wipe = random.sample(self.get_all_items(), total_num_to_wipe)

        for id_ in ids_to_wipe:
            if id_ in self.decks[self.deck_of_items[id_]]:
                self.decks[self.deck_of_items[id_]].remove(id_)

        self.update_dicts()

    def init_first_deck_er(
        self, init_lt_scheduler, lang=None, i_task=None, use_wipe=False
    ):
        if use_wipe:
            if i_task and i_task > 1:
                self.wipe(i_task)

            num_items_to_pick = self.max_mem_sz // i_task
        else:
            num_items_to_pick = self.max_mem_sz // self.num_er_langs

        items_l_lang = []

        if self.er_strategy in ["easy", "hard"]:
            total_num_items = num_items_to_pick
            lower_deck, upper_deck = 0, self.num_decks - 1
            while len(items_l_lang) < total_num_items and 0 <= lower_deck <= upper_deck:
                if self.er_strategy == "easy":
                    # If easy pick from upper decks till you fill up the memory
                    deck_items = init_lt_scheduler.decks[upper_deck]
                    upper_deck -= 1
                else:
                    # If hard pick from lower decks till you fill up the memory
                    deck_items = init_lt_scheduler.decks[lower_deck]
                    lower_deck += 1

                if use_wipe:
                    # TODO TODO Testing the other option
                    items_l_lang.extend(deck_items[:num_items_to_pick])
                else:
                    if len(deck_items) <= num_items_to_pick:
                        items_l_lang.extend(deck_items)
                    else:
                        items_l_lang.extend(
                            random.choices(deck_items, k=num_items_to_pick)
                        )

                num_items_to_pick = total_num_items - len(items_l_lang)

        elif self.er_strategy == "random":
            total_num_items = num_items_to_pick
            # if use_wipe:
            #     # TODO TODO Testing the other option
            #     items_l_lang = random.sample(
            #         list(init_lt_scheduler.all_items.keys()), total_num_items
            #     )
            # else:
            #     items_l_lang = random.choices(
            #         list(init_lt_scheduler.all_items.keys()), k=total_num_items
            #     )

            items_l_lang = random.choices(
                list(init_lt_scheduler.all_items.keys()), k=total_num_items
            )

        elif self.er_strategy == "extreme":
            # Pick from easy side
            lower_deck, upper_deck = 0, self.num_decks - 1
            total_num_items = num_items_to_pick // 2
            num_items_to_pick = total_num_items
            while len(items_l_lang) < total_num_items and 0 <= lower_deck <= upper_deck:
                deck_items = init_lt_scheduler.decks[upper_deck]
                upper_deck -= 1

                if use_wipe:
                    # TODO TODO Testing the other option
                    items_l_lang.extend(deck_items[:num_items_to_pick])
                else:
                    if len(deck_items) <= num_items_to_pick:
                        items_l_lang.extend(deck_items)
                    else:
                        items_l_lang.extend(
                            random.choices(deck_items, k=num_items_to_pick)
                        )

                num_items_to_pick = total_num_items - len(items_l_lang)

            # Pick from hard side
            lower_deck, upper_deck = 0, self.num_decks - 1
            num_items_to_pick = total_num_items
            while len(items_l_lang) < total_num_items and 0 <= lower_deck <= upper_deck:
                # If hard pick from lower decks till you fill up the memory
                deck_items = init_lt_scheduler.decks[lower_deck]
                lower_deck += 1

                if use_wipe:
                    # TODO TODO Testing the other option
                    items_l_lang.extend(deck_items[:num_items_to_pick])
                else:
                    if len(deck_items) <= num_items_to_pick:
                        items_l_lang.extend(deck_items)
                    else:
                        items_l_lang.extend(
                            random.choices(deck_items, k=num_items_to_pick)
                        )

                num_items_to_pick = total_num_items - len(items_l_lang)

        # Populate the first deck
        self.decks[0].extend(items_l_lang)
        for i, id_ in enumerate(items_l_lang):
            self.deck_of_items.update({id_: 0})
            self.all_items.update({id_: i})
            self.ids_movements.update({id_: [0]})

    def init_first_deck_er_old(self, init_lt_scheduler, lang=None):
        """
        Initialization method used to initialize ER scheduler by placing everything in the first queue
        In this case, we are given the memory directly
        :param items_l: List of input identifiers
        """
        total_num_items = len(init_lt_scheduler.all_items.keys())
        prop_num_items = math.floor(total_num_items * self.er_strategy_prop)
        lower_deck, upper_deck = 0, self.num_decks - 1
        if self.er_strategy == "hard":
            # Finds the lowest deck with elements
            while len(init_lt_scheduler.decks[lower_deck]) == 0:
                lower_deck += 1
            items_l = init_lt_scheduler.decks[lower_deck]

        elif self.er_strategy == "easy":
            # Finds the highest deck with elements
            while len(init_lt_scheduler.decks[upper_deck]) == 0:
                upper_deck -= 1
            items_l = init_lt_scheduler.decks[upper_deck]

        # Now randomly picks the proportion of elements needed
        if self.er_strategy_prop != 0.0 and len(items_l) > prop_num_items:
            items_l = random.choices(items_l, k=prop_num_items)

        # Populate the first deck with those elements
        if lang:
            items_l_lang = [item for item in items_l if item.split("_")[1] == lang]
        else:
            items_l_lang = items_l

        self.decks[0].extend(items_l_lang)
        for i, id_ in enumerate(items_l_lang):
            self.deck_of_items.update({id_: 0})
            self.all_items.update({id_: i})
            self.ids_movements.update({id_: [0]})

    def next_items(self, epoch):
        """
        Depending on the epoch we are currently at, we review the corresponding queue(s)
        :param epoch:
        :return:
        """
        fifo_batch = []
        for i in range(self.num_decks):
            if (epoch + 1) % (2**i) == 0:
                fifo_batch.extend(self.decks[i])

        if self.lt_sampling_mode == "fifo":
            current_batch = fifo_batch
            current_batch = self.order_items(current_batch)
        elif self.lt_sampling_mode == "rand":
            if self.args.data_name == "mtop":
                not_in_last_deck = [
                    el for el in list(self.all_items.keys()) if el not in self.decks[-1]
                ]
                current_batch = random.choices(not_in_last_deck, k=len(fifo_batch))
                current_batch = self.order_items(current_batch)
            else:
                current_batch = []
                for i in range(self.num_decks):
                    current_batch.extend(self.decks[i])
        else:  # "rand-prop"
            current_deck_n = random.choice(list(range(self.num_decks)))
            while len(self.decks[current_deck_n]) == 0:
                current_deck_n = random.choice(list(range(self.num_decks)))

            rand_num_items = len(
                fifo_batch
            )  # random.choice(list(range(1, self.decks[current_deck_n])))

            current_batch = random.choices(self.decks[current_deck_n], k=rand_num_items)
            current_batch = self.order_items(current_batch)

        return current_batch

    def order_items(self, items):
        # Order the items in the deck since this has an effect on the performance evaluation between different baselines
        return sorted(items, key=lambda d: self.all_items[d])

    def place_items(self, id_outcomes):
        """
        Promotes what needs to be promoted and demotes what needs to be demoted based on the performance
        :param id_outcomes:
        :return:
        """
        for id, outcome in tqdm(id_outcomes.items()):
            current_deck = self.deck_of_items[id]
            if (
                outcome >= self.thresh
            ):  # promote to next deck # TODO COULD PLAY WITH THE THRESHOLD HERE
                next_deck = min(current_deck + 1, self.num_decks - 1)
            else:  # everything else
                if self.demote_to_first:
                    # demote to first deck
                    next_deck = 0
                else:
                    # demote to previous deck
                    next_deck = max(0, current_deck - 1)

            previous_deck = current_deck

            if previous_deck != next_deck:
                # remove from previous deck
                self.decks[previous_deck].remove(id)

                # add to new deck
                self.decks[next_deck].append(id)

                self.deck_of_items.update({id: next_deck})

            self.ids_movements[id].append(next_deck)

    def rep_sched(self):
        """
        Refines and outputs the statistics of the contents of the items on decks
        :return: new_repr: dictionary of the number of items in each queue by language
        """
        new_repr = {}
        for deck_num in range(self.num_decks):
            items = self.decks[deck_num]
            items_repr = {}
            for item in items:
                split, lang, _ = item.split("_")
                if split + "_" + lang not in items_repr:
                    items_repr.update({split + "_" + lang: 0})
                items_repr[split + "_" + lang] += 1

            new_repr.update({deck_num: items_repr})
        return new_repr

    def rep_movements(self):
        """
        Refines the representation of the items on decks
        :return: new_repr: dictionary of the number of movements for each item by queue
        """
        new_repr = {}
        for deck_num in range(self.num_decks):
            new_repr.update({deck_num: self.ids_movements.count(deck_num)})
        return new_repr

    def print_count_decks(self):
        counts = {}
        for deck_num in range(self.num_decks):
            counts.update({deck_num: len(self.decks[deck_num])})
            # print("Deck_num:", deck_num, " count:", len(self.decks[deck_num]))
        return counts

    def populate_memory(self, dataset):
        memory_ids = random.choices(
            self.decks[0], k=self.max_mem_sz // self.languages_len
        )
        memory_examples = [dataset.get_item_by_id(_id) for _id in memory_ids]
        return memory_examples

    def get_by_descriptor(self, descriptor):
        if descriptor == "decks":
            return self.decks
        elif descriptor == "idmovements":
            return self.ids_movements


# class ArgsTest:
#     def __init__(self):
#         self.num_decks = 5
#         self.demote_to_first_deck = False
#         self.lt_sampling_mode = "fifo"
#         self.er_strategy = "random"
#         self.er_strategy_prop = 0.0
#         self.ltn_promote_thresh = 1.0
#         self.max_mem_sz = 1000
#         self.order_lst = "en_de_hi_th"


# if __name__ == "__main__":
#     args_test = ArgsTest()
#     # Main LTN Queues
#     ltn = LeitnerQueue(args=args_test)
#     ltn.decks = [
#         list(range(0, 100)),
#         list(range(100, 140, 1)),
#         list(range(140, 1000, 1)),
#         list(range(1000, 1050, 1)),
#         list(range(1050, 1200, 1)),
#     ]
#     for j, deck in enumerate(ltn.decks):
#         for i, id_ in enumerate(deck):
#             ltn.deck_of_items.update({id_: 0})
#             ltn.all_items.update({id_: i})
#             ltn.ids_movements.update({id_: [0]})

#     # ER LTN QUEUES
#     er_ltn = LeitnerQueue(args=args_test)
#     er_ltn.init_first_deck_er(ltn, i_task=1, use_wipe=True)
#     print("AFTER HOP 1:", len(er_ltn.decks[0]))
#     print("******************************")
#     er_ltn.init_first_deck_er(ltn, i_task=2, use_wipe=True)
#     print("AFTER HOP 2:", len(er_ltn.decks[0]))
#     print("******************************")
#     er_ltn.init_first_deck_er(ltn, i_task=3, use_wipe=True)
#     print("AFTER HOP 3:", len(er_ltn.decks[0]))
