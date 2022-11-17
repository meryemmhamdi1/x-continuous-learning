from tqdm import tqdm
class LeitnerQueue(object):
    def __init__(self,
                 num_decks=5,
                 demote_to_first=False):
        """
        n decks (or FIFO queues)
        """
        self.num_decks = num_decks
        self.decks = [[] for _ in range(self.num_decks)]
        self.deck_of_items = {}
        self.demote_to_first=demote_to_first

    def init_first_deck(self, train_examples, nb_examples, dataset):
        # Place everything in the first deck as an initialization
        for _ in tqdm(range(nb_examples)):
            batch_one, examples \
                = dataset.next_batch(1, train_examples)

            input_identifiers = [example.unique_id for example in examples]

            self.decks[0].append(input_identifiers[0])
            self.deck_of_items.update({input_identifiers[0]: 0})

    # deck 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    # deck 1: 2, 4, 6, 8
    # deck 2: 4, 8

    # epoch 0: whole pass over deck 0
    #
    def next_items(self, epoch):
        """
        Depending on the epoch we are currently at, we review the corresponding queue(s)
        :param epoch:
        :return:
        """
        current_batch = []
        for i in range(self.num_decks):
            if epoch % (2 ** i) == 0:
                current_batch.extend(self.decks[i])

        return current_batch

    def place_items(self, id_outcomes):
        """
        Promotes what needs to be promoted and demotes what needs to be demoted based on the performance
        :param id_outcomes:
        :return:
        """
        for id, outcome in id_outcomes.items():
            current_deck = self.deck_of_items[id]
            if outcome == 1: # promote to next deck
                next_deck = min(current_deck+1, self.num_decks-1)

            else:
                if self.demote_to_first:
                    # demote to first deck
                    next_deck = 0
                else:
                    # demote to previous deck
                    next_deck = max(0, current_deck -1)

            previous_deck = current_deck

            if previous_deck != next_deck:
                # remove from previous deck
                self.decks[previous_deck].remove(id)

                # add to new deck
                self.decks[next_deck].append(id)

                self.deck_of_items.update({id: next_deck})

    def rep_sched(self):
        """
        Refines the representation of the items on decks
        :return:
        """
        new_repr = {}
        for deck_num in range(self.num_decks):
            items = self.decks[deck_num]
            items_repr = {}
            for item in items:
                split, lang, num = item.split("_")
                if split + "_" + lang not in items_repr:
                    items_repr.update({split + "_" + lang: 0})
                items_repr[split + "_" + lang] += 1

            new_repr.update({deck_num: items_repr})
        return new_repr

