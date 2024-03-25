from tqdm import tqdm
from schedulers.kernel import Kernel
import random

class LeitnerQueue(object):
    def __init__(self,
                 args):

        """
        args: 
        """
        self.nu = args.nu # recall confidence
        self.kern = args.kern
        self.lt_sampling_mode = args.lt_sampling_mode
        self.er_lq_scheduler_prop = args.er_lq_scheduler_prop
        self.delays = {}
        self.all_items = {}
        
    def init_first_deck(self, dataset, train_examples, nb_examples):
        # Place everything in the first deck as an initialization
        _, examples, _ \
            = dataset.next_batch(nb_examples, train_examples)

        self.input_identifiers = [example.unique_id for example in examples]

        for i, id_ in enumerate(self.input_identifiers):
            self.delays.update({id_: 1})
            self.all_items.update({id_: i})
        
    def init_first_deck_er(self, items_l):
        for i, id_ in enumerate(items_l):
            self.delays.update({id_: 1})
            self.all_items.update({id_: i})

    def next_items(self, epoch):
        """
        :return: the current batch
        """
        fifo_batch = []
        for id_ in self.all_items.keys():
            if self.delays[id_] <= 1:
                fifo_batch.append(id_)

        if self.lt_sampling_mode == "fifo":
            current_batch = fifo_batch
        else:
            current_batch = random.choices(self.all_items.keys(), k=len(fifo_batch))

        current_batch = self.order_items(current_batch)

        # Sampling for ER
        # current_batch = random.choices(current_batch, k=int(len(fifo_batch)*self.er_lq_scheduler_prop))

        return current_batch
    
    def order_items(self, items): 
        print("order_items ===> self.all_items:", len(self.all_items.keys()))
        # print("order_items ===> self.all_items.index(train_en_564):", self.all_items.index("train_en_564"))
        return sorted(items, key=lambda d: self.all_items[d]) 

    def get_delayed_items(self):
        """
        :return: the delayed batch 
        """
        delayed_batch = []
        for id_ in self.all_items.keys():
            if self.delays[id_] > 1:
                delayed_batch.append(id_)

        return delayed_batch

    def place_items(self, val_accs, val_losses, train_accs, train_losses):
        """
        Promotes what needs to be promoted and demotes what needs to be demoted based on the performance
        :param id_outcomes:
        :return:
        """
        curr_batch = self.next_items(0)
        delayed_batch = self.get_delayed_items()

        # Get the updated delays per item in the training data
        print("len(train_accs):", len(train_accs), \
              " len(self.delays):", len(self.delays), \
              " len(curr_batch):", len(curr_batch), \
              " len(val_accs):", len(val_accs),
              " len(train_loss):", len(train_losses))
        kernel = Kernel(self.kern, self.nu, val_accs, val_losses, train_accs, train_losses, self.delays, curr_batch)
        self.delays = kernel.get_optimal_delay()

        # Update at the end for delayed items
        for id_ in delayed_batch:
            self.delays[id_] -= 1
        
    def rep_sched(self):
        """
        Refines the representation of the items on decks
        :return:
        """
        curr_batch = self.next_items(0)
        delayed_batch = self.get_delayed_items()

        items_dict = {"current": curr_batch, "delayed": delayed_batch}
        new_repr = {}
        for items_name, items in items_dict.items():
            items_repr = {}
            for item in items:
                split, lang, num = item.split("_")
                if split + "_" + lang not in items_repr:
                    items_repr.update({split + "_" + lang: 0})
                items_repr[split + "_" + lang] += 1

            new_repr.update({items_name: items_repr})
        return new_repr
    
    def print_count_decks(self):
        print("**********************Contents Decks*************************")
        for deck_num in range(self.num_decks):
            print("Deck_num:", deck_num, " count:", len(self.decks[deck_num]))
        print("*************************************************************")

