from tqdm import tqdm
from kernel import Kernel

class LeitnerQueue(object):
    def __init__(self,
                 nu,
                 kern):
        """
        n decks (or FIFO queues)
        """
        self.nu = nu # recall confidence
        self.kern = kern
        self.delays = {}
        

    def init_first_deck(self, train_examples, nb_examples, dataset):
        # Place everything in the first deck as an initialization
        _, examples \
            = dataset.next_batch(nb_examples, train_examples)

        self.input_identifiers = [example.unique_id for example in examples]

        for id_ in self.input_identifiers:
            self.delays.update({id_: 1})
        

    def next_items(self):
        """
        :return: the current batch
        """
        current_batch = []
        for id_ in self.input_identifiers:
            if self.delays[id_] <= 1:
                current_batch.append(id_)

        return current_batch

    def get_delayed_items(self):
        """
        :return: the delayed batch 
        """
        delayed_batch = []
        for id_ in self.input_identifiers:
            if self.delays[id_] > 1:
                delayed_batch.append(id_)

        return delayed_batch

    def place_items(self, val_accs, val_loss, train_accs, train_loss):
        """
        Promotes what needs to be promoted and demotes what needs to be demoted based on the performance
        :param id_outcomes:
        :return:
        """
        curr_batch = self.next_items()
        delayed_batch = self.get_delayed_items()

        # Get the updated delays per item in the training data
        kernel = Kernel(self.kern, self.nu, val_accs, val_loss, train_accs, train_loss, self.delays, curr_batch)
        self.delays = kernel.get_optimal_delay()

        # Update at the end for delayed items
        for id_ in delayed_batch:
            self.delays[id_] -= 1
        
    def rep_sched(self):
        """
        Refines the representation of the items on decks
        :return:
        """
        curr_batch = self.next_items()
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

