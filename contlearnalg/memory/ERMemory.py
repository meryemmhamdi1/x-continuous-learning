import random
from consts import intent_types, slot_types
import numpy as np
import torch
# from utils import euclid_dist
from consts import EPSILON

def euclid_dist(x1, x2):
    """
    Computes the euclidean distance between two embedding vectors
    :param x1:
    :param x2:
    :return:
    """
    # return torch.mean(torch.cdist(x1, x2))
    return torch.sqrt(torch.sum((x1-x2)**2))


"""
    memory += train_examples.next_items(int(train_subtask_lang["size"]*args.old_task_prop))
    memory_aug_list = AugmentedList(memory, shuffle_between_epoch=True)
"""


def init_var(num_tasks, num_intents, init_val):
    """
    For reservoir sampling, it creates a simple empty memory buffer for all intent classes
    For "ring" and "k-means" add examples per intent class
    :param num_tasks:
    :param num_intents:
    :param init_val:
    :return:
    """
    if isinstance(init_val, list):
        return {task: {intent: [] for intent in range(num_intents)} for task in range(num_tasks)}
    else:
        return {task: {intent: init_val for intent in range(num_intents)} for task in range(num_tasks)}


class Example(object):
    def __init__(self,
                 embed,
                 x,
                 y_intent,
                 y_slot,
                 distance,
                 task_id):
        self.embed = embed  ## key
        ## Values
        self.x = x  # dictionary of input_ids, token_type_ids, input_masks, lengths, input_texts
        self.y_intent = y_intent  # intent_labels
        self.y_slot = y_slot  # slot_labels
        self.distance = distance
        self.task_id = task_id
        if distance != 0:
            self.weight = 1.0/(EPSILON*distance)  # should be only applicable in the case of MbPA
        else:
            self.weight = 0.0

    def get_embed(self):
        return self.embed

    def get_x(self):
        return self.x

    def get_y_intent(self):
        return self.y_intent

    def get_y_slot(self):
        return self.y_slot

    def get_distance(self):
        return self.distance

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def __str__(self):
        return self.x["input_texts"]#, " Intent:", intent_types[self.y_intent.squeeze().item()],\
               #" Slots:", [slot_types[slot] for slot in self.y_slot.squeeze()]


class Memory(object):
    def __init__(self,
                 max_mem_sz,
                 total_num_tasks,
                 embed_dim,
                 storing_type,
                 sampling_type):
        """
        Base replay memory class with different utilties that can be used for MbPA as well or any other
            episodic memory approach
        :param max_mem_sz: the maximum size of the memory
        :param total_num_tasks: the total number of tasks in the whole continual learning process
        :param storing_type: choices -> reservoir, ring, k-means, mof # TODO implement mof if time, interest
        :param sampling_type: choices -> random
        :param embed_dim: the dimensionality of the embeddings
        """

        self.max_mem_sz = max_mem_sz
        self.total_num_tasks = total_num_tasks
        self.storing_type = storing_type
        self.sampling_type = sampling_type

        num_intents = 1 if self.storing_type == "reservoir" else len(intent_types)
        self.memory = init_var(total_num_tasks, num_intents, [])
        self.j = init_var(total_num_tasks, num_intents,  0)
        self.cluster_count = init_var(total_num_tasks, num_intents,  0)
        self.cur_sz = init_var(total_num_tasks, num_intents, 0)

        self.centroids = {intent: torch.zeros(embed_dim) for intent in range(len(intent_types))}
        # TODO: maybe have those already computed in the first task but maybe the memory is only used in the second task

        # TODO: add clusters per intents and slots at the same time

    def update(self, batch, embeds, task):
        """
        THIS IS SEEN ONLY ONCE PER TASK SO THE MEMORY WILL STAY STATIC PER TASK UNLESS IF WE MIX EVERYTHING THEN THE MEMORY WILL BE UPDATED AND WE WILL LOSE PREVIOUS TASKS
        WE COULD IMPLEMENT A ROUTINE TO CONTINUOUSLY UPDATE THE MEMORY PER TASK EVEN AFTER THE TASK WAS SEEN
        Looks at the examples to select which examples to add to the memory
        It then stores them as tuples of input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts
        :param batch: current examples in terms of: (input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts)
        :param embeds: embeddings of all the inputs
        :param task: current task to store examples from
        :return:
        """
        input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts = batch
        self.update_centroids(embeds, intent_labels)
        batch_size = input_ids.shape[0]
        seen_intents = []
        for i in range(batch_size):
            embed = embeds[i]
            x = {"input_ids": input_ids[i],
                 "token_type_ids": token_type_ids[i],
                 "input_masks": input_masks[i],
                 "lengths": lengths[i],
                 "input_texts": input_texts[i]}
            y_intent = intent_labels[i]
            y_slot = slot_labels[i]

            max_size_key = self.max_mem_sz // self.total_num_tasks

            if self.storing_type == "reservoir":
                key = 0
            elif self.storing_type == "ring":
                key = intent_labels[i].squeeze().item()
                max_size_key = max_size_key // len(intent_types)
                seen_intents.append(key)
            else:  # k-means":
                distances = torch.stack([euclid_dist(embed, self.centroids[intent])
                                         for intent in self.centroids])

                key = torch.argmin(distances).squeeze().item()
                max_size_key = max_size_key // len(intent_types)

            distance = euclid_dist(embed, self.centroids[key])

            # if task == 2:
            #     print(x["input_texts"], y_intent, key, len(self.memory[task][key]), max_size_key)
            # print("len(self.memory[task][key]):", len(self.memory[task][key]),
            #       " max_size_key:", max_size_key)
            if len(self.memory[task][key]) < max_size_key:
                self.cur_sz[task][key] += 1
                if self.storing_type == "k-means":
                    # Update the centroids
                    self.cluster_count[task][key] += 1
                    self.centroids[key] += (embeds[i] - self.centroids[key]) / self.cur_sz[task][key]

                self.memory[task][key].append(Example(embed=embed,
                                                      x=x,
                                                      y_intent=y_intent,
                                                      y_slot=y_slot,
                                                      distance=distance,
                                                      task_id=task))
            else:
                if self.storing_type != "k-means":
                    # pick a number at random to be replaced by the current task
                    # print("self.cur_sz[task][key]+self.j[task][key]:", self.cur_sz[task][key]+self.j[task][key],
                    #       "self.memory:", self.memory)
                    k = random.sample(list(range(0, self.cur_sz[task][key]+self.j[task][key])), k=1)[0]
                    if k < max_size_key:
                        self.memory[task][key][k] = Example(embed=embed,
                                                            x=x,
                                                            y_intent=y_intent,
                                                            y_slot=y_slot,
                                                            distance=distance,
                                                            task_id=task)
                else:

                    distances = [pair.distance for pair in self.memory[task][key]]
                    k = np.argmax(distances)
                    if distance <= distances[k]:
                        self.memory[task][key][k] = Example(embed=embed,
                                                            x=x,
                                                            y_intent=y_intent,
                                                            y_slot=y_slot,
                                                            distance=distance,
                                                            task_id=task)
            self.j[task][key] += 1

        # print("task:", task, " seen_intents:", seen_intents, " max_size_key:", max_size_key)

    def sample(self, task_num, sample_sz):
        """
        Randomly samples
            # Either randomly sampled from all tasks
            # or randomly sampled equally from each task seen
        :param sample_sz: batch_size of the memory sampled
        :return:
        """
        sampled = []
        # This one uses randomly sampled FROM ALL PREVIOUS TASKS
        for task in range(task_num):
            for intent in self.memory[task]:
                if len(self.memory[task][intent]) >= sample_sz//task_num:
                    sampled.extend(random.sample(self.memory[task][intent], k=sample_sz//task_num))
        return sampled

    def size(self):
        """
        The current size of the memory (the number of elements inserted so far across all tasks)
        :return:
        """
        total_size = 0
        for task_num in self.cur_sz:
            size_task = 0
            for intent in self.cur_sz[task_num]:
                total_size += self.cur_sz[task_num][intent]
                size_task += self.cur_sz[task_num][intent]

            print("task_num:", task_num, " size_task:", size_task)
        return total_size

    def update_centroids(self, embeds, intent_labels):
        """
        Computes the centroid of each intent class
        :param embeds:
        :return:
        """
        center_points = {intent: [] for intent in self.centroids}
        for i, intent in enumerate(intent_labels):
            intent = intent.squeeze().item()
            center_points[intent].append(embeds[i])

        for intent in center_points:
            new_intent_list = [self.centroids[intent]] + center_points[intent]
            self.centroids[intent] = torch.mean(torch.stack(new_intent_list), 0)

    def __str__(self):
        str_format = {task_num: [] for task_num in self.memory}
        for task_num in self.memory:
            for intent in self.memory[task_num]:
                for eg in self.memory[task_num][intent]:
                    str_format[task_num].append(eg.__str__())

        return str_format
