import torch
import random

from utils import euclid_dist
from contlearnalg.memory.ERMemory import Memory as ERMemory


class Memory(ERMemory):
    def __init__(self,
                 max_mem_sz,
                 total_num_tasks,
                 embed_dim,
                 storing_type="ring",
                 sampling_type="near_n"):

        """
        Fixed size but dynamically-updated as it acts as a circular buffer:
            when it is full, the oldest data is overwritten first
            Either fixed or trainable key network
            MbPA: samples k-nearest neighbours examples
            MbPA++: uses randomly sampled examples
        :param max_mem_sz: the maximum size of the memory
        :param total_num_tasks: the total number of tasks in the whole continual learning process
        :param embed_dim: the dimensionality of the embeddings
        :param storing_type: choices -> reservoir, ring, k-means, mof # TODO implement mof if time, interest
        :param sampling_type: choices -> random, near_n if it random it is MbPA++
        """

        super(Memory, self).__init__(max_mem_sz=max_mem_sz,
                                     total_num_tasks=total_num_tasks,
                                     embed_dim=embed_dim,
                                     storing_type=storing_type,
                                     sampling_type=sampling_type)

    def get_neighbours(self, q, task, k):
        """
        Finds the nearest neighbours of a testing query example q
        :param q: content + its embeddings of one query example (content, embed)
        :param task: the task up to which the training has reached
        :param k: the number of nearest neighbours to sample
        :return:
        """
        distances = []
        examples = []
        for task_num in range(task):  # sample from all previously seen TASKS before
            for intent in self.memory[task_num]:
                for eg in self.memory[task_num][intent]:
                    # For each example in the memory, find the distance with the query
                    examples.append(eg)
                    distances.append(euclid_dist(eg.embed, q.embed))

        distances = torch.stack(distances)

        neighbour_keys = torch.topk(distances, k, largest=False)
        print(neighbour_keys[:2])

        # neighbours = examples.index_select(0, neighbour_keys)
        neighbours = [examples[key.squeeze().item()] for key in neighbour_keys]

        return neighbours

    def sample_for_q(self, q, task, sample_sz):
        """
         Use randomly or k-nearest neighbours sampled examples up to a particular task for local adaptation
        :param q: the testing example (content, embeddings)
        :param task: the current task till when we have trained so far
        :param sample_sz: the number of examples to be randomly sampled or the number of nearest neighbours
        :return:
        """
        if self.sampling_type == "random":
            sampled = self.sample(task, sample_sz)

        else:
            # Use k-nearest neighbours
            sampled = self.get_neighbours(q, task, sample_sz)

        return sampled
        # TODO get_neighbours with the batch format for forward pass in the main
        # TODO normalize the format of the memory