import numpy as np
import torch
import quadprog
from qpth.qp import QPFunction
from torch.autograd import Function, Variable
from utils import read_saved_pickle, format_store_grads, name_in_list
import random

""" Auxiliary functions copied from Copyright 2017-present, Facebook, Inc. https://github.com/
facebookresearch/GradientEpisodicMemory """


def overwrite_grad(pp, newgrad, grad_dims, cont_comp):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """

    cnt = 0
    for n, p in pp:
        if p.grad is not None and name_in_list(cont_comp, n):
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            print("cnt:", cnt, " n:", n, "beg:", beg, " en:", en, " newgrad.shape:", newgrad[beg: en].shape, " p.grad.data.size():", p.grad.data.size())
            this_grad = newgrad[beg: en].contiguous().view(
                p.grad.data.size())
            p.grad.data.copy_(this_grad)
        if name_in_list(cont_comp, n):
            cnt += 1

    return


def project2cone2_cpu(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    print("Solving the GEM dual QP ===>")

    memories_np = memories.cpu().t().double().numpy()
    memories_cuda = memories.t().double()
    print("---------memories:", memories, " memories.shape:", memories.shape)
    print("---------memories_np:", memories_np, " memories_np.shape:", memories_np.shape)
    print("---------memories_cuda:", memories_cuda, " memories_cuda.shape:", memories_cuda.shape)

    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    gradient_cuda = gradient.contiguous().view(-1).double()

    print("---------gradient:", gradient, " gradient.shape:", gradient.shape)
    print("---------gradient_np:", gradient_np, " gradient_np.shape:", gradient_np.shape)
    print("---------gradient_cuda:", gradient_cuda, " gradient_cuda.shape:", gradient_cuda.shape)
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    print("Solving the GEM dual QP ===>")
    memories_cuda = memories.t()
    gradient_cuda = gradient.contiguous().view(-1)
    t = memories_cuda.shape[0]
    print("memories_cuda.shape:", memories_cuda.shape, " gradient_cuda.shape", gradient_cuda.shape, " t:", t)
    P = torch.matmul(memories_cuda, memories_cuda.t())
    P = 0.5 * (P + P.t()) + torch.eye(t).cuda() * eps
    q = torch.matmul(memories_cuda, gradient_cuda) * -1
    G = torch.eye(t).cuda()
    h = torch.zeros(t).cuda() + margin
    e = Variable(torch.Tensor()).cuda()
    print("P.type():", P.type(), " q.type():", q.type(), " G.type():", G.type(), " h.type():", h.type)
    v = QPFunction(verbose=False)(P, q, G, h, e, e)[0]
    x = torch.matmul(v, memories_cuda) + gradient_cuda
    gradient.copy_(x.view(-1, 1))
    print("x:", x)


class GEM(object):
    def __init__(self,
                 dataset,
                 use_slots,
                 use_a_gem,
                 a_gem_n,
                 checkpoint_dir,
                 cont_comp):
        """
        Implements GEM and A-GEM
        :param model:
        :param batch_list:
        :param use_slots:
        :param use_a_gem:
        """
        self.dataset = dataset
        self.use_slots = use_slots
        self.use_a_gem = use_a_gem
        self.a_gem_n = a_gem_n
        self.checkpoint_dir = checkpoint_dir
        self.cont_comp = cont_comp

    def forward_pass(self, model, rand_task):
        model.zero_grad()
        batch, _ = self.dataset.next_batch(1, self.dataset.train_stream[rand_task]["examples"])

        input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, input_texts = batch

        input_ids = input_ids.cuda()
        lengths = lengths.cuda()
        input_masks = input_masks.cuda()
        intent_labels = intent_labels.cuda()
        slot_labels = slot_labels.cuda()

        if self.use_slots:
            logits_intents, logits_slots, intent_loss, slot_loss, loss, pooled_output \
                = model(input_ids=input_ids,
                        input_masks=input_masks,
                        train_idx=0,  # TODO dummy
                        lengths=lengths,
                        intent_labels=intent_labels,
                        slot_labels=slot_labels)

        else:
            logits_intents, intent_loss, loss, pooled_output = model(input_ids=input_ids,
                                                                     input_masks=input_masks,
                                                                     train_idx=0,  # TODO dummy
                                                                     lengths=lengths,
                                                                     intent_labels=intent_labels)

        loss.backward()

    def run(self, i_task, sample_sizes, model, grad_dims):
        if self.use_a_gem:
            # A pass over a random batch or over all batches from old tasks
            for _ in range(self.a_gem_n):
                rand_task = random.choice(range(i_task))

                self.forward_pass(model, rand_task)

            # Formatting g_{ref}
            grads_ref = format_store_grads(pp=model.named_parameters(),
                                           grad_dims=grad_dims,
                                           cont_comp=self.cont_comp,
                                           store=False).cuda()

            # Loading g
            grads = read_saved_pickle(self.checkpoint_dir,
                                      i_task,
                                      "grads").cuda()

            dotp = torch.mm(grads.unsqueeze(0),
                            grads_ref.unsqueeze(1))

            if (dotp < 0).sum() != 0:
                # Apply closed form solution
                dotref = torch.mm(grads_ref.unsqueeze(0),
                                  grads_ref.unsqueeze(1))
                new_grad = grads - (dotp / dotref) * grads_ref

                print("new_grad:", new_grad.shape)

                overwrite_grad(model.named_parameters(),  # this is overwritten by the function
                               torch.transpose(new_grad, 0, 1),   # these are the new gradients
                               grad_dims,
                               self.cont_comp)
        else:
            ## GEM
            for old_task_i in range(i_task):  # for all old tasks before current task i
                for _ in range(sample_sizes[old_task_i]):
                    # forward pass on the previous data stored in the memory

                    self.forward_pass(model, old_task_i)

                # Store the gradients for the current step
                format_store_grads(pp=model.named_parameters(),
                                   grad_dims=grad_dims,
                                   cont_comp=self.cont_comp,
                                   checkpoint_dir=self.checkpoint_dir,
                                   tid=old_task_i,
                                   store=True)

                # Solve the dual of the quadratic equation # TODO double check if this is done only once at the end of all
                indx = torch.LongTensor(list(range(i_task))).cuda() if torch.cuda.device_count() > 0 \
                    else torch.LongTensor(list(range(i_task)))

                grads_old = read_saved_pickle(self.checkpoint_dir,
                                              indx.data.cpu().numpy()[0],
                                              "grads").cuda()

                grads = read_saved_pickle(self.checkpoint_dir,
                                          i_task,
                                          "grads").cuda()

                dotp = torch.mm(grads.unsqueeze(0),
                                grads_old.unsqueeze(1))  # grads.index_select(1, indx))

                # print("Computed multiplication")

                # check if the constraints have been violated
                if (dotp < 0).sum() != 0:
                    # If it is the case:
                    # Copy the gradients

                    # print("Project2cone2")
                    project2cone2(grads.unsqueeze(1),
                                  grads_old.unsqueeze(1))  # grads.index_select(1, indx))

                    # Update the named_parameters accordingly
                    overwrite_grad(model.named_parameters(),
                                   grads,
                                   grad_dims,
                                   self.cont_comp)

                    format_store_grads(pp=grads,
                                       grad_dims=grad_dims,
                                       cont_comp=self.cont_comp,
                                       checkpoint_dir=self.checkpoint_dir,
                                       tid=i_task,
                                       store=True)  # storing after changing
