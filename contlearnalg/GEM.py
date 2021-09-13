import numpy as np
import torch
import quadprog
from qpth.qp import QPFunction
from torch.autograd import Function, Variable

""" Auxiliary functions copied from Copyright 2017-present, Facebook, Inc. https://github.com/
facebookresearch/GradientEpisodicMemory """


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for n, p in pp:
        if p.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                p.grad.data.size())
            p.grad.data.copy_(this_grad)
        cnt += 1


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
                 model,
                 batch_list,
                 use_slots,
                 use_a_gem):
        """
        Implements GEM and A-GEM
        :param model:
        :param batch_list:
        :param use_slots:
        :param use_a_gem:
        """
        pass
