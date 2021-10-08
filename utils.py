import pickle
import os
import torch


def read_saved_pickle(checkpoint_dir,
                      task_i,
                      obj="grads"):

    with open(os.path.join(checkpoint_dir, "pytorch_"+obj+"_"+str(task_i)), "rb") as file:
        read_obj = pickle.load(file)

    return read_obj


def format_store_grads(pp,
                       grad_dims,
                       checkpoint_dir,
                       tid=-1,
                       store=True):
    """
        This stores parameter gradients of one task at a time.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads = torch.Tensor(sum(grad_dims))
    grads.fill_(0.0)
    cnt = 0
    for n, p in pp:
        if p.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(p.grad.data.view(-1))
        cnt += 1

    if store:
        with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(tid)), "wb") as file:
            pickle.dump(grads, file)

    return grads
