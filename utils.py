import pickle
import os
import torch
import logging


def read_saved_pickle(checkpoint_dir,
                      task_i,
                      obj="grads"):

    with open(os.path.join(checkpoint_dir, "pytorch_"+obj+"_"+str(task_i)), "rb") as file:
        read_obj = pickle.load(file)

    return read_obj


def name_in_list(list, name):
    for el in list:
        if el in name:
            return True
    return False


def format_store_grads(pp,
                       grad_dims,
                       cont_comp,
                       checkpoint_dir=None,
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
        if name_in_list(cont_comp, n):
            if p.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(p.grad.data.view(-1))
            cnt += 1

    if store:
        with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(tid)), "wb") as file:
            pickle.dump(grads, file)

    return grads


def logger(log_file):
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                      datefmt='%d/%m/%Y %H:%M:%S')


    #Setup File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    #Setup Stream Handler (i.e. console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.INFO)

    #Get our logger
    app_log = logging.getLogger('root')
    app_log.setLevel(logging.INFO)

    #Add both Handlers
    app_log.addHandler(file_handler)
    app_log.addHandler(stream_handler)
    return app_log


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)

