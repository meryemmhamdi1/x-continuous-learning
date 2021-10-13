from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from basemodels.crf import CRFLayer
from utils import read_saved_pickle, name_in_list


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):

    """
    if i_task > 0
        # the list of previous tasks
        # batches_list = []
        # for sub_task_i in range(i_task):
        #     for _ in range(sample_sizes[sub_task_i]):
        #         batch, _ = dataset.next_batch(1, old_dataset[sub_task_i]["stream"])
        #         batches_list.append(batch)
        #
        # ewc = EWC(model,
        #           batches_list,
        #           args.use_slots,
        #           device,
        #           args.use_online)
        # print((args.ewc_lambda / 2) * ewc.penalty(model))

        reg_term = 0
        if args.use_online:
            fisher_param = {n: 0 for n, p in model.named_parameters() if p.requires_grad}
            for fj in range(1, i_task):
                with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(i_task-fj)), "rb") as file:
                    grads = pickle.load(file)

                for n, p in model.named_parameters():
                    grad_k = grads[n]
                    fisher_param[n] += (args.gamma_ewc ** (fj-1)) * grad_k
                    del grad_k

            with open(os.path.join(checkpoint_dir, "pytorch_params_"+str(i_task-1)), "rb") as file:
                params = pickle.load(file)

            for n, p in model.named_parameters():
                p_k = params[n]
                _reg_term = fisher_param[n] * (p - p_k) ** 2
                del p_k
                reg_term += _reg_term.sum()
        else:
            for k in range(0, i_task):
                with open(os.path.join(checkpoint_dir, "pytorch_params_"+str(k)), "rb") as file:
                    params = pickle.load(file)

                with open(os.path.join(checkpoint_dir, "pytorch_grads_"+str(k)), "rb") as file:
                    grads = pickle.load(file)

                for n, p in model.named_parameters():
                    torch.cuda.empty_cache()
                    if p.grad is not None and p.requires_grad:
                        p_k = params[n]
                        grad_k = grads[n]
                        _reg_term = grad_k * (p - p_k) ** 2
                        del grad_k
                        del p_k
                        reg_term += _reg_term.sum()

        print("i_task: ", i_task, " reg_term:", reg_term, " reg_term.grad:", reg_term.grad)

        loss += (args.ewc_lambda / 2) * reg_term
    """
    def __init__(self,
                 device,
                 checkpoint_dir,
                 use_online,
                 gamma_ewc,
                 cont_comp):

        self.device = device
        self.use_online = use_online
        self.checkpoint_dir = checkpoint_dir
        self.gamma_ewc = gamma_ewc
        self.cont_comp = cont_comp

    def compute_online_fisher(self, i_task, model):
        fisher_param = {n: 0 for n, p in model.named_parameters() if p.requires_grad and name_in_list(self.cont_comp, n)}
        for fj in range(1, i_task):
            grads = read_saved_pickle(self.checkpoint_dir,
                                      i_task-fj,
                                      "grads")

            for n, p in model.named_parameters():
                if name_in_list(self.cont_comp, n):
                    grad_k = grads[n]
                    fisher_param[n] += (self.gamma_ewc ** (fj-1)) * grad_k
                    del grad_k

        return fisher_param

    def penalty(self, i_task, model):
        reg_term = 0
        if self.use_online:
            fisher_param = self.compute_online_fisher(i_task, model)
            params = read_saved_pickle(self.checkpoint_dir, i_task-1, "params")
            for n, p in model.named_parameters():
                if name_in_list(self.cont_comp, n):
                    p_k = params[n]
                    _reg_term = fisher_param[n] * (p - p_k) ** 2
                    del p_k
                    reg_term += _reg_term.sum()
        else:
            for k in range(0, i_task):
                fisher_param = read_saved_pickle(self.checkpoint_dir, k, "grads")

                params = read_saved_pickle(self.checkpoint_dir, k, "params")

                for n, p in model.named_parameters():
                    if name_in_list(self.cont_comp, n):
                        torch.cuda.empty_cache()
                        if p.grad is not None and p.requires_grad:
                            p_k = params[n]
                            fisher_param_k = fisher_param[n]
                            _reg_term = fisher_param_k * (p - p_k) ** 2
                            del fisher_param_k
                            del p_k
                            reg_term += _reg_term.sum()
        return reg_term
