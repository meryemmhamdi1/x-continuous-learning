from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from downstreammodels.crf import CRFLayer


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self,
                 model,
                 batch_list,
                 use_slots,
                 device,
                 use_online=True):

        self.model = model
        self.batch_list = batch_list
        self.use_slots = use_slots
        self.device = device
        self.use_online = use_online

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._grads_test = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        self.crf_layer = CRFLayer(self.num_slots, self.device)

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data) # Previous task parameters

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for jj, batch in enumerate(self.batch_list):
            input_ids, lengths, token_type_ids, input_masks, attention_mask, intent_labels, slot_labels, \
            input_texts = batch

            input_ids = input_ids.to(self.device)
            lengths = lengths.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            input_masks = input_masks.to(self.device)
            attention_mask = attention_mask.to(self.device)
            intent_labels = intent_labels.to(self.device)
            slot_labels = slot_labels.to(self.device)

            self.model.zero_grad()
            if self.use_slots:
                logits_intents, logits_slots, intent_loss, slot_loss, loss = self.model(input_ids,
                                                                                        lengths=lengths,
                                                                                        input_masks=input_masks,
                                                                                        intent_labels=intent_labels,
                                                                                        slot_labels=slot_labels)

            else:
                logits_intents, intent_loss, loss = self.model(input_ids,
                                                               lengths=lengths,
                                                               input_masks=input_masks,
                                                               intent_labels=intent_labels)
            output_intents = logits_intents.view(1, -1)
            label_intents = output_intents.max(1)[1].view(-1)
            loss = torch.nn.CrossEntropyLoss()(output_intents, label_intents)
            if self.use_slots:
                slot_loss = self.crf_layer.loss(logits_slots, self.crf_layer(logits_slots))
                loss += slot_loss

            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.batch_list)
                else:
                    print("n:", n)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
