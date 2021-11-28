import torch
from torch.optim import Adam

from copy import deepcopy
from basemodels.transNLUCRF import TransNLUCRF
from tqdm import tqdm
import gc


def euclid_dist(x1, x2):
    """
    Computes the euclidean distance between two embedding vectors
    :param x1:
    :param x2:
    :return:
    """
    # return torch.mean(torch.cdist(x1, x2))
    return torch.sqrt(torch.sum((x1-x2)**2, dim=len(list(x1.shape))-1))


def freeze_trans_get_weights(model):
    for n, p in model.named_parameters():
        if p.requires_grad and "trans_model" in n:
            p.requires_grad = False

    return list(model.parameters())

def get_pred_weights(model):
    return [p for n, p in model.named_parameters() if "trans_model" not in n]


def sample_for_q_custom(q, task_id, memory, model, sampling_k, sampling_type, dataset, device):
    task_batches = []
    if sampling_type == "random":
        for task_num in range(task_id):
            task_batch, _ = dataset.next_batch(sampling_k//task_id, memory[task_num])
            input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, _, _ = task_batch
            task_batch = input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels
            task_batches.append(task_batch)
    else:
        for task_num in range(task_id):
            # for task_num in range(task_id+1): IF we want to consider all previously seen tasks
            batch, _ = dataset.next_batch(sampling_k*10//task_id, memory[task_num])

            # Computing distances in a bulk on all of the memory items
            input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, _ , _ = batch

            if device != torch.device("cpu"):
                input_ids = input_ids.cuda()
                input_masks = input_masks.cuda()

            # Finding the embeddings of this
            embeddings = model.get_embeddings(input_ids, input_masks)
            distances = euclid_dist(embeddings, q)

            # Finding the nearest neighbour keys
            neighbour_keys = torch.topk(distances, sampling_k//task_id, largest=False)[1]

            # Get only the indices of the nearest neighbours
            task_batch = input_ids[neighbour_keys], \
                lengths[neighbour_keys],  \
                token_type_ids[neighbour_keys], \
                input_masks[neighbour_keys], \
                intent_labels[neighbour_keys], \
                slot_labels[neighbour_keys] #, \
                # input_texts[neighbour_keys]

            task_batches.append(task_batch)

    return task_batches # tuple(map(torch.cat, zip(*task_batches)))

def sample_for_q_custom_many_batches_per_task(q, task_id, memory, model, sampling_k, sampling_type, dataset, device, num_batches):
    task_batches = []
    if sampling_type == "random":
        for _ in range(num_batches):
            task_batch, _ = dataset.next_batch(sampling_k//task_id, memory[task_id])
            task_batches.append(task_batch)
    else:
        for _ in range(num_batches):
            # for task_num in range(task_id+1): IF we want to consider all previously seen tasks
            batch, _ = dataset.next_batch(sampling_k*10//task_id, memory[task_id])

            # Computing distances in a bulk on all of the memory items
            input_ids, lengths, token_type_ids, input_masks, intent_labels, slot_labels, _, _ = batch

            if device != torch.device("cpu"):
                input_ids = input_ids.cuda()
                input_masks = input_masks.cuda()

            # Finding the embeddings of this
            embeddings = model.get_embeddings(input_ids, input_masks)
            distances = euclid_dist(embeddings, q)

            # Finding the nearest neighbour keys
            neighbour_keys = torch.topk(distances, sampling_k//task_id, largest=False)[1]

            # Get only the indices of the nearest neighbours
            task_batch = input_ids[neighbour_keys], \
                lengths[neighbour_keys],  \
                token_type_ids[neighbour_keys], \
                input_masks[neighbour_keys], \
                intent_labels[neighbour_keys], \
                slot_labels[neighbour_keys] #, \
                # input_texts[neighbour_keys]

            task_batches.append(task_batch)

    return task_batches # tuple(map(torch.cat, zip(*task_batches)))


class MBPA(object):
    def __init__(self,
                 args,
                 device,
                 model_trans,
                 num_tasks,
                 num_intents,
                 eff_num_intents_task,
                 eff_num_slot):
        self.args = args
        self.device = device
        self.model_trans = model_trans
        self.num_tasks = num_tasks
        self.num_intents = num_intents
        self.eff_num_intents_task = eff_num_intents_task
        self.eff_num_slot = eff_num_slot

        self.adaptive_model = TransNLUCRF(args=self.args,
                                          trans_model=self.model_trans,
                                          num_tasks=self.num_tasks,
                                          num_intents=self.num_intents,
                                          eff_num_intents_task=self.eff_num_intents_task,
                                          device=self.device,
                                          num_slots=self.eff_num_slot)

    def forward(self, memory, q, task_id, model, dataset):

        # Using local parameters different each time for each example reinitialized deepcopy
        model_dict = model.state_dict()
        self.adaptive_model.load_state_dict(model_dict)

        if self.device != torch.device("cpu"):
            self.adaptive_model.cuda()
        # adaptive_examples = memory.sample_for_q(q, train_idx, self.args.sampling_k) # OLD
        adaptive_batches = sample_for_q_custom(q, \
                                               task_id, \
                                               memory, \
                                               model, \
                                               self.args.sampling_k, \
                                               self.args.sampling_type, \
                                               dataset, \
                                               self.device)

        self.adaptive_model.train()

        # Freeze BertModel and adapt only the classifier module # key network is non-trainable in this case
        adaptive_weights = freeze_trans_get_weights(self.adaptive_model)
        # base_weights = freeze_trans_get_weights(model)
        base_weights = [p for n, p in model.named_parameters() if "trans_model" not in n]

        # Set an adaptive optimizer each time from scratch
        adaptive_optimizer = Adam(filter(lambda p: p.requires_grad, self.adaptive_model.parameters()),
                                  betas=(self.args.beta_1, self.args.beta_2),
                                  eps=self.args.adam_eps,
                                  lr=self.args.adaptive_adam_lr)

        # Train the model for a few epochs over the batch retrieved from the memory
        for adaptive_batch in adaptive_batches:
            # Freeze BertModel and adapt only the classifier module # key network is non-trainable in this case
            adaptive_weights = freeze_trans_get_weights(self.adaptive_model)
            base_weights = [p for n, p in model.named_parameters() if "trans_model" not in n]

            # Pass the batch to the current device
            ada_input_ids, ada_lengths, ada_token_type_ids, ada_input_masks, \
                ada_intent_labels, ada_slot_labels = adaptive_batch

            if self.device != torch.device("cpu"):
                ada_input_ids = ada_input_ids.cuda()
                ada_input_masks = ada_input_masks.cuda()
                ada_lengths = ada_lengths.cuda()
                ada_intent_labels = ada_intent_labels.cuda()
                ada_slot_labels = ada_slot_labels.cuda()

            # Train adaptive classifier for args.adaptive_epochs epochs
            # total_loss = 0
            for _ in range(self.args.adaptive_epochs): # tqdm
                # Step by step forward pass on the batch for adaptive_epochs epochs
                adaptive_optimizer.zero_grad()

                if self.args.use_slots:
                    intent_logits, slot_logits, logits_slots_, intent_loss_ada, slot_loss_ada, \
                      adaptive_loss, pooled_output = self.adaptive_model(input_ids=ada_input_ids,
                                                                         input_masks=ada_input_masks,
                                                                         train_idx=task_id,
                                                                         lengths=ada_lengths,
                                                                         intent_labels=ada_intent_labels,
                                                                         slot_labels=ada_slot_labels)

                    del slot_logits
                    del logits_slots_
                    del slot_loss_ada
                else:
                    intent_logits, intent_loss_ada, adaptive_loss, pooled_output \
                        = self.adaptive_model(input_ids=ada_input_ids,
                                              input_masks=ada_input_masks,
                                              train_idx=task_id,
                                              lengths=ada_lengths,
                                              intent_labels=ada_intent_labels)

                del intent_logits
                del intent_loss_ada
                del pooled_output

                # Initialize delta_loss_x to zero
                delta_loss_x = torch.Tensor([0]).to(self.device)

                # Compute the regularization term using adaptive_weights
                # loss with respect to only classifier model
                for base_param, curr_param in zip(base_weights, adaptive_weights):
                    if curr_param.requires_grad:
                        delta_loss_x += (curr_param-base_param).pow(2).sum()

                # Total loss due to log likelihood and weight restraint
                # total_loss = self.args.beta*delta_loss_x + eg.weight.detach() * adaptive_loss
                total_loss = self.args.beta*delta_loss_x + adaptive_loss
                del delta_loss_x
                del adaptive_loss

                total_loss = total_loss.mean()
                total_loss.backward()
                adaptive_optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

        return self.adaptive_model

    def forward_reptile_one_batch(self, memory, q, task_id, model, dataset):

        # Using local parameters different each time for each example reinitialized deepcopy
        model_dict = model.state_dict()
        self.adaptive_model.load_state_dict(model_dict)

        if self.device != torch.device("cpu"):
            self.adaptive_model.cuda()
        # adaptive_examples = memory.sample_for_q(q, train_idx, self.args.sampling_k) # OLD
        adaptive_batches = sample_for_q_custom(q,
                                               task_id,
                                               memory,
                                               model,
                                               self.args.sampling_k,
                                               self.args.sampling_type,
                                               dataset,
                                               self.device)

        # Freeze BertModel and adapt only the classifier module # key network is non-trainable in this case
        freeze_trans_get_weights(self.adaptive_model)
        self.adaptive_model.train()

        # Set an adaptive optimizer each time from scratch
        adaptive_optimizer = Adam(filter(lambda p: p.requires_grad, self.adaptive_model.parameters()),
                                  betas=(self.args.beta_1, self.args.beta_2),
                                  eps=self.args.adam_eps,
                                  lr=self.args.alpha_reptile)

        within_batch_ini_weights = {n:p for n, p in model.named_parameters() if "trans_model" not in n}

        # Train the model for a few epochs over the batch retrieved from the memory
        for adaptive_batch in adaptive_batches:

            # We initialize our adaptive_model based on initial_weights
            adaptive_model_dict = self.adaptive_model.state_dict()
            adaptive_model_dict.update(within_batch_ini_weights)
            self.adaptive_model.load_state_dict(adaptive_model_dict)

            # Freeze BertModel and adapt only the classifier module # key network is non-trainable in this case
            adaptive_weights = get_pred_weights(self.adaptive_model)
            base_weights = get_pred_weights(model)

            # Pass the batch to the current device
            ada_input_ids, ada_lengths, ada_token_type_ids, ada_input_masks, \
                ada_intent_labels, ada_slot_labels = adaptive_batch

            if self.device != torch.device("cpu"):
                ada_input_ids = ada_input_ids.cuda()
                ada_input_masks = ada_input_masks.cuda()
                ada_lengths = ada_lengths.cuda()
                ada_intent_labels = ada_intent_labels.cuda()
                ada_slot_labels = ada_slot_labels.cuda()

            # Train adaptive classifier for args.adaptive_epochs epochs
            # total_loss = 0
            for _ in range(self.args.adaptive_epochs): # tqdm
                # Step by step forward pass on the batch for adaptive_epochs epochs
                adaptive_optimizer.zero_grad()

                if self.args.use_slots:
                    intent_logits, slot_logits, logits_slots_, intent_loss_ada, slot_loss_ada, \
                      adaptive_loss, pooled_output = self.adaptive_model(input_ids=ada_input_ids,
                                                                         input_masks=ada_input_masks,
                                                                         train_idx=task_id,
                                                                         lengths=ada_lengths,
                                                                         intent_labels=ada_intent_labels,
                                                                         slot_labels=ada_slot_labels)

                    del slot_logits
                    del logits_slots_
                    del slot_loss_ada
                else:
                    intent_logits, intent_loss_ada, adaptive_loss, pooled_output \
                        = self.adaptive_model(input_ids=ada_input_ids,
                                              input_masks=ada_input_masks,
                                              train_idx=task_id,
                                              lengths=ada_lengths,
                                              intent_labels=ada_intent_labels)

                del intent_logits
                del intent_loss_ada
                del pooled_output

                # Initialize delta_loss_x to zero
                delta_loss_x = torch.Tensor([0]).to(self.device)

                # Compute the regularization term using adaptive_weights
                # loss with respect to only classifier model
                for base_param, curr_param in zip(base_weights, adaptive_weights):
                    if curr_param.requires_grad:
                        delta_loss_x += (curr_param-base_param).pow(2).sum()

                # Total loss due to log likelihood and weight restraint
                # total_loss = self.args.beta*delta_loss_x + eg.weight.detach() * adaptive_loss
                total_loss = self.args.beta*delta_loss_x + adaptive_loss
                del delta_loss_x
                del adaptive_loss

                total_loss = total_loss.mean()
                total_loss.backward()
                adaptive_optimizer.step()


            # After adaptive_epochs steps we reach W
            # Within Batch Reptile Update
            batch_weights = {n: p for n, p in self.adaptive_model.named_parameters() \
                             if "trans_model" not in n}
            for n in batch_weights:
                within_batch_ini_weights[n] = \
                    within_batch_ini_weights[n] + self.args.beta_reptile * (within_batch_ini_weights[n] - batch_weights[n])

            torch.cuda.empty_cache()
            gc.collect()

        return self.adaptive_model

    def forward_reptile_many_batches(self, memory, q, task_id, model, dataset):

        # Using local parameters different each time for each example reinitialized deepcopy
        model_dict = model.state_dict()
        self.adaptive_model.load_state_dict(model_dict)

        if self.device != torch.device("cpu"):
            self.adaptive_model.cuda()

        # Freeze BertModel and adapt only the classifier module # key network is non-trainable in this case
        freeze_trans_get_weights(self.adaptive_model)
        self.adaptive_model.train()

        # Set an adaptive optimizer each time from scratch
        adaptive_optimizer = Adam(filter(lambda p: p.requires_grad, self.adaptive_model.parameters()),
                                  betas=(self.args.beta_1, self.args.beta_2),
                                  eps=self.args.adam_eps,
                                  lr=self.args.alpha_reptile)

        within_task_ini_weights = {n: p for n, p in model.named_parameters() if "trans_model" not in n}

        # Train the model for a few epochs over the batch retrieved from the memory
        for task in range(task_id):
            adaptive_model_dict = self.adaptive_model.state_dict()
            adaptive_model_dict.update(within_task_ini_weights)
            self.adaptive_model.load_state_dict(adaptive_model_dict)

            adaptive_batches = sample_for_q_custom_many_batches_per_task(q,
                                                                         task_id,
                                                                         memory,
                                                                         model,
                                                                         self.args.sampling_k,
                                                                         self.args.sampling_type,
                                                                         dataset,
                                                                         self.device,
                                                                         self.args.num_batches_reptile)
            for adaptive_batch in adaptive_batches:
                within_batch_ini_weights = {n: p for n, p in self.adaptive_model.named_parameters() if "trans_model" not in n}

                # We initialize our adaptive_model based on initial_weights
                adaptive_model_dict = self.adaptive_model.state_dict()
                adaptive_model_dict.update(within_batch_ini_weights)
                self.adaptive_model.load_state_dict(adaptive_model_dict)

                # Freeze BertModel and adapt only the classifier module # key network is non-trainable in this case
                adaptive_weights = get_pred_weights(self.adaptive_model)
                base_weights = get_pred_weights(model)

                # Pass the batch to the current device
                ada_input_ids, ada_lengths, ada_token_type_ids, ada_input_masks, \
                    ada_intent_labels, ada_slot_labels = adaptive_batch

                if self.device != torch.device("cpu"):
                    ada_input_ids = ada_input_ids.cuda()
                    ada_input_masks = ada_input_masks.cuda()
                    ada_lengths = ada_lengths.cuda()
                    ada_intent_labels = ada_intent_labels.cuda()
                    ada_slot_labels = ada_slot_labels.cuda()

                # Train adaptive classifier for args.adaptive_epochs epochs
                # total_loss = 0
                for _ in range(self.args.adaptive_epochs): # tqdm
                    # Step by step forward pass on the batch for adaptive_epochs epochs
                    adaptive_optimizer.zero_grad()

                    if self.args.use_slots:
                        intent_logits, slot_logits, logits_slots_, intent_loss_ada, slot_loss_ada, \
                          adaptive_loss, pooled_output = self.adaptive_model(input_ids=ada_input_ids,
                                                                             input_masks=ada_input_masks,
                                                                             train_idx=task_id,
                                                                             lengths=ada_lengths,
                                                                             intent_labels=ada_intent_labels,
                                                                             slot_labels=ada_slot_labels)

                        del slot_logits
                        del logits_slots_
                        del slot_loss_ada
                    else:
                        intent_logits, intent_loss_ada, adaptive_loss, pooled_output \
                            = self.adaptive_model(input_ids=ada_input_ids,
                                                  input_masks=ada_input_masks,
                                                  train_idx=task_id,
                                                  lengths=ada_lengths,
                                                  intent_labels=ada_intent_labels)

                    del intent_logits
                    del intent_loss_ada
                    del pooled_output

                    # Initialize delta_loss_x to zero
                    delta_loss_x = torch.Tensor([0]).to(self.device)

                    # Compute the regularization term using adaptive_weights
                    # loss with respect to only classifier model
                    for base_param, curr_param in zip(base_weights, adaptive_weights):
                        if curr_param.requires_grad:
                            delta_loss_x += (curr_param-base_param).pow(2).sum()

                    # Total loss due to log likelihood and weight restraint
                    # total_loss = self.args.beta*delta_loss_x + eg.weight.detach() * adaptive_loss
                    total_loss = self.args.beta*delta_loss_x + adaptive_loss
                    del delta_loss_x
                    del adaptive_loss

                    total_loss = total_loss.mean()
                    total_loss.backward()
                    adaptive_optimizer.step()


                # After adaptive_epochs steps we reach W
                # Within Batch Reptile Update
                batch_weights = {n: p for n, p in self.adaptive_model.named_parameters() \
                                if "trans_model" not in n}
                for n in batch_weights:
                    within_batch_ini_weights[n] \
                    = within_batch_ini_weights[n] + self.args.beta_reptile * (within_batch_ini_weights[n] - batch_weights[n])

                torch.cuda.empty_cache()
                gc.collect()

            task_weights = {n: p for n, p in self.adaptive_model.named_parameters() \
                            if "trans_model" not in n}
            for n in task_weights:
                within_task_ini_weights[n] \
                = within_task_ini_weights[n] + self.args.gamma_reptile * (within_task_ini_weights[n] - task_weights[n])

        return self.adaptive_model
    """
    OLD FORWARD
    # Using local parameters different each time for each example reinitialized deepcopy
    # adaptive_model = TransNLUCRF(args=self.args,
    #                              trans_model=self.model_trans,
    #                              num_tasks=self.num_tasks,
    #                              num_intents=self.num_intents,
    #                              eff_num_intents_task=self.eff_num_intents_task,
    #                              device=self.device,
    #                              num_slots=self.eff_num_slot)
    #
    # # Initialize adaptive_model from the model
    # model_dict = model.state_dict()
    # adaptive_model.load_state_dict(model_dict)
    # if self.device != torch.device("cpu"):
    #     adaptive_model.cuda()
    for eg in tqdm(adaptive_examples):
        ada_input_ids = torch.unsqueeze(eg.x["input_ids"], dim=0)
        ada_input_masks = torch.unsqueeze(eg.x["input_masks"], dim=0)
        ada_lengths = torch.unsqueeze(eg.x["lengths"], dim=0)
        ada_y_intents = torch.unsqueeze(eg.y_intent, dim=0)
        ada_y_slots = torch.unsqueeze(eg.y_slot, dim=0)
        ada_i_task = eg.task_id

        if self.device != torch.device("cpu"):
            ada_input_ids = ada_input_ids.cuda()
            ada_input_masks = ada_input_masks.cuda()
            ada_lengths = ada_lengths.cuda()
            ada_y_intents = ada_y_intents.cuda()
            ada_y_slots = ada_y_slots.cuda()

        # Forward pass on each item in the memory
        adaptive_optimizer.zero_grad()

        if self.args.use_slots:
            intent_logits, slot_logits, logits_slots_, intent_loss_ada, slot_loss_ada, \
              adaptive_loss, pooled_output = adaptive_model(input_ids=ada_input_ids,
                                                            input_masks=ada_input_masks,
                                                            train_idx=ada_i_task,
                                                            lengths=ada_lengths,
                                                            intent_labels=ada_y_intents,
                                                            slot_labels=ada_y_slots)

            del slot_logits
            del logits_slots_
            del slot_loss_ada

        else:
            intent_logits, intent_loss_ada, adaptive_loss, pooled_output \
                = adaptive_model(input_ids=ada_input_ids,
                                 input_masks=ada_input_masks,
                                 train_idx=ada_i_task,
                                 lengths=ada_lengths,
                                 intent_labels=ada_y_intents)

        del intent_logits
        del intent_loss_ada
        del pooled_output

        # Initialize delta_loss_x to zero
        delta_loss_x = torch.Tensor([0]).to(self.device)

        # Compute the regularization term using adaptive_weights
        # loss with respect to only classifier model
        for base_param, curr_param in zip(base_weights, adaptive_weights):
            if curr_param.requires_grad:
                delta_loss_x += (curr_param-base_param).pow(2).sum()

        # Total loss due to log likelihood and weight restraint
        total_loss = self.args.beta*delta_loss_x + eg.weight.detach() * adaptive_loss
        del delta_loss_x
        del adaptive_loss
        torch.cuda.empty_cache()
        total_loss = total_loss.mean()
        total_loss.backward()
        adaptive_optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()
    """
