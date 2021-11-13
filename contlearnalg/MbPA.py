import torch
from torch.optim import Adam

# from copy import deepcopy
from basemodels.transNLUCRF import TransNLUCRF
from tqdm import tqdm


def freeze_trans_get_weights(model):
    for n, p in model.named_parameters():
        if p.requires_grad and "trans_model" in n:
            p.requires_grad = False

    return list(model.parameters())


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

    def forward(self, memory, q, train_idx, model):
        # Using local parameters different each time for each example reinitialized deepcopy
        adaptive_model = TransNLUCRF(args=self.args,
                                     trans_model=self.model_trans,
                                     num_tasks=self.num_tasks,
                                     num_intents=self.num_intents,
                                     eff_num_intents_task=self.eff_num_intents_task,
                                     device=self.device,
                                     num_slots=self.eff_num_slot)

        # Initialize adaptive_model from the model
        model_dict = model.state_dict()
        adaptive_model.load_state_dict(model_dict)
        if self.device != torch.device("cpu"):
            adaptive_model.cuda()
        # Using local parameters different each time for each example reinitialized deepcopy
        # adaptive_model = deepcopy(model)
        adaptive_examples = memory.sample_for_q(q, train_idx, self.args.sampling_k)

        adaptive_model.train()

        # Freeze BertModel and adapt only the classifier module # key network is non-trainable in this case
        adaptive_weights = freeze_trans_get_weights(adaptive_model)
        base_weights = freeze_trans_get_weights(model)

        # Set an adaptive optimizer each time from scratch
        adaptive_optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  betas=(self.args.beta_1, self.args.beta_2),
                                  eps=self.args.adam_eps,
                                  lr=self.args.adaptive_adam_lr)

        # Train the model for a few epochs over the batch retrieved from the memory
        for _ in tqdm(range(self.args.adaptive_epochs)):
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
                    _, _, _,_, _, adaptive_loss, _ = adaptive_model(input_ids=ada_input_ids,
                                                                  input_masks=ada_input_masks,
                                                                  train_idx=ada_i_task,
                                                                  lengths=ada_lengths,
                                                                  intent_labels=ada_y_intents,
                                                                  slot_labels=ada_y_slots)

                else:
                    _, _, adaptive_loss, _ = adaptive_model(input_ids=ada_input_ids,
                                                            input_masks=ada_input_masks,
                                                            train_idx=ada_i_task,
                                                            lengths=ada_lengths,
                                                            intent_labels=ada_y_intents)

                # Initialize delta_loss_x to zero
                delta_loss_x = torch.Tensor([0]).to(self.device)

                # Compute the regularization term using adaptive_weights
                # loss with respect to only classifier model
                for base_param, curr_param in zip(base_weights, adaptive_weights):
                    if curr_param.requires_grad:
                        delta_loss_x += (curr_param-base_param).pow(2).sum()

                # Total loss due to log likelihood and weight restraint
                total_loss = self.args.beta*delta_loss_x + eg.weight.detach() * adaptive_loss
                total_loss = total_loss.mean()
                total_loss.backward()

                adaptive_optimizer.step()

        return adaptive_model
