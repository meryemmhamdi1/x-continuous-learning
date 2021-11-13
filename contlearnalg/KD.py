import torch
import torch.nn.functional as F

class KD(object):
    def __init__(self,
                 args,
                 device,
                 writer,
                 kd_mode):
        self.args = args
        self.device = device
        self.writer = writer
        self.kd_mode = kd_mode

    def forward(self,
                memory,
                model,
                optimizer,
                i_task,
                epoch,
                num_steps,
                main_intent_logits,
                main_slot_logits,
                main_embs):
        # Randomly sample from the memory
        sampled_to_replay = memory.sample(i_task, self.args.sampling_k)  # sampling_k should be the same as batch_size

        memory_embs = []
        memory_intent_logits = []
        memory_slot_logits = []

        # Forward pass over the memory to find
        for eg in sampled_to_replay:
            optimizer.zero_grad()
            er_input_ids = torch.unsqueeze(eg.x["input_ids"], dim=0)
            er_input_masks = torch.unsqueeze(eg.x["input_masks"], dim=0)
            er_lengths = torch.unsqueeze(eg.x["lengths"], dim=0)
            er_y_intents = torch.unsqueeze(eg.y_intent, dim=0)
            er_y_slots = torch.unsqueeze(eg.y_slot, dim=0)
            er_i_task = eg.task_id

            if self.device != torch.device("cpu"):
                er_input_ids = er_input_ids.cuda()
                er_input_masks = er_input_masks.cuda()  # TODO compare when you use this versus when you don't
                er_lengths = er_lengths.cuda()
                er_y_intents = er_y_intents.cuda()
                er_y_slots = er_y_slots.cuda()

            if self.args.use_slots:
                intent_logits, slot_logits, logits_slots_, intent_loss_er, slot_loss_er, loss_er, pooled_outputs \
                    = model(input_ids=er_input_ids,
                            input_masks=er_input_masks,
                            train_idx=er_i_task,
                            lengths=er_lengths,
                            intent_labels=er_y_intents,
                            slot_labels=er_y_slots)

                self.writer.add_scalar('memory_kd_intent_loss_'+str(i_task), intent_loss_er.mean(), num_steps*epoch)
                self.writer.add_scalar('memory_kd_slot_loss_'+str(i_task), slot_loss_er.mean(), num_steps*epoch)
            else:
                intent_logits, intent_loss_er, loss_er, pooled_outputs = model(input_ids=er_input_ids,
                                                                               input_masks=er_input_masks,
                                                                               train_idx=er_i_task,
                                                                               lengths=er_lengths,
                                                                               intent_labels=er_y_intents)

                self.writer.add_scalar('memory_kd_intent_loss_'+str(i_task), intent_loss_er.mean(), num_steps*epoch)

            memory_embs.append(pooled_outputs)

            memory_intent_logits.append(intent_logits)
            memory_slot_logits.append(logits_slots_)

            # loss_er = loss_er.mean()
            # loss_er.backward()
            # optimizer.step()

        if self.kd_mode == "logits":
            memory_intent_logits = torch.squeeze(torch.stack(memory_intent_logits)) #, dtype=float
            memory_slot_logits = torch.squeeze(torch.stack(memory_slot_logits)) # , dtype=float

            kd_intent_loss = F.mse_loss(main_intent_logits.float(), memory_intent_logits.float())
            # kd_intent_loss = torch.nn.KLDivLoss(size_average=False)(main_intent_logits, memory_intent_logits)
            kd_slot_loss = F.mse_loss(main_slot_logits, memory_slot_logits)
            # kd_slot_loss = torch.nn.KLDivLoss(size_average=False)(main_slot_logits, memory_slot_logits)
            kd_loss = kd_intent_loss + kd_slot_loss
        else:  # rep
            memory_embs = torch.stack(memory_embs)
            kd_loss = torch.nn.MSELoss(main_embs, memory_embs)

        return kd_loss









