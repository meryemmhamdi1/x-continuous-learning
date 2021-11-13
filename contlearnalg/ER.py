import torch


class ER(object):
    def __init__(self,
                 args,
                 device,
                 writer):
        self.args = args
        self.device = device
        self.writer = writer

    def forward(self, memory, model, optimizer, i_task, epoch, num_steps):
        # Randomly sample from the memory
        sampled_to_replay = memory.sample(i_task, self.args.sampling_k)

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
                _, _, _, intent_loss_er, slot_loss_er, loss_er, _ \
                    = model(input_ids=er_input_ids,
                            input_masks=er_input_masks,
                            train_idx=er_i_task,
                            lengths=er_lengths,
                            intent_labels=er_y_intents,
                            slot_labels=er_y_slots)

                self.writer.add_scalar('memory_er_intent_loss_'+str(i_task), intent_loss_er.mean(), num_steps*epoch)
                self.writer.add_scalar('memory_er_slot_loss_'+str(i_task), slot_loss_er.mean(), num_steps*epoch)
            else:
                _, intent_loss_er, loss_er, _ = model(input_ids=er_input_ids,
                                                      input_masks=er_input_masks,
                                                      train_idx=er_i_task,
                                                      lengths=er_lengths,
                                                      intent_labels=er_y_intents)

                self.writer.add_scalar('memory_er_intent_loss_'+str(i_task), intent_loss_er.mean(), num_steps*epoch)

            loss_er = loss_er.mean()
            loss_er.backward()
            optimizer.step()
