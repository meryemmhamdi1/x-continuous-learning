import torch
import gc


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
                intent_logits, slot_logits, logits_slots_, intent_loss_er, slot_loss_er, loss_er, pooled_output \
                    = model(input_ids=er_input_ids,
                            input_masks=er_input_masks,
                            train_idx=er_i_task,
                            lengths=er_lengths,
                            intent_labels=er_y_intents,
                            slot_labels=er_y_slots)

                # slot_logits = slot_logits.to(torch.device("cpu"))
                # logits_slots_ = logits_slots_.to(torch.device("cpu"))

                self.writer.add_scalar('memory_er_intent_loss_'+str(i_task), intent_loss_er.mean(), num_steps*epoch)
                self.writer.add_scalar('memory_er_slot_loss_'+str(i_task), slot_loss_er.mean(), num_steps*epoch)
            else:
                intent_logits, intent_loss_er, loss_er, pooled_output = model(input_ids=er_input_ids,
                                                                              input_masks=er_input_masks,
                                                                              train_idx=er_i_task,
                                                                              lengths=er_lengths,
                                                                              intent_labels=er_y_intents)

                self.writer.add_scalar('memory_er_intent_loss_'+str(i_task), intent_loss_er.mean(), num_steps*epoch)

            # pooled_output = pooled_output.to(torch.device("cpu"))
            # intent_logits = intent_logits.to(torch.device("cpu"))

            loss_er = loss_er.mean()
            loss_er.backward()
            optimizer.step()

            del pooled_output
            del intent_logits
            if self.args.use_slots:
                del slot_logits
                del logits_slots_

            torch.cuda.empty_cache()  # TODO try this emptying of cache after backward
            gc.collect()
