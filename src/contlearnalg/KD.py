import torch
import torch.nn.functional as F
import gc


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
                main_intent_logits=None,
                main_slot_logits=None,
                main_embs=None):
        # Randomly sample from the memory
        sampled_to_replay = memory.sample(i_task, self.args.sampling_k)  # sampling_k should be the same as batch_size

        memory_embs = []
        memory_intent_logits = []
        memory_slot_logits = []

        # Forward pass over the memory to find
        for eg in sampled_to_replay:
            gc.collect()
            optimizer.zero_grad()
            kd_input_ids = torch.unsqueeze(eg.x["input_ids"], dim=0)
            kd_input_masks = torch.unsqueeze(eg.x["input_masks"], dim=0)
            kd_lengths = torch.unsqueeze(eg.x["lengths"], dim=0)
            kd_y_intents = torch.unsqueeze(eg.y_intent, dim=0)
            kd_y_slots = torch.unsqueeze(eg.y_slot, dim=0)
            kd_i_task = eg.task_id

            if self.device != torch.device("cpu"):
                kd_input_ids = kd_input_ids.cuda()
                kd_input_masks = kd_input_masks.cuda()  # TODO compare when you use this versus when you don't
                kd_lengths = kd_lengths.cuda()
                kd_y_intents = kd_y_intents.cuda()
                kd_y_slots = kd_y_slots.cuda()

            if self.args.use_slots:
                intent_logits, slot_logits, logits_slots_, intent_loss_kd, slot_loss_kd, loss_kd, pooled_outputs \
                    = model(input_ids=kd_input_ids,
                            input_masks=kd_input_masks,
                            train_idx=kd_i_task,
                            lengths=kd_lengths,
                            intent_labels=kd_y_intents,
                            slot_labels=kd_y_slots)

                self.writer.add_scalar('memory_kd_slot_loss_'+str(kd_i_task), slot_loss_kd.mean(), num_steps*epoch)
            else:
                intent_logits, intent_loss_kd, loss_er, pooled_outputs = model(input_ids=kd_input_ids,
                                                                               input_masks=kd_input_masks,
                                                                               train_idx=kd_i_task,
                                                                               lengths=kd_lengths,
                                                                               intent_labels=kd_y_intents)

            self.writer.add_scalar('memory_kd_intent_loss_'+str(kd_i_task), intent_loss_kd.mean(), num_steps*epoch)

            if self.kd_mode == "logits":
                max_seq_len = main_slot_logits.shape[1]
                logits_slot_len = main_slot_logits.shape[-1]
                # store logits and shape them
                memory_intent_logits.append(intent_logits)

                logits_slots_zeros = torch.zeros(1, max_seq_len, logits_slot_len)
                min_seq_len = min(max_seq_len, logits_slots_.shape[1])
                logits_slots_zeros[:, :min_seq_len, :] = logits_slots_[:, :min_seq_len, :]
                memory_slot_logits.append(logits_slots_zeros)

                # put the embeddings into cpu and delete them and free the cache because they are not needed at all
                del pooled_outputs
                torch.cuda.empty_cache()

            else:  # rep
                # store embeddings
                memory_embs.append(pooled_outputs)

                # put the logits into cpu and delete them and free the cache because they are not needed at all
                del intent_logits
                if self.args.use_slots:
                    del slot_logits
                    del logits_slots_

                torch.cuda.empty_cache()

        if self.kd_mode == "logits":
            min_num_sent = min(main_intent_logits.shape[0], len(memory_intent_logits))
            memory_intent_logits = torch.squeeze(torch.stack(memory_intent_logits[:min_num_sent])).to(self.device)  # , dtype=float
            memory_slot_logits = torch.squeeze(torch.stack(memory_slot_logits[:min_num_sent])).to(self.device)  # , dtype=float

            kd_intent_loss = F.mse_loss(main_intent_logits[:min_num_sent].to(self.device).float(),
                                        memory_intent_logits.float())

            # kd_intent_loss = torch.nn.KLDivLoss(size_average=False)(main_intent_logits, memory_intent_logits)
            kd_slot_loss = F.mse_loss(main_slot_logits[:min_num_sent].to(self.device).float(),
                                      memory_slot_logits.float())

            # kd_slot_loss = torch.nn.KLDivLoss(size_average=False)(main_slot_logits, memory_slot_logits)
            kd_loss = kd_intent_loss + kd_slot_loss

            del main_intent_logits
            del memory_intent_logits
            del main_slot_logits
            del memory_slot_logits

            torch.cuda.empty_cache()
        else:  # rep
            min_num_sent = min(main_embs.shape[0], len(memory_embs))
            memory_embs = torch.stack(memory_embs).to(self.device)
            kd_loss = F.mse_loss(main_embs[:min_num_sent].to(self.device).float(),
                                 memory_embs.float())

            del memory_embs
            del main_embs
            torch.cuda.empty_cache()

        return kd_loss









