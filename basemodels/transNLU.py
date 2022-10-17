import torch.nn as nn
import torch


class TransNLU(nn.Module):
    def __init__(self, 
                 trans_model, 
                 num_intents, 
                 use_slots=False, 
                 num_slots=0):
                 
        super(TransNLU, self).__init__()
        self.num_intents = num_intents
        self.config = trans_model.config
        self.trans_model = trans_model
        self.dropout = nn.Dropout(trans_model.config.hidden_dropout_prob)
        self.use_slots = use_slots

        self.intent_classifier = nn.Linear(trans_model.config.hidden_size, num_intents)
        self.intent_criterion = torch.nn.CrossEntropyLoss()

        if use_slots:
            self.slot_classifier = nn.Linear(trans_model.config.hidden_size, num_slots)
            self.slot_criterion = torch.nn.CrossEntropyLoss()

    def forward(self,
                input_ids,
                input_masks,
                train_idx,
                lengths,
                intent_labels=None,
                slot_labels=None):
        if self.training:
            self.trans_model.train()
            lm_output = self.trans_model(input_ids)
        else:
            self.trans_model.eval()
            with torch.no_grad():
                lm_output = self.trans_model(input_ids)

        cls_token = lm_output[0][:, 0, :]
        pooled_output = self.dropout(cls_token)
        logits_intents = self.intent_classifier(pooled_output)
        intent_loss = self.intent_criterion(logits_intents, intent_labels)

        loss = intent_loss

        if self.use_slots:
            logits_slots = self.slot_classifier(lm_output[0])
            slot_loss = 0
            for i in range(logits_slots.shape[1]):
                slot_loss += self.slot_criterion(logits_slots[:, i, :], slot_labels[:, i])

            loss += slot_loss

        if intent_labels is not None:
            if slot_labels is None:
                return logits_intents, intent_loss, loss

            return logits_intents, logits_slots, intent_loss, slot_loss, loss, pooled_output
        else:
            if self.use_slots:
                return intent_loss, slot_loss, loss, pooled_output

            return intent_loss

    def get_embeddings(self, input_ids, input_masks):
        """
        Computes the embeddings in eval mode
        :param input_ids:
        :param input_masks:
        :return:
        """
        self.trans_model.eval()
        with torch.no_grad():
            lm_output = self.trans_model(input_ids=input_ids,
                                         attention_mask=input_masks)

            cls_token = lm_output[0][:, 0, :]

        pooled_output = self.dropout(cls_token)


        return pooled_output
