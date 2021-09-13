import torch.nn as nn
import torch

SLOT_PAD = 0


class TransNLU(nn.Module):
    def __init__(self, trans_model, num_intents, use_slots=False, num_slots=0):
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

    def forward(self, input_ids, lengths, input_masks, intent_labels=None, slot_labels=None):
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

        if self.use_slots:
            logits_slots = self.slot_classifier(lm_output[0])
            slot_loss = 0
            for i in range(logits_slots.shape[1]):
                slot_loss += self.slot_criterion(logits_slots[:, i, :], slot_labels[:, i])

        if intent_labels is not None:
            if slot_labels is None:
                return logits_intents, intent_loss

            return logits_intents, logits_slots, intent_loss, slot_loss
        else:
            if self.use_slots:
                return intent_loss, slot_loss

            return intent_loss

    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths)]

        return prediction

    def make_mask(self, lengths):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        mask = torch.LongTensor(bsz, max_len).fill_(1)
        for i in range(bsz):
            length = lengths[i]
            mask[i, length:max_len] = 0
        return mask

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            y_i = y[i]
            padded_y[i, 0:] = torch.LongTensor(y_i)

        return padded_y

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())
