import torch.nn as nn
import torch
from basemodels.crf import CRFLayer
from utils import import_from

SLOT_PAD = 0


def _make_mask(lengths):
    bsz = len(lengths)
    max_len = torch.max(lengths)
    mask = torch.LongTensor(bsz, max_len).fill_(1)
    for i in range(bsz):
        length = lengths[i]
        mask[i, length:max_len] = 0
    return mask


def _pad_label(lengths, y):
    bsz = len(lengths)
    max_len = torch.max(lengths)
    padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
    for i in range(bsz):
        y_i = y[i]
        padded_y[i, 0:] = y_i

    return padded_y


class TransNLUCRF(nn.Module):
    def __init__(self,
                 args,
                 trans_model,
                 num_tasks,
                 num_intents,  # total number of intents
                 eff_num_intents_task,  # effective number of intents per task
                 device,
                 num_slots=0):

        super(TransNLUCRF, self).__init__()

        self.args = args
        self.use_multi_head_in = args.multi_head_in
        self.use_multi_head_out = args.multi_head_out
        self.num_tasks = num_tasks

        self.config = trans_model.config
        self.use_slots = args.use_slots
        self.use_adapters = args.use_adapters

        self.trans_model = trans_model

        self.num_intents = num_intents
        self.eff_num_intents_task = eff_num_intents_task

        self.device = device

        # model_name, tokenizer_alias, model_trans_alias, _ = MODELS_dict[args.trans_model]
        #
        # self.trans_model = model_trans_alias.from_pretrained(os.path.join(args.model_root, model_name))

        self.use_slots = args.use_slots
        self.num_slots = num_slots

        self.dropout = nn.Dropout(trans_model.config.hidden_dropout_prob)

        if "cil" in self.args.setup_opt and self.use_multi_head_out:
            self.intent_classifier = nn.ModuleList([nn.Linear(trans_model.config.hidden_size, self.eff_num_intents_task[i])
                                                    for i in range(self.num_tasks)])
        else:
            self.intent_classifier = nn.Linear(trans_model.config.hidden_size, self.num_intents)

        self.intent_criterion = torch.nn.CrossEntropyLoss()

        if self.use_slots:
            self.slot_classifier = nn.Linear(trans_model.config.hidden_size, self.num_slots)

            self.crf_layer = CRFLayer(self.num_slots, self.device)

        if self.use_adapters:
            Adapter = import_from('contlearnalg.adapter', args.adapter_type)
            self.adapter = Adapter(trans_model.config.hidden_size)  # (16, 24, 768)

            self.adapter_layers = args.adapter_layers

    def forward(self,
                input_ids,
                input_masks,
                train_idx,
                lengths,
                intent_labels=None,
                slot_labels=None):

        if self.training:
            self.trans_model.train()
            self.crf_layer.train()
            lm_output = self.trans_model(input_ids=input_ids,
                                         attention_mask=input_masks)
        else:
            self.trans_model.eval()
            self.crf_layer.eval()
            with torch.no_grad():
                lm_output = self.trans_model(input_ids=input_ids,
                                             attention_mask=input_masks)

        if self.use_adapters:
            hidden_outputs = []
            embeddings = lm_output[2][0]  # embeddings
            hidden_outputs.append(embeddings)
            for i in range(1, self.trans_model.config.num_hidden_layers+1):
                output_hidden = lm_output[2][i]
                if str(i) in self.adapter_layers:
                    output_hidden = self.adapter(output_hidden)[0]

                hidden_outputs.append(output_hidden)

            stacked_tensor = torch.stack(hidden_outputs)
            avg_pooled = torch.mean(stacked_tensor, axis=0)

            cls_token = avg_pooled[:, 0, :]
        else:
            cls_token = lm_output[0][:, 0, :]
        pooled_output = self.dropout(cls_token)

        if "cil" in self.args.setup_opt and self.use_multi_head_out:
            logits_intents = self.intent_classifier[train_idx](pooled_output)
        else:
            logits_intents = self.intent_classifier(pooled_output)

        intent_loss = self.intent_criterion(logits_intents, intent_labels)
        loss = intent_loss

        if self.use_slots:
            logits_slots = self.slot_classifier(lm_output[0])
            slot_loss = self.crf_layer.loss(logits_slots, slot_labels)

            loss += slot_loss

            logits_slots = self.crf_decode(logits_slots, lengths)

        if intent_labels is not None:
            if slot_labels is None:
                return logits_intents, intent_loss, loss

            return logits_intents, logits_slots, intent_loss, slot_loss, loss
        else:
            if self.use_slots:
                return intent_loss, slot_loss, loss

            return intent_loss, loss

    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = _pad_label(lengths, y)
        mask = _make_mask(lengths)

        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_decode: the predictions
        """

        prediction = self.crf_layer(inputs)

        return prediction
