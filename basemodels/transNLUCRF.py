import torch.nn as nn
import torch
from basemodels.transNLU import TransNLU
from basemodels.crf import CRFLayer
from utils import import_from, Results
# from transformers import AdapterConfig


class TransNLUCRF(TransNLU):
    def __init__(self,
                 args,
                 trans_model,
                 num_tasks,
                 num_labels,  # total number of intents
                 eff_num_classes_task,  # effective number of intents per task
                 device,
                 num_slots=0):

        super(TransNLUCRF, self).__init__(trans_model,
                                          num_labels,
                                          args.use_slots,
                                          num_slots)

        self.args = args
        self.use_multi_head_in = args.multi_head_in
        self.use_multi_head_out = args.multi_head_out
        self.num_tasks = num_tasks

        self.config = trans_model.config
        self.use_slots = args.use_slots
        self.use_adapters = args.use_adapters

        self.trans_model = trans_model

        self.num_labels = num_labels
        self.eff_num_classes_task = eff_num_classes_task

        self.device = device

        self.use_slots = args.use_slots
        self.num_slots = num_slots

        self.dropout = nn.Dropout(trans_model.config.hidden_dropout_prob)

        if "cil" in self.args.setup_opt and self.use_multi_head_out:
            self.gclassifier = nn.ModuleList([nn.Linear(trans_model.config.hidden_size, self.eff_num_classes_task[i])
                                                    for i in range(self.num_tasks)])
        else:
            self.gclassifier = nn.Linear(trans_model.config.hidden_size, self.num_labels)

        self.criterion = torch.nn.CrossEntropyLoss()

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
                labels=None,
                slot_labels=None):
        """
        Forward pass
        :param input_ids:
        :param input_masks:
        :param train_idx:
        :param lengths:
        :param labels:
        :param slot_labels:
        :return:
        """

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
                if str(i) in self.adapter_layers.split("_"):
                    output_hidden = self.adapter(output_hidden)[0]

                hidden_outputs.append(output_hidden)

            stacked_tensor = torch.stack(hidden_outputs)
            avg_pooled = torch.mean(stacked_tensor, axis=0)

            cls_token = avg_pooled[:, 0, :]
        else:
            cls_token = lm_output[0][:, 0, :]
        pooled_output = self.dropout(cls_token)

        output = Results(pool_out=pooled_output, 
                         hidden_states=lm_output.hidden_states, 
                         attentions=lm_output.attentions)

        if "cil" in self.args.setup_opt and self.use_multi_head_out:
            logits_intents = self.gclassifier[train_idx](pooled_output)
        else:
            logits_intents = self.gclassifier(pooled_output)

        intent_loss = self.criterion(logits_intents, labels)

        output.add_loss("intent_class", intent_loss)
        output.add_logits("class", logits_intents.to(torch.device("cpu")))

        loss = intent_loss

        if self.use_slots:
            logits_slots_nocrf = self.slot_classifier(lm_output[0])
            slot_loss = self.crf_layer.loss(logits_slots_nocrf, slot_labels)

            loss += slot_loss

            logits_slots = self.crf_decode(logits_slots_nocrf, lengths)

            output.add_loss("slots", slot_loss)
            output.add_logits("slots_nokd", logits_slots.to(torch.device("cpu")))
            output.add_logits("slots_nocrf", logits_slots_nocrf.to(torch.device("cpu")))

        output.add_loss("overall", loss)

        return output 

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
        
