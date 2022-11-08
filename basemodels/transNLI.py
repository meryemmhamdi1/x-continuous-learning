import torch.nn as nn
import torch
from utils import import_from, Results
# from transformers import AdapterConfig


class TransNLI(nn.Module):
    def __init__(self,
                 args,
                 trans_model,
                 num_tasks,
                 num_labels,  # total number of labels
                 eff_num_classes_task,  # effective number of labels per task
                 device):

        super(TransNLI, self).__init__()

        self.args = args
        self.use_multi_head_in = args.multi_head_in
        self.use_multi_head_out = args.multi_head_out
        self.num_tasks = num_tasks

        self.config = trans_model.config
        self.use_adapters = args.use_adapters

        self.trans_model = trans_model

        self.num_labels = num_labels
        self.eff_num_classes_task = eff_num_classes_task

        self.device = device

        self.dropout = nn.Dropout(trans_model.config.hidden_dropout_prob)

        if "cil" in self.args.setup_opt and self.use_multi_head_out:
            self.gclassifier = nn.ModuleList([nn.Linear(trans_model.config.hidden_size, self.eff_num_classes_task[i])
                                                    for i in range(self.num_tasks)])
        else:
            self.gclassifier = nn.Linear(trans_model.config.hidden_size, self.num_labels)

        self.criterion = torch.nn.CrossEntropyLoss()

        if self.use_adapters:
            Adapter = import_from('contlearnalg.adapter', args.adapter_type)
            self.adapter = Adapter(trans_model.config.hidden_size)

            self.adapter_layers = args.adapter_layers

    def forward(self,
                input_ids,
                input_masks,
                token_type_ids,
                train_idx,
                labels=None):
        """
        Forward pass
        :param input_ids:
        :param input_masks:
        :param train_idx:
        :param labels:
        :return:
        """

        if self.training:
            self.trans_model.train()
            lm_output = self.trans_model(input_ids=input_ids,
                                         attention_mask=input_masks,
                                         token_type_ids=token_type_ids)
        else:
            self.trans_model.eval()
            with torch.no_grad():
                lm_output = self.trans_model(input_ids=input_ids,
                                             attention_mask=input_masks,
                                             token_type_ids=token_type_ids)

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
            logits = self.gclassifier[train_idx](pooled_output)
        else:
            logits = self.gclassifier(pooled_output)

        loss = self.criterion(logits, labels)

        output.add_loss("nli_class", loss)
        output.add_logits("class", logits.to(torch.device("cpu")))
        
        output.add_loss("overall", loss)

        return output


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
