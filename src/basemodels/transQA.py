import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import sys
from typing import Optional

import os

sys.path.append(os.getcwd())

from src.utils import import_from, Results

# from transformers import AdapterConfig


class TransModel(nn.Module):
    def __init__(self, trans_model, args, device):
        super(TransModel, self).__init__()

        self.args = args
        self.trans_model = trans_model
        self.config = trans_model.config

        self.use_adapters = args.use_adapters
        self.use_multi_head_in = args.multi_head_in
        self.use_multi_head_out = args.multi_head_out

        self.num_tasks = args.num_tasks
        self.num_labels = args.num_labels  # total number of labels
        self.eff_num_classes_task = (
            args.eff_num_classes_task
        )  # effective number of labels per task

        self.device = device

        self.dropout = nn.Dropout(trans_model.config.hidden_dropout_prob)

        if "cil" in self.args.setup_opt and self.use_multi_head_out:
            self.gclassifier = nn.ModuleList(
                [
                    nn.Linear(
                        trans_model.config.hidden_size, self.eff_num_classes_task[i]
                    )
                    for i in range(self.num_tasks)
                ]
            )
        else:
            self.gclassifier = nn.Linear(
                trans_model.config.hidden_size, self.num_labels
            )

        if self.use_adapters:
            Adapter = import_from("contlearnalg.adapter", args.adapter_type)
            self.adapter = Adapter(trans_model.config.hidden_size)
            self.adapter_layers = args.adapter_layers

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_masks: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        train_idx: Optional[int] = None,
    ):
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
            lm_output = self.trans_model(
                input_ids=input_ids, attention_mask=input_masks
            )
        else:
            self.trans_model.eval()
            with torch.no_grad():
                lm_output = self.trans_model(
                    input_ids=input_ids, attention_mask=input_masks
                )

        if self.use_adapters:
            hidden_outputs = []
            embeddings = lm_output[2][0]  # embeddings
            hidden_outputs.append(embeddings)
            for i in range(1, self.trans_model.config.num_hidden_layers + 1):
                output_hidden = lm_output[2][i]
                if str(i) in self.adapter_layers.split("_"):
                    output_hidden = self.adapter(output_hidden)[0]

                hidden_outputs.append(output_hidden)

            sequence_outputs = torch.stack(hidden_outputs)
        else:
            sequence_outputs = lm_output[
                0
            ]  # last_hidden_state => 0 , pooler_output => 1

        results = Results(
            pool_out=sequence_outputs,  # TODO VERIFY THIS
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
        )

        if "cil" in self.args.setup_opt and self.use_multi_head_out:
            logits = self.gclassifier[train_idx](sequence_outputs)
        else:
            logits = self.gclassifier(sequence_outputs)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # TODO handle this case
        # if not return_dict:
        #     output = (start_logits, end_logits) + lm_output[2:]
        #     return ((total_loss,) + output) if total_loss is not None else output

        results.add_loss("overall", total_loss)
        results.add_logits("start_positions", start_logits)
        results.add_logits("end_positions", end_logits)

        return results

    def get_embeddings(self, input_ids, input_masks):
        """
        Computes the embeddings in eval mode
        :param input_ids:
        :param input_masks:
        :return:
        """
        self.trans_model.eval()
        with torch.no_grad():
            lm_output = self.trans_model(
                input_ids=input_ids, attention_mask=input_masks
            )

            cls_token = lm_output[0][:, 0, :]

        pooled_output = self.dropout(cls_token)

        return pooled_output
