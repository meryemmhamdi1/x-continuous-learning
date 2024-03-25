import torch.nn as nn
import torch

import sys

sys.path.append("/home1/mmhamdi/x-continuous-learning")
from src.utils import import_from, Results


class TransModel(nn.Module):
    """
    A Transformers based model for named entity recognition.
    Args:
        :param trans_model: transformers
            Transformers pre-trained model.
        :param args: argparse dictionary
            Different options/hyperparameters fed to run this code.
        :param device: torch.device
            CPU or CUDA (if CUDA which cuda machine used).

    """

    def __init__(self, trans_model, args, device):
        super(TransModel, self).__init__()

        self.trans_model = trans_model
        self.config = trans_model.config
        self.args = args
        self.num_labels = args.num_labels  # total number of labels
        self.setup_opt = args.setup_opt
        self.device = device

        # For CIL Architectures
        self.num_tasks = args.num_tasks
        self.eff_num_classes_task = (
            args.eff_num_classes_task
        )  # effective number of labels per task

        # For Multi-headed Architectures
        self.use_multi_head_out = args.multi_head_out
        self.use_adapters = args.use_adapters

        self.dropout = nn.Dropout(trans_model.config.hidden_dropout_prob)

        if "cil" in self.setup_opt and self.use_multi_head_out:
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

        self.criterion = torch.nn.CrossEntropyLoss()

        if self.use_adapters:
            Adapter = import_from("contlearnalg.adapter", args.adapter_type)
            self.adapter = Adapter(trans_model.config.hidden_size)

            self.adapter_layers = args.adapter_layers

    def forward(
        self,
        input_ids,
        input_masks,
        token_type_ids,
        train_idx,
        labels=None,
        lengths=None,
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
                input_ids=input_ids,
                attention_mask=input_masks,
            )
        else:
            self.trans_model.eval()
            with torch.no_grad():
                lm_output = self.trans_model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
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

            stacked_tensor = torch.stack(hidden_outputs)
            avg_pooled_seq = torch.mean(stacked_tensor, axis=0)

            token_encs = avg_pooled_seq
        else:
            token_encs = lm_output[0]
        seq_output = self.dropout(token_encs)

        output = Results(
            pool_out=seq_output,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
        )

        if "cil" in self.setup_opt and self.use_multi_head_out:
            logits = self.gclassifier[train_idx](seq_output)
        else:
            logits = self.gclassifier(seq_output)

        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))

        output.add_loss("nre_tag", loss)
        output.add_logits("tags", logits.to(torch.device("cpu")))

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
            lm_output = self.trans_model(
                input_ids=input_ids, attention_mask=input_masks
            )

            cls_token = lm_output[0][:, 0, :]

        pooled_output = self.dropout(cls_token)

        return pooled_output
