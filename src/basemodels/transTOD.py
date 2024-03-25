import torch.nn as nn
import torch, sys

import os

sys.path.append(os.getcwd())

from src.basemodels.crf import CRFLayer
from src.utils import import_from, Results


class TransModel(nn.Module):
    """
    A Transformers based model for task-oriented dialog (that performs both intent classification and if use_slots slot filling with or without CRF).
    Args:
        :param trans_model: transformers
            Transformers pre-trained model.
        :param args: argparse dictionary
            Different options/hyperparameters fed to run this code.
        :param device: torch.device
            CPU or CUDA (if CUDA which cuda machine used).
    """

    def __init__(self, trans_model, args, device):
        """
        :param trans_model: transformers
            Transformers pre-trained model.
        :param num_labels: int
            The total number of intents.
        :param num_slots: int
            The number of slots to be used (useful in initializing the sequence labelling layer).
        :param num_tasks: int
            For CIL (class incremental learning) architectures only: number of groups of classes.
        :param eff_num_classes_task: int
            For CIL architectures only: effective number of intents per task.
        :param args: argparse dictionary
            Different options/hyperparameters fed to run this code.
        :param device: torch.device
            CPU or CUDA (if CUDA which cuda machine used).
        """

        super(TransModel, self).__init__()

        self.trans_model = trans_model
        self.config = trans_model.config
        self.setup_opt = args.setup_opt

        self.num_labels = args.num_labels
        self.use_slots = args.use_slots
        self.use_crf = args.use_crf
        self.num_slots = args.num_slots

        self.device = device

        # For CIL Architectures
        self.num_tasks = args.num_tasks
        self.eff_num_classes_task = args.eff_num_classes_task

        # For Multi-headed Architectures
        self.use_multi_head_out = args.multi_head_out
        self.use_adapters = args.use_adapters

        # LAYERS
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

        if self.use_slots:
            self.slot_classifier = nn.Linear(
                trans_model.config.hidden_size, self.num_slots
            )
            self.crf_layer = CRFLayer(self.num_slots, self.device)
            self.slot_criterion = torch.nn.CrossEntropyLoss()

        if self.use_adapters:
            Adapter = import_from("contlearnalg.adapter", args.adapter_type)
            self.adapter = Adapter(trans_model.config.hidden_size)  # (16, 24, 768)

            self.adapter_layers = args.adapter_layers

    def forward(
        self,
        input_ids,
        input_masks,
        token_type_ids,
        train_idx,
        lengths=None,
        labels=None,
        slot_labels=None,
    ):
        """
        Forward pass and evaluation pass at the same time
        :param input_ids: batch_size x sequence_length torch long tensor
            indices of input sequence tokens in the vocabulary.
            They are numerical representations of tokens that build the input sequence.
        :param input_masks: batch_size x sequence_length torch long tensor
            allows the model to cleanly differentiate between the content and the padding.
            # N.B. no token_type_ids are used since those are binary masks identifying two types
              of sequence in the model and we have only one sequence
        :param train_idx: int
            Used only for CIL architectures indicating
        :param labels:
            intent labels used to compute the loss
        :param slot_labels:
            slot labels used to compute the CRF loss
        :return: output results: of instance Results
        """

        # 1. Forward pass over transformers encoder model with or without gradients
        if self.training:
            self.trans_model.train()
            self.crf_layer.train()
            lm_output = self.trans_model(
                input_ids=input_ids, attention_mask=input_masks
            )
        else:
            self.trans_model.eval()
            self.crf_layer.eval()
            with torch.no_grad():
                lm_output = self.trans_model(
                    input_ids=input_ids, attention_mask=input_masks
                )

        # If adapters are used
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
            avg_pooled = torch.mean(stacked_tensor, axis=0)

            cls_token = avg_pooled[:, 0, :]
        else:
            # Otherwise, just output the average pooled on top of the CLS token
            cls_token = lm_output[0][:, 0, :]

        # 2. Apply the dropout layer
        pooled_output = self.dropout(cls_token)

        if "cil" in self.setup_opt and self.use_multi_head_out:
            logits_intents = self.gclassifier[train_idx](pooled_output)
        else:
            logits_intents = self.gclassifier(pooled_output)

        intent_loss = self.criterion(logits_intents, labels)

        output = Results(
            pool_out=pooled_output,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
        )

        output.add_loss("intent_class", intent_loss)
        output.add_logits("class", logits_intents.to(torch.device("cpu")))

        loss = intent_loss

        if self.use_slots:
            # Slot logits without CRF
            logits_slots = self.slot_classifier(lm_output[0])

            # Slot loss WITHOUT or WITH CRF
            if not self.use_crf:
                print("With CRF")
                slot_loss = 0
                for i in range(logits_slots.shape[1]):
                    slot_loss += self.slot_criterion(
                        logits_slots[:, i, :], slot_labels[:, i]
                    )

            else:
                slot_loss = self.crf_layer.loss(logits_slots, slot_labels)

            output.add_loss("slots", slot_loss)

            # Slot logits WITH CRF
            if self.use_crf:
                logits_slots = self.crf_decode(logits_slots)

            output.add_logits("slots_nokd", logits_slots.to(torch.device("cpu")))
            loss += slot_loss

        output.add_loss("overall", loss)

        return output

    def crf_decode(self, inputs):
        """crf decode
        :param inputs: (bsz, seq_len, num_slot) torch tensor
            output of SlotPredictor
        :return predictions:
            crf_decode predictions
        """

        predictions = self.crf_layer(inputs)

        return predictions

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
