# Credits Taken from https://github.com/Adapter-Hub/adapter-transformers/blob/19e9d85225446424d8623bd6244a90f58c37bd86/src/transformers/adapters/modeling.py#L50

import math

import torch.nn as nn
import torch


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "gelu_orig":
            self.f = nn.functional.gelu
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)


class MADX(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(
            self,
            input_size,
            down_sample=None,
            non_linearity="relu",
            init_bert_weights=True,
            add_layer_norm_before=True,
            add_layer_norm_after=False,
            residual_before_ln=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln

        self.embed_dim = input_size

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.embed_dim // 2

        # Linear down projection of the input
        seq_list.append(nn.Linear(self.embed_dim, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down_1 = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up_1 = nn.Linear(self.down_sample, self.embed_dim)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after_1 = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if init_bert_weights:
            self.adapter_down_1.apply(self.init_bert_weights)
            self.adapter_up_1.apply(self.init_bert_weights)

    def forward(self, x, residual_input=1.0):  # , residual_input=None):
        # print("x.shape:", x.shape)
        # print("self.adapter_down:", self.adapter_down)
        #
        # print("self.adapter_norm_before:", self.adapter_norm_before)
        #
        # print("self.adapter_norm_before(x):", self.adapter_norm_before(x))
        down = self.adapter_down_1(x)

        up = self.adapter_up_1(down)

        output = up

        # apply residual connection before layer norm if configured in this way
        if self.residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after_1(output)

        # if residual should be applied after layer norm, apply it here
        if not self.residual_before_ln:
            output = output + residual_input

        return output, down, up

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Houlsby(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(
            self,
            input_size,
            down_sample=None,
            non_linearity="relu",
            init_bert_weights=True,
            add_layer_norm_before=True,
            add_layer_norm_after=True,
            residual_before_ln=True, # could try with
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln

        self.embed_dim = input_size

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list_1 = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list_1.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.embed_dim // 2

        # Linear down projection of the input
        seq_list_1.append(nn.Linear(self.embed_dim, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list_1.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down_1 = nn.Sequential(*seq_list_1)

        # Up projection to input size
        self.adapter_up_1 = nn.Linear(self.down_sample, self.embed_dim)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        self.adapter_norm_after_1 = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if init_bert_weights:
            self.adapter_down_1.apply(self.init_bert_weights)
            self.adapter_up_1.apply(self.init_bert_weights)

        self.feed_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.feed_layer_2 = nn.Linear(self.embed_dim, self.embed_dim)

        seq_list_2=[]
        seq_list_2.append(nn.Linear(self.embed_dim, self.down_sample))
        seq_list_2.append(Activation_Function_Class(non_linearity.lower()))
        self.adapter_down_2 = nn.Sequential(*seq_list_2)

        self.adapter_up_2 = nn.Linear(self.down_sample, self.embed_dim)

        self.adapter_norm_after_2 = nn.LayerNorm(self.input_size)

    def forward(self, x, residual_input=1.0):  # , residual_input=None):
        # print("x.shape:", x.shape)
        # print("self.adapter_down:", self.adapter_down)
        #
        # print("self.adapter_norm_before:", self.adapter_norm_before)
        #
        # print("self.adapter_norm_before(x):", self.adapter_norm_before(x))
        down = self.adapter_down_1(x)

        up = self.adapter_up_1(down)

        output = up

        # apply residual connection before layer norm if configured in this way
        if self.residual_before_ln:
            output = output + residual_input

        output = self.adapter_norm_after_1(output)

        output = self.feed_layer_1(output)
        output = self.feed_layer_2(output)

        down = self.adapter_down_2(output)

        up = self.adapter_up_2(down)

        output = up

        output = self.adapter_norm_after_2(output)
        # if residual should be applied after layer norm, apply it here
        if not self.residual_before_ln:
            output = output + residual_input

        return output, down, up

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
