'''
Original Copyright Ibrahim Taha Aksu, 2023.
All Rights Reserved.
'''

import copy
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from transformers.models.t5.modeling_t5 import T5Block

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

# Adapter block implementation taken from Adapter-Hub
# https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/modeling.py
class Adapter(nn.Module):
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

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # ensure that the down sample size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        # Linear down projection of the input
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)

        # HACK My modification to adapter architecture 
        # Adding a dropout layer.
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, residual_input):  # , residual_input=None):
        down = self.adapter_down(x)

        up = self.adapter_up(down)

        output = up

        # apply residual connection before layer norm if configured in this way
        if self.residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # HACK my modification to adapter architecture, dropout layer.
        # output = self.dropout(output)

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

class Prefix_Prompter(pl.LightningModule):
    def __init__(self, args, model,down_proj,num_layers,dropout,device="cuda"):
        super().__init__()
        self.args = args
        self.num_heads = model.config.num_heads
        self.dropout = dropout
        self.cross_attention = MultiheadAttention(embed_dim = model.config.d_model,
                                                dropout = self.dropout,
                                                num_heads = self.num_heads)

        self.q = nn.Linear(model.config.d_model, model.config.d_model, bias=False)
        self.k = nn.Linear(model.config.d_model, model.config.d_model, bias=False)
        self.v = nn.Linear(model.config.d_model, model.config.d_model, bias=False)
        self.o = nn.Linear(model.config.d_model, model.config.d_model, bias=False)

        self.keys = torch.nn.ModuleList()
        self.values = torch.nn.ModuleList()
        

        for i in range(num_layers):
            self.keys.append(Adapter(
                                    input_size=model.config.d_model,
                                    down_sample=down_proj)
                            )
            self.values.append(Adapter(
                        input_size=model.config.d_model,
                        down_sample=down_proj)
                )         

    def forward(self,global_prompt,prompt_embed,prompt_attn,batch_size):
        '''
        Generate key and value prefixes [B x num_heads x dim_per_head] (num_headsxdim_per_head==model_dim) from prompt_embed [B x seq_len x model_dim]
        '''
        # e = sself.adapter(prompt_embed,prompt_embed)[0]
        # e = prompt_embed
        prefix_keys = []
        prefix_values = []
        number_of_tokens = global_prompt.shape[0]
        prompt_embed = torch.transpose(prompt_embed, 0, 1)
        # Cross attend global prompt with the slot description
        queries = self.q(global_prompt)  # BUG Has 2 dimensions, needs batch d 
        values = self.v(prompt_embed)  # BUG Has 2 dimensions, needs batch d 
        keys = self.k(prompt_embed)
        attention_out= self.cross_attention(query = queries,
                                    key = keys,
                                    value = values,
                                    key_padding_mask = ~prompt_attn.to(bool)  # For padding get the inverse of the mask. Trues are ignored.
                            )
        
        attended_prompt = self.o(torch.transpose(attention_out[0],0,1))


        # Generate key prefixes
        for i in range(len(self.keys)):
            prefix_key_init = self.keys[i](attended_prompt,attended_prompt)[0]
            prefix_key = prefix_key_init.reshape((batch_size,number_of_tokens,self.num_heads,-1))
            prefix_keys.append(torch.unsqueeze(prefix_key,0))
        
        # Generate value prefixes
        for i in range(len(self.values)):
            prefix_value_init = self.values[i](attended_prompt,attended_prompt)[0]
            prefix_value = prefix_value_init.reshape((batch_size,number_of_tokens,self.num_heads,-1))
            prefix_values.append(torch.unsqueeze(prefix_value,0))

        #Concat keys and value into a single matrix
        prefix_key = torch.cat(prefix_keys)
        prefix_value = torch.cat(prefix_values)

        return prefix_key,prefix_value

        



class PromptLayer_Adapter(pl.LightningModule):
    def __init__(self, args, model,down_proj):
        super().__init__()

        self.adapter = Adapter(
                input_size=model.config.d_model,
                down_sample=down_proj
            )


    def forward(self, inputs, attn_mask, history_size):
        # Split as dialogue history and prompt
        dh = inputs[:, :history_size+1, :]
        prompt = inputs[:, history_size+1:, :]

        prompter_out = self.adapter(prompt, prompt)

        output = torch.cat((dh, prompter_out[0]), 1)

        return output

class PromptLayer_T5Block(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.embed_dim = 512
        self.num_heads = 1

        # T5 block
        self.t5Block = T5Block(self.args["model_config"], 
                               has_relative_attention_bias=True)

        # Uncomment below to copy the first block's parameters rather then a cold start
        # self.t5Block.load_state_dict(model.encoder.block[0].state_dict())

    def forward(self, inputs, attn_mask, history_size):
        # Split as dialogue history and prompt
        dh = inputs[:, :history_size+1, :]
        prompt = inputs[:, history_size+1:, :]

        # Split as dialogue history and prompt attention
        prompt_attn = attn_mask[:, history_size+1:]
        prompt_attn = torch.reshape(prompt_attn, (prompt_attn.shape[0],1,1,-1))

        # Possible solution to multi gpu setting bug
        prompt_attn = prompt_attn.type_as(inputs)
        # Make it b,1,1,N for self attention

        output = self.t5Block(prompt, attention_mask=prompt_attn)
        x = torch.cat((dh, output[0]), 1)

        return x