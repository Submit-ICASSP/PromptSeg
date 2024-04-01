
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Type, Union

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
try:
    from torch import inf
except:
    from torch._six import inf

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
    
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class Swish(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)



class GPT2Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, scale_attn_weights, 
                 max_position_embeddings, attention_dropout_rate, hidden_dropout_rate ):
        super().__init__()

        max_positions = max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = scale_attn_weights


        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.resid_dropout = nn.Dropout(hidden_dropout_rate)


    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if attention_mask is not None:
            # Apply the attention mask
            # attn_weights = attn_weights + attention_mask
            attn_weights.masked_fill_(attention_mask, -inf)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
      
        
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)



class GPT2MLP(nn.Module):
    def __init__(self, hidden_size = 768, intermediate_size = 3072, hidden_dropout_rate=0.1):
        super().__init__()
        self.c_fc = Conv1D(intermediate_size, hidden_size)
        self.c_proj = Conv1D(hidden_size, intermediate_size)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(hidden_dropout_rate)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states



class GPT2Block(nn.Module):
    def __init__(self, hidden_size = 768, ffn_hidden_size = None, 
                 num_attention_heads = 12, scale_attention_weights = True, 
                 max_position_embeddings=4096, attention_dropout_rate = 0.1,
                 hidden_dropout_rate = 0.1, norm_eps = 1e-5, layer_idx=None):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = GPT2Attention(hidden_size = hidden_size, 
                                  num_attention_heads = num_attention_heads, 
                                  scale_attn_weights = scale_attention_weights, 
                                  max_position_embeddings = max_position_embeddings, 
                                  attention_dropout_rate = attention_dropout_rate, 
                                  hidden_dropout_rate = hidden_dropout_rate)
        
        self.ln_2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = GPT2MLP(hidden_size = hidden_size, 
                           intermediate_size = ffn_hidden_size, 
                           hidden_dropout_rate = hidden_dropout_rate)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2Model(nn.Module):
    def __init__(self, in_chans=256, depth = 12, hidden_size = 768, ffn_hidden_size = None, 
                 mlp_ratio = 4., num_attention_heads = 12, scale_attention_weights = True, 
                 max_position_embeddings=4096, attention_mask=None, attention_dropout_rate = 0.1,
                 hidden_dropout_rate = 0.1, embedding_drop_rate = 0.1,  norm_eps = 1e-5):
        super().__init__()
        
        ffn_hidden_size = ffn_hidden_size if ffn_hidden_size is not None else int(mlp_ratio * hidden_size)

        
        self.wte = nn.Linear(in_chans, hidden_size)
        self.wpe = nn.Embedding(max_position_embeddings, hidden_size)

        self.drop = nn.Dropout(embedding_drop_rate)
        self.h = nn.ModuleList(
            [GPT2Block(hidden_size = hidden_size, ffn_hidden_size = ffn_hidden_size, 
            num_attention_heads = num_attention_heads, scale_attention_weights = scale_attention_weights, 
            max_position_embeddings=max_position_embeddings, attention_dropout_rate = attention_dropout_rate,
            hidden_dropout_rate = hidden_dropout_rate, norm_eps = norm_eps, layer_idx=i) for i in range(depth)])

        self.ln_f = nn.LayerNorm(hidden_size, eps=norm_eps)
        
        if attention_mask is not None:
            self.register_buffer("attention_mask", attention_mask)
        else:
            self.attention_mask = None

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor],
        past: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        change_dimension: Optional[bool] = True,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, ...]:  
        """
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
            last_hidden_state (:

        """
        if attention_mask is None:
            attention_mask = self.attention_mask
        
        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1] # b,s
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify inputs_embeds")

        if past is None:
            past_length = 0
            past = tuple([None] * len(self.h))
        else:
            past_length = past[0][0].size(-2)

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if change_dimension:
            inputs_embeds = self.wte(inputs_embeds)
        
        if head_mask is None:
            head_mask = [None] * len(self.h)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),) # b,s,h

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            if len(outputs)<2: # for (a, ) case, a is a tensor, can't assign second element to present
                hidden_states = outputs[0]
                present = ()
            else:
                hidden_states, present = outputs[:2]    
            presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

        return tuple(v for v in [
                hidden_states, presents, all_hidden_states, all_attentions] if v is not None)