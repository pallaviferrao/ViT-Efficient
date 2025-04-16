from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import math

from Layers.Attention.vit_finetune import ViTSelfAttention


class ViTSelfAttentionRoute(nn.Module):
    LOAD_BALANCING_LOSSES = []
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.shared_num = int(self.num_attention_heads / 4)
        self.total_routed_attention_heads = int((self.num_attention_heads * 3) / 4)
        self.routed_head = int((self.num_attention_heads * 2) / 4)
        self.num_attention_heads = self.shared_num + self.total_routed_attention_heads

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        self.n_router = nn.Linear(config.hidden_size, self.total_routed_attention_heads)

        self.routing_head = nn.Linear(config.hidden_size, self.total_routed_attention_heads)
        self.sharing_head = nn.Linear(config.hidden_size, self.shared_num)
        self.wg_0 = torch.nn.Linear(config.hidden_size, 2, bias=False)

        # self.shared_num = int(self.num_attention_heads / 4)
        # self.total_routed_attention_heads = int((self.num_attention_heads * 3) / 4)
        # self.routed_head = int((self.num_attention_heads * 2) / 4)
        # self.num_attention_heads = self.shared_num + self.total_routed_attention_heads

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None,dropout_p=0.0,
                                     is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)



        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        B, N, C = hidden_states.shape

        _x = hidden_states.reshape(B * N, C)
        logits = self.routing_head(_x)
        gates = F.softmax(logits, dim=-1)
        # choose_attn_heads = F.sigmoid(self.n_router(_x))
        num_tokens, num_experts = gates.shape
        _, indices = torch.topk(gates, k=self.routed_head, dim=1)
        mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)
        me = gates.mean(dim=0)
        ce = mask.float().mean(dim=0)
        l_aux = torch.mean(me * ce) * num_experts * num_experts

        # me = gates.mean(dim=0)
        # ce = choose_attn_heads.float().mean(dim=0)
        # l_aux = torch.mean(me * ce) * num_experts * num_experts

        ViTSelfAttentionRoute.LOAD_BALANCING_LOSSES.append(l_aux)

        routed_head_gates = gates * mask
        denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        routed_head_gates /= denom_s
        routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head

        shared_head_weight = self.sharing_head(_x)
        shared_head_gates = F.softmax(shared_head_weight, dim=-1).reshape(B, N, -1) * self.shared_num
        weight_0 = self.wg_0(_x)

        weight_0 = F.softmax(weight_0, dim=-1).reshape(B, N, 2) * 2
        shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 0], shared_head_gates)
        routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 1], routed_head_gates)
        # print("shared shape", shared_head_gates[0][0])
        # masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2).repeat_interleave(self.attention_head_size, dim=2)
        masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = torch.einsum("bne,bned->bned", masked_gates, context_layer)
        # context_layer = context_layer.view(B, N, -1)


        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None