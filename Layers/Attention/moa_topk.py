
import torch
from torch import nn
import torch.nn.functional as F
import math

class MixtureOfAttentionTopk(nn.Module):
    LOAD_BALANCING_LOSSES = []
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.shared_num = int(self.num_attention_heads/4)
        self.total_routed_attention_heads = int((self.num_attention_heads * 3) / 4)
        self.routed_head = int((self.num_attention_heads * 2) / 4)
        self.num_attention_heads = self.shared_num + self.total_routed_attention_heads

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        # print("All head size",self.all_head_size)
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.n_router = nn.Linear(self.hidden_size, self.total_routed_attention_heads)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.routing_head = nn.Linear(self.hidden_size, self.total_routed_attention_heads)
        self.sharing_head = nn.Linear(self.hidden_size, self.shared_num)
        self.wg_0 = torch.nn.Linear(self.hidden_size, 2, bias=False)


    def forward(self,x, output_attentions=False, is_training= False):
        B,N,C = x.shape
        _x = x.reshape(B * N, C)
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

        MixtureOfAttentionTopk.LOAD_BALANCING_LOSSES.append(l_aux)


        # if is_training:
        #     me = gates.mean(dim=0)
        #     ce = mask.float().mean(dim=0)
        #     l_aux = torch.mean(me * ce) * num_experts * num_experts
        #
        #     # me = gates.mean(dim=0)
        #     # ce = choose_attn_heads.float().mean(dim=0)
        #     # l_aux = torch.mean(me * ce) * num_experts * num_experts
        #
        #     MixtureOfAttention.LOAD_BALANCING_LOSSES.append(l_aux)
            # print("l_aux", l_aux)
            # print("Loss load balancing", l_aux)
        # routed_head_gates = gates * mask
        routed_head_gates = gates * mask
        denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        routed_head_gates /= denom_s
        routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head
        qkv = self.qkv_projection(x)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,
                                                                                                                      2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,
                                                                                                                      2)

        # print("query shape 2", query.shape)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        # attention_output = attention_output.transpose(1, 2) \
        #     .contiguous() \
        #     .view(batch_size, sequence_length, self.all_head_size)
        attention_output = attention_output.transpose(1, 2)

        shared_head_weight = self.sharing_head(_x)
        shared_head_gates = F.softmax(shared_head_weight, dim=-1).reshape(B, N, -1) * self.shared_num
        weight_0 = self.wg_0(_x)
        weight_0 = F.softmax(weight_0, dim=-1).reshape(B, N, 2) * 2
        shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 0], shared_head_gates)
        routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 1], routed_head_gates)
        # print("shared shape", shared_head_gates[0][0])
        # masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2).repeat_interleave(self.attention_head_size, dim=2)
        masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
        # print("shape masked",masked_gates.shape)
        # print("x masked", masked_gates.shape)
        # print("attention masked", attention_output.shape)
        attention_output = torch.einsum("bne,bned->bned", masked_gates, attention_output)
        attention_output = attention_output.view(B, N, -1)
        # print("shape after everthing", x.shape)
        # print("attention shape after everthing", attention_output.shape)

        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)