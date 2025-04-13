
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

class MixtureOfAttentionHeterogenousSparseShare(nn.Module):
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
        self.register_buffer("attn_mask_strided", None)
        self.register_buffer("attn_mask_both", None)

    def update_mask_strided(self, n_timesteps):
        self.attn_mask_strided = self.get_attn_mask(n_timesteps, "strided", local_attn_ctx=3).float()

    def update_mask_both(self, n_timesteps):
        self.attn_mask_both = self.get_attn_mask(n_timesteps, "both", local_attn_ctx=3).float()

    def calculate_sparsity(self, matrix):
        return np.count_nonzero(matrix == 0) / matrix.numel()

    def forward(self,x, output_attentions=False, is_training= False):
        torch.cuda.empty_cache()
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

        MixtureOfAttentionHeterogenousSparseShare.LOAD_BALANCING_LOSSES.append(l_aux)


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
        n_timesteps = key.size()[2]
        if self.attn_mask_strided is None or self.attn_mask_strided.shape[-1] != n_timesteps:  # Update if needed
            self.update_mask_strided(n_timesteps)


        if self.attn_mask_both is None or self.attn_mask_both.shape[-1] != n_timesteps:  # Update if needed
            self.update_mask_both(n_timesteps)

        # device = attention_probs.device  # Ensure everything is on the same device

        # self.attn_mask = self.attn_mask.to(device)  # Move mask to the same device
        # self.attn_mask = self.attn_mask.to(attention_probs.dtype)
        # mask2 = self.get_attn_mask(N, "both", local_attn_ctx=3).float()
        # mask2 = mask2.to(attention_probs.device)
        # attention_probs = torch.cat((attention_probs[:, :self.shared_num, :, :],
        #    attention_probs[:, self.shared_num:, :, :] * mask2), dim=1)

        attention_probs = torch.cat(
            (
                attention_probs[:, :self.shared_num, :, :],
                attention_probs[:, self.shared_num: self.shared_num+3, :, :].masked_fill(self.attn_mask_strided == 0,
                                                                                           0),
                attention_probs[:, self.shared_num + 3:, :, :].masked_fill(self.attn_mask_both == 0, 0)
            ), dim=1
        )

        print("sparsity percentage", self.calculate_sparsity(attention_probs))


        # self.attn_mask = self.attn_mask.to(attention_probs.dtype)
        # attention_probs[:, self.shared_num:, :, :] = attention_probs[:, self.shared_num:, :, :].masked_fill(self.attn_mask == 0, 0)
        # attention_probs = torch.cat(
        #     (attention_probs[:, :self.shared_num, :, :],
        #      attention_probs[:, self.shared_num:, :, :].masked_fill(self.attn_mask == 0, 0)), dim=1
        # )

        # attention_probs[:, self.shared_num:, :, :] = torch.where(
        #     self.attn_mask == 0, torch.tensor(0.0, device=attention_probs.device, dtype=attention_probs.dtype),
        #     attention_probs[:, self.shared_num:, :, :]
        # )

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
        # shared_head_gates = shared_head_gates * mask + -1e9 * (1 - mask)
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

    def get_attn_mask(self, n, attn_mode, local_attn_ctx=None):
        if attn_mode == 'all':
            b = torch.tril(torch.ones([n, n]))
        elif attn_mode == 'local':
            bandwidth = local_attn_ctx
            ctx = min(n - 1, bandwidth - 1)
            b = torch.logical_and(torch.tril(torch.ones([n, n]), ctx), torch.triu(torch.ones([n, n]), -ctx)).float()
        elif attn_mode == 'strided':
            stride = local_attn_ctx
            x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
            y = torch.transpose(x, 0, 1)
            z = torch.zeros([n, n], dtype=torch.int32)
            q = z + x
            k = z + y
            c1 = q >= k
            c2 = torch.eq(torch.fmod(q - k, stride), 0)
            c3 = torch.logical_and(c1, c2)
            b = c3.float()
        elif attn_mode == 'both':
            stride = local_attn_ctx
            x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
            y = torch.transpose(x, 0, 1)
            z = torch.zeros([n, n], dtype=torch.int32)
            q = z + x
            k = z + y
            c1 = q >= k
            c2 = torch.eq(torch.fmod(q - k, stride), 0)
            c3 = torch.logical_and(c1, c2)
            bandwidth = local_attn_ctx
            ctx = min(n - 1, bandwidth - 1)
            a = torch.logical_and(torch.tril(torch.ones([n, n]), ctx), torch.triu(torch.ones([n, n]), -ctx))
            b = torch.logical_or(a, c3).float()
        else:
            raise ValueError('Not yet implemented')
        b = torch.reshape(b, [1, 1, n, n])
        return b
    # LOAD_BALANCING_LOSSES = []
    # def __init__(self, config):
    #     super().__init__()
    #     self.hidden_size = config["hidden_size"]
    #     self.num_attention_heads = config["num_attention_heads"]
    #     # The attention head size is the hidden size divided by the number of attention heads
    #     self.attention_head_size = self.hidden_size // self.num_attention_heads
    #     self.shared_num = int(self.num_attention_heads/4)
    #     self.total_routed_attention_heads = int((self.num_attention_heads * 3) / 4)
    #     self.routed_head = int((self.num_attention_heads * 2) / 4)
    #     self.num_attention_heads = self.shared_num + self.total_routed_attention_heads
    #
    #     self.all_head_size = self.num_attention_heads * self.attention_head_size
    #     # Whether or not to use bias in the query, key, and value projection layers
    #     # print("All head size",self.all_head_size)
    #     self.qkv_bias = config["qkv_bias"]
    #     # Create a linear layer to project the query, key, and value
    #     self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
    #     self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
    #     # Create a linear layer to project the attention output back to the hidden size
    #     # In most cases, all_head_size and hidden_size are the same
    #     self.n_router = nn.Linear(self.hidden_size, self.total_routed_attention_heads)
    #     self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
    #     self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
    #
    #     self.routing_head = nn.Linear(self.hidden_size, self.total_routed_attention_heads)
    #     self.sharing_head = nn.Linear(self.hidden_size, self.shared_num)
    #     self.wg_0 = torch.nn.Linear(self.hidden_size, 2, bias=False)
    #     self.register_buffer("attn_mask_strided", None)
    #     self.register_buffer("attn_mask_both", None)
    #
    # def update_mask_strided(self, n_timesteps):
    #     self.attn_mask_strided = self.get_attn_mask(n_timesteps, "strided", local_attn_ctx=3).float()
    #
    # def update_mask_both(self, n_timesteps):
    #     self.attn_mask_both = self.get_attn_mask(n_timesteps, "both", local_attn_ctx=3).float()
    #
    # def forward(self,x, output_attentions=False, is_training= False):
    #     torch.cuda.empty_cache()
    #     B,N,C = x.shape
    #     _x = x.reshape(B * N, C)
    #     logits = self.routing_head(_x)
    #     gates = F.softmax(logits, dim=-1)
    #     # choose_attn_heads = F.sigmoid(self.n_router(_x))
    #     num_tokens, num_experts = gates.shape
    #     _, indices = torch.topk(gates, k=self.routed_head, dim=1)
    #     mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)
    #
    #     me = gates.mean(dim=0)
    #     ce = mask.float().mean(dim=0)
    #     l_aux = torch.mean(me * ce) * num_experts * num_experts
    #
    #     # me = gates.mean(dim=0)
    #     # ce = choose_attn_heads.float().mean(dim=0)
    #     # l_aux = torch.mean(me * ce) * num_experts * num_experts
    #
    #     MixtureOfAttentionHeterogenousSparseShare.LOAD_BALANCING_LOSSES.append(l_aux)
    #
    #
    #     # if is_training:
    #     #     me = gates.mean(dim=0)
    #     #     ce = mask.float().mean(dim=0)
    #     #     l_aux = torch.mean(me * ce) * num_experts * num_experts
    #     #
    #     #     # me = gates.mean(dim=0)
    #     #     # ce = choose_attn_heads.float().mean(dim=0)
    #     #     # l_aux = torch.mean(me * ce) * num_experts * num_experts
    #     #
    #     #     MixtureOfAttention.LOAD_BALANCING_LOSSES.append(l_aux)
    #         # print("l_aux", l_aux)
    #         # print("Loss load balancing", l_aux)
    #     # routed_head_gates = gates * mask
    #     routed_head_gates = gates * mask
    #     denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
    #     denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    #     routed_head_gates /= denom_s
    #     routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head
    #     qkv = self.qkv_projection(x)
    #     query, key, value = torch.chunk(qkv, 3, dim=-1)
    #     batch_size, sequence_length, _ = query.size()
    #     query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,
    #                                                                                                                   2)
    #     key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
    #     value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,
    #                                                                                                                   2)
    #
    #     # print("query shape 2", query.shape)
    #     # Calculate the attention scores
    #     # softmax(Q*K.T/sqrt(head_size))*V
    #     attention_scores = torch.matmul(query, key.transpose(-1, -2))
    #     attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    #     attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    #     attention_probs = self.attn_dropout(attention_probs)
    #     n_timesteps = key.size()[2]
    #     if self.attn_mask_strided is None or self.attn_mask_strided.shape[-1] != n_timesteps:  # Update if needed
    #         self.update_mask_strided(n_timesteps)
    #
    #
    #     # if self.attn_mask_both is None or self.attn_mask_both.shape[-1] != n_timesteps:  # Update if needed
    #     #     self.update_mask_both(n_timesteps)
    #     # device = attention_probs.device  # Ensure everything is on the same device
    #
    #     # self.attn_mask = self.attn_mask.to(device)  # Move mask to the same device
    #     # self.attn_mask = self.attn_mask.to(attention_probs.dtype)
    #     # mask2 = self.get_attn_mask(N, "both", local_attn_ctx=3).float()
    #     # mask2 = mask2.to(attention_probs.device)
    #     # attention_probs = torch.cat((attention_probs[:, :self.shared_num, :, :],
    #     #    attention_probs[:, self.shared_num:, :, :] * mask2), dim=1)
    #
    #     # Ensure masks are on the same device and dtype
    #     # self.attn_mask_strided = self.attn_mask_strided.to(attention_probs.device, dtype=attention_probs.dtype)
    #     # self.attn_mask_both = self.attn_mask_both.to(attention_probs.device, dtype=attention_probs.dtype)
    #
    #     attention_probs_masked = torch.cat(
    #         (
    #             attention_probs[:, :self.shared_num, :, :],
    #             attention_probs[:, self.shared_num: , :, :].masked_fill(self.attn_mask_strided == 0,
    #                                                                                        0)
    #             # attention_probs[:, self.shared_num + 3:, :, :].masked_fill(self.attn_mask_strided == 0, 0)
    #         ), dim=1
    #     )
    #
    #     # self.attn_mask = self.attn_mask.to(attention_probs.dtype)
    #     # attention_probs[:, self.shared_num:, :, :] = attention_probs[:, self.shared_num:, :, :].masked_fill(self.attn_mask == 0, 0)
    #     # attention_probs = torch.cat(
    #     #     (attention_probs[:, :self.shared_num, :, :],
    #     #      attention_probs[:, self.shared_num:, :, :].masked_fill(self.attn_mask == 0, 0)), dim=1
    #     # )
    #
    #     # attention_probs[:, self.shared_num:, :, :] = torch.where(
    #     #     self.attn_mask == 0, torch.tensor(0.0, device=attention_probs.device, dtype=attention_probs.dtype),
    #     #     attention_probs[:, self.shared_num:, :, :]
    #     # )
    #     attention_probs_masked = attention_probs_masked.detach()
    #     # Calculate the attention output
    #     attention_output = torch.matmul(attention_probs_masked, value)
    #     # Resize the attention output
    #     # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
    #     # To (batch_size, sequence_length, all_head_size)
    #     # attention_output = attention_output.transpose(1, 2) \
    #     #     .contiguous() \
    #     #     .view(batch_size, sequence_length, self.all_head_size)
    #     attention_output = attention_output.transpose(1, 2)
    #
    #     shared_head_weight = self.sharing_head(_x)
    #
    #
    #     shared_head_gates = F.softmax(shared_head_weight, dim=-1).reshape(B, N, -1) * self.shared_num
    #     weight_0 = self.wg_0(_x)
    #     weight_0 = F.softmax(weight_0, dim=-1).reshape(B, N, 2) * 2
    #     shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 0], shared_head_gates)
    #     routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 1], routed_head_gates)
    #     # print("shared shape", shared_head_gates[0][0])
    #     # masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2).repeat_interleave(self.attention_head_size, dim=2)
    #     # shared_head_gates = shared_head_gates * mask + -1e9 * (1 - mask)
    #     masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
    #     # print("shape masked",masked_gates.shape)
    #     # print("x masked", masked_gates.shape)
    #     # print("attention masked", attention_output.shape)
    #     attention_output = torch.einsum("bne,bned->bned", masked_gates, attention_output)
    #     attention_output = attention_output.view(B, N, -1)
    #     # print("shape after everthing", x.shape)
    #     # print("attention shape after everthing", attention_output.shape)
    #
    #     # Project the attention output back to the hidden size
    #     attention_output = self.output_projection(attention_output)
    #     attention_output = self.output_dropout(attention_output)
    #     # Return the attention output and the attention probabilities (optional)
    #
    #     if not output_attentions:
    #         return (attention_output, None)
    #     else:
    #         return (attention_output, attention_probs)
    #
    # def get_attn_mask(self, n, attn_mode, local_attn_ctx=None):
    #     if attn_mode == 'all':
    #         b = torch.tril(torch.ones([n, n]))
    #     elif attn_mode == 'local':
    #         bandwidth = local_attn_ctx
    #         ctx = min(n - 1, bandwidth - 1)
    #         b = torch.logical_and(torch.tril(torch.ones([n, n]), ctx), torch.triu(torch.ones([n, n]), -ctx)).float()
    #     elif attn_mode == 'strided':
    #         stride = local_attn_ctx
    #         x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
    #         y = torch.transpose(x, 0, 1)
    #         z = torch.zeros([n, n], dtype=torch.int32)
    #         q = z + x
    #         k = z + y
    #         c1 = q >= k
    #         c2 = torch.eq(torch.fmod(q - k, stride), 0)
    #         c3 = torch.logical_and(c1, c2)
    #         b = c3.float()
    #     elif attn_mode == 'both':
    #         stride = local_attn_ctx
    #         x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
    #         y = torch.transpose(x, 0, 1)
    #         z = torch.zeros([n, n], dtype=torch.int32)
    #         q = z + x
    #         k = z + y
    #         c1 = q >= k
    #         c2 = torch.eq(torch.fmod(q - k, stride), 0)
    #         c3 = torch.logical_and(c1, c2)
    #         bandwidth = local_attn_ctx
    #         ctx = min(n - 1, bandwidth - 1)
    #         a = torch.logical_and(torch.tril(torch.ones([n, n]), ctx), torch.triu(torch.ones([n, n]), -ctx))
    #         b = torch.logical_or(a, c3).float()
    #     else:
    #         raise ValueError('Not yet implemented')
    #     b = torch.reshape(b, [1, 1, n, n])
    #     return b