import torch
from torch import nn
import torch.nn.functional as F
import math


class FSparseMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_attention_heads = 6
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.register_buffer("attn_mask", None)

    def update_mask(self, n_timesteps):
        self.attn_mask = self.get_attn_mask(n_timesteps, "both", local_attn_ctx=3).float()

    def forward(self, x, output_attentions=False, is_training=False):
        # print("x shape", x.shape)
        # print("hidden size", self.hidden_size)
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)

        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,
                                                                                                                      2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,
                                                                                                                      2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V

        n_timesteps = key.size()[2]

        if self.attn_mask is None or self.attn_mask.shape[-1] != n_timesteps:  # Update if needed
            self.update_mask(x.shape[1])

        # self.register_buffer("cached_mask", self.get_attn_mask(n_timesteps, "both", local_attn_ctx=3).float())
        # mask = self.get_attn_mask(n_timesteps, "both", local_attn_ctx=3).float()
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores * self.attn_mask + -1e9 * (1 - self.attn_mask)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
            .contiguous() \
            .view(batch_size, sequence_length, self.all_head_size)
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
            breakpoint()
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