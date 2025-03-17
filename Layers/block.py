from Layers.Attention.fast_multihead import FasterMultiHeadAttention
from Layers.Attention.sparse_attention import FSparseMultiHeadAttention
from Layers.Attention.moa_topk import MixtureOfAttention
from Layers.Attention.sparse_moa_share import MixtureOfAttentionSparseShare
from torch import nn

from Layers.mlp import MLP

class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config, attention_type):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if attention_type == 'moa':
            self.attention = MixtureOfAttention(config)
        if attention_type == 'multihead':
            self.attention = FasterMultiHeadAttention(config)
        if attention_type == 'sparse':
            self.attention = FSparseMultiHeadAttention(config)
        if attention_type == 'moa-sparse-share':
            self.attention = MixtureOfAttentionSparseShare(config)
        # if  self.use_faster_attention:
        #     self.attention = FasterMultiHeadAttention(config)
        #     # self.attention = MixtureOfAttention(config)
        # else:
        #     # self.attention = MultiHeadAttention(config)
        #     self.attention = FSparseMultiHeadAttention(config)
        #     # self.attention = SparseAttention(6, attn_mode="strided", local_attn_ctx=5, blocksize=36)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])


    def forward(self, x):
        # Self-attention
        x = self.layernorm_1(x)
        output_attentions = False
        attention_output, attention_probs = \
            self.attention(x, output_attentions=output_attentions, is_training=False)
        # output_attentions = False
        # B, L, E = x.shape  # batch size, sequence length, embedding size
        # print("batch, lenght, embedding", B,L,E)
        # q = torch.randn(B, L, E)
        # k = torch.randn(B, L, E)
        # v = torch.randn(B, L, E)

        # # Create Linear layers for Q, K, V
        # q_layer = nn.Linear(E, E)
        # k_layer = nn.Linear(E, E)
        # v_layer = nn.Linear(E, E)
        #
        # # Apply Xavier initialization to the weights
        # init.xavier_uniform_(q_layer.weight)
        # init.xavier_uniform_(k_layer.weight)
        # init.xavier_uniform_(v_layer.weight)
        #
        # # Generate the tensors using the linear layers
        # q = q_layer(torch.randn(B, L, E))
        # k = k_layer(torch.randn(B, L, E))
        # v = v_layer(torch.randn(B, L, E))
        # # Forward pass through the SparseAttention module
        # attention_output = self.attention(q, k, v)

        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        # if not output_attentions:
        #     return (x, None)
        # else:
        #     return (x, attention_probs)
        return (x, None)
        # return x