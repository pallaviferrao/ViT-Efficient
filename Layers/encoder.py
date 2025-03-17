from torch import nn

from Layers.block import Block


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def  __init__(self, config, attention_type = None):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config, attention_type = attention_type)
            self.blocks.append(block)

        # enc_list = [Block(config, attention_type = attention_type) for _ in
        #             range(config["num_hidden_layers"])]
        # self.blocks = nn.Sequential(*enc_list)


    def forward(self, x, output_attentions=False, is_training=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        # if output_attentions:
        #
        #     all_attentions.append(self.blocks(x)[1])
        # else:
        #
        #     all_attentions.append(self.blocks(x))
        #
        # return (x, all_attentions)
        for block in self.blocks:
            # x, attention_probs = block(x,output_attentions=output_attentions, is_training= is_training)
            x, attention_probs = block(x)

            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)