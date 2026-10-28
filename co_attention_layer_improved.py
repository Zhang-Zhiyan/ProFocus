import logging
import math

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GeLU(nn.Module):
    def forward(self, x):
        return gelu_new(x)


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )

    class LayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            super(LayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class SimpleTextToImageFusion(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout_prob=0.05):
        super(SimpleTextToImageFusion, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query1 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.image_norm = LayerNorm(hidden_size, eps=1e-12)
        self.text_norm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, image_tensor, text_tensor, image_attention_mask=None, text_attention_mask=None):
        batch_size, num_regions, _ = image_tensor.size()
        _, num_words, _ = text_tensor.size()

        normed_image = self.image_norm(image_tensor)
        normed_text = self.text_norm(text_tensor)
        mixed_query_layer = self.query1(normed_image)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        mixed_key_layer = self.key2(normed_text)
        mixed_value_layer = self.value2(normed_text)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if text_attention_mask is not None:
            attention_scores = attention_scores + text_attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out_proj(context_layer)

        return attention_output


class TextToImageConnectionLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout_prob=0.1, intermediate_size=3072):
        super(TextToImageConnectionLayer, self).__init__()
        self.norm1 = LayerNorm(hidden_size, eps=1e-12)
        self.norm2 = LayerNorm(hidden_size, eps=1e-12)

        self.fusion = SimpleTextToImageFusion(hidden_size, num_attention_heads, dropout_prob)

        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = GeLU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, image_tensor, text_tensor, text_attention_mask=None):
        attention_output = self.fusion(image_tensor, text_tensor, None, text_attention_mask)
        attention_output = self.dropout(attention_output)
        hidden = self.norm1(image_tensor + attention_output)

        ffn_output = self.dense1(hidden)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout(ffn_output)
        layer_output = self.norm2(hidden + ffn_output)

        return layer_output


class ThreeStageTextToImageFusion(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout_prob=0.1,
                 intermediate_size=3072, use_output_projection=True):
        super(ThreeStageTextToImageFusion, self).__init__()
        self.stage1 = TextToImageConnectionLayer(hidden_size, num_attention_heads, dropout_prob, intermediate_size)
        self.stage2 = TextToImageConnectionLayer(hidden_size, num_attention_heads, dropout_prob, intermediate_size)
        self.stage3 = TextToImageConnectionLayer(hidden_size, num_attention_heads, dropout_prob, intermediate_size)
        
        self.use_output_projection = use_output_projection
        if use_output_projection:
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                GeLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(intermediate_size, hidden_size),
                nn.Dropout(dropout_prob)
            )
        
        self.final_norm = LayerNorm(hidden_size, eps=1e-12)

    def forward(self, image_tensor, text_stage1, text_stage2, text_stage3,
                text_attention_mask1=None, text_attention_mask2=None, text_attention_mask3=None):
        enhanced_image1 = self.stage1(image_tensor, text_stage1, text_attention_mask1)

        enhanced_image2 = self.stage2(enhanced_image1, text_stage2, text_attention_mask2)

        enhanced_image3 = self.stage3(enhanced_image2, text_stage3, text_attention_mask3)
        
        final_enhanced = enhanced_image3
        
        if self.use_output_projection:
            final_output = final_enhanced + self.output_projection(final_enhanced)
        else:
            final_output = final_enhanced
        
        final_output = self.final_norm(final_output)
        
        return final_output