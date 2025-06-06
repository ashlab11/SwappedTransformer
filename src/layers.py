import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings 
from attention import SelfAttention, CrossAttention, SlotAttention

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, ff_dim, n_heads, activation_function=nn.ReLU,
                 dropout=0.1, layer_norm_eps=1e-5):
        """
        A single Transformer layer for the encoder.
        Args:
            embed_dim (int): Dimension of the input embeddings.
            ff_dim (int): Dimension of the feed-forward network.
            n_heads (int): Number of attention heads.
            activation_function (callable): Activation function to use in the feed-forward network.
            dropout (float): Dropout rate for attention and feed-forward layers.
            layer_norm_eps (float): Epsilon value for layer normalization.
        """
        super(EncoderLayer, self).__init__()
        self.activation_function = activation_function
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )
        self.ff_dim = ff_dim
        self.activation_function = activation_function
        
        
        self.pre_attn_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.pre_ffn_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            self.activation_function(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ff_dim, self.embed_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, x, attention_mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        attention_mask: [batch_size, seq_len] which tokens to attend to
        """
        res1 = x
        x = self.pre_attn_norm(x)
        x = self.attention(x, attention_mask=attention_mask)
        x = res1 + x
        res2 = x
        x = self.pre_ffn_norm(x)
        x = self.ffn(x)
        x = res2 + x
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, ff_dim, n_heads, encoder_dim, activation_function=nn.ReLU,dropout=0.1, layer_norm_eps=1e-5):
        """
        A single Transformer layer for the decoder.
        Args:
            embed_dim (int): Dimension of the input embeddings.
            ff_dim (int): Dimension of the feed-forward network.
            n_heads (int): Number of attention heads.
            activation_function (callable): Activation function to use in the feed-forward network.
            encoder_dim (int): Dimension of the encoder inputs for cross-attention.
            dropout (float): Dropout rate for attention and feed-forward layers.
            layer_norm_eps (float): Epsilon value for layer normalization.
        """
        super(DecoderLayer, self).__init__()
        self.activation_function = activation_function
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.encoder_dim = encoder_dim
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        
        self.self_attention = SelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )
        
        self.cross_attention = CrossAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            encoder_dim=self.encoder_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, ff_dim),
            self.activation_function(),
            nn.Dropout(self.dropout),
            nn.Linear(ff_dim, self.embed_dim),
            nn.Dropout(self.dropout)
        )
        
        self.pre_self_attn_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.pre_cross_attn_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.pre_ffn_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)

    def forward(self, x, encoder_inputs, attention_mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        encoder_inputs: [batch_size, seq_len_enc, encoder_dim]
        """
        res1 = x
        x = self.pre_self_attn_norm(x)
        x = self.self_attention(x, attention_mask=attention_mask)
        x = res1 + x
        res2 = x
        x = self.pre_cross_attn_norm(x)
        x = self.cross_attention(x, encoder_inputs=encoder_inputs, attention_mask=attention_mask)
        x = res2 + x
        res3 = x
        x = self.pre_ffn_norm(x)
        x = self.ffn(x)
        x = res3 + x
        return x