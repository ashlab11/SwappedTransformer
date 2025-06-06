import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings 

class DefaultEncoderLayerRoPE(nn.Module):
    def __init__(self, embed_dim, ff_dim, n_heads, activation_function=nn.ReLU, 
                 dropout=0.1, layer_norm_eps=1e-5):
        super(DefaultEncoderLayerRoPE, self).__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.activation_function = activation_function
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        
        # Rotary Positional Embeddings
        self.RoPE = RotaryPositionalEmbeddings(
            dim=self.head_dim, 
            base=10000, 
            max_seq_len=256
        )
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.pre_attn_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.pre_ffn_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            self.activation_function(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ff_dim, self.embed_dim),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        x: [batch_size, seq_len, embed_dim]
        attention_mask: [batch_size, seq_len] which tokens to attend to
        """
        B, L, D = input_ids.size()
        x = self.pre_attn_norm(input_ids)
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2) 
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.RoPE(q)
        k = self.RoPE(k)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, embed_dim]
        
        x = self.out_proj(attn_output)
        x = input_ids + x
        x = self.pre_ffn_norm(x)
        x = self.ffn(x)
        x = input_ids + x
        return x
    
    
#New architecture with swapped attention/FF layers 
class SwappedTransformerLayerRoPE(nn.Module):
    def __init__(self, embed_dim, ff_dim, n_heads, activation_function=nn.ReLU, 
                 dropout=0.1, layer_norm_eps=1e-5):
        super(DefaultEncoderLayerRoPE, self).__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.activation_function = activation_function
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        
        # Rotary Positional Embeddings
        self.RoPE = RotaryPositionalEmbeddings(
            dim=self.embed_dim, 
            base=10000, 
            max_seq_len=512
        )
        
        self.q_proj = nn.Linear(self.ff_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.ff_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.ff_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.pre_ffn_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.pre_attn_norm = nn.LayerNorm(self.ff_dim, eps=self.layer_norm_eps)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            self.activation_function(),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        x: [batch_size, seq_len, embed_dim]
        attention_mask: [batch_size, seq_len] which tokens to attend to
        """
        B, L, D = input_ids.size()
        x = self.pre_ffn_norm(input_ids)  # [B, L, embed_dim]
        x = self.ffn(x) # [B, L, ff_dim]
        
        x = self.pre_attn_norm(input_ids) #Unsure if this is necessary? I'll leave it for now.
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2) 
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.RoPE(q)
        k = self.RoPE(k)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, embed_dim]
        
        x = self.out_proj(attn_output)
        x = input_ids + x
        return x