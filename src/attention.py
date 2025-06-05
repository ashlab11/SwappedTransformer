import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings 

class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1, encoder_dim = None, layer_norm_eps=1e-5):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim / n_heads
        assert self.embed_dim % self.n_heads == 0, "embed_dim must be divisible by n_heads"
        self.encoder_dim = encoder_dim if encoder_dim is not None else embed_dim
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        
        # Rotary Positional Embeddings
        self.RoPE = RotaryPositionalEmbeddings(
            dim=self.head_dim, 
            base=10000, 
            max_seq_len=256
        )
        
        #Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.encoder_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.encoder_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, x, attention_mask=None, encoder_inputs = None):
        B, L, D = x.size()

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.RoPE(q)
        
        if encoder_inputs is not None:
            # if encoder inputs are provided, use cross-attention
            v = self.v_proj(encoder_inputs).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(encoder_inputs).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            causal_mask = torch.ones(L, L, device=x.device).view(1, 1, L, L).to(bool)  # No causal mask for encoder inputs
        
        else:
            # if no encoder inputs are provided use self-attention with a causal mask
            v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.RoPE(k) # Apply RoPE to keys, only if self-attention
            causal_mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L).to(bool)
        
        #Dealing with causal/attention masks
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = attention_mask & causal_mask
        else:
            mask = causal_mask
                    
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, embed_dim]
        return attn_output