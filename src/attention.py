import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings 

class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1, encoder_dim = None, layer_norm_eps=1e-5):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
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
        Bq, Lq, Dq = x.size()
        assert Dq == self.embed_dim, f"Input dimension {Dq} does not match embed_dim {self.embed_dim}"

        q = self.q_proj(x).view(Bq, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.RoPE(q)
        
        if encoder_inputs is not None:
            Be, Le, De = encoder_inputs.size()
            assert Be == Bq, f"Batch size of encoder inputs {Be} does not match query inputs {Bq}"
            # if encoder inputs are provided, use cross-attention
            v = self.v_proj(encoder_inputs).view(Be, Le, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(encoder_inputs).view(Be, Le, self.n_heads, self.head_dim).transpose(1, 2)
            causal_mask = torch.ones(Le, Le, device=x.device).view(1, 1, Le, Le).to(bool)  # No causal mask for encoder inputs
        
        else:
            # if no encoder inputs are provided use self-attention with a causal mask
            v = self.v_proj(x).view(Bq, Lq, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(Bq, Lq, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.RoPE(k) # Apply RoPE to keys, only if self-attention
            causal_mask = torch.tril(torch.ones(Lq, Lq, device=x.device)).view(1, 1, Lq, Lq).to(bool)
        
        #Dealing with causal/attention masks
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(bool)
            attention_mask = attention_mask.expand(-1, -1, Lq, -1)  # Expand to match the shape of the attention mask
            mask = attention_mask & causal_mask
        else:
            mask = causal_mask
        
        print(q.shape, k.shape, v.shape, mask.shape)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
        
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(Bq, Lq, self.embed_dim)  # [B, L, embed_dim]
        attn_output = self.out_proj(attn_output)  # [B, n_heads, L, head_dim]

        return attn_output