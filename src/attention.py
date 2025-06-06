import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings 

class SelfAttention(nn.Module):
    """
    Self-Attention module with Rotary Positional Embeddings (RoPE).
    Args:
        embed_dim (int): Dimension of the input embeddings.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate for attention weights.
        layer_norm_eps (float): Epsilon value for layer normalization.
    """
    def __init__(self, embed_dim, n_heads, dropout=0.1, layer_norm_eps=1e-5):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.embed_dim % self.n_heads == 0, "embed_dim must be divisible by n_heads"
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        
        # Rotary Positional Embeddings
        self.RoPE = RotaryPositionalEmbeddings(
            dim=self.head_dim, 
            base=10000, 
            max_seq_len=256
        )
        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, x, attention_mask=None):
        B, L, D = x.size()
        assert D == self.embed_dim, f"Input dimension {D} does not match embed_dim {self.embed_dim}"

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)


        q = self.RoPE(q)
        k = self.RoPE(k) # Apply RoPE to keys, only if self-attention
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L).to(bool)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).to(bool) & mask
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)  # [B, L, embed_dim]
        attn_output = self.out_proj(attn_output)  # [B, n_heads, L, head_dim]
        return attn_output

class CrossAttention(nn.Module):
    """
    Cross-Attention module with Rotary Positional Embeddings (RoPE).
    Args:
        embed_dim (int): Dimension of the input embeddings.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate for attention weights.
        encoder_dim (int): Dimension of the encoder inputs.
        layer_norm_eps (float): Epsilon value for layer normalization.
    """
    def __init__(self, embed_dim, encoder_dim, n_heads, dropout=0.1, layer_norm_eps=1e-5):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.encoder_dim = encoder_dim
        assert self.embed_dim % self.n_heads == 0, "embed_dim must be divisible by n_heads"
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        
        # Rotary Positional Embeddings
        self.RoPE = RotaryPositionalEmbeddings(
            dim=self.head_dim, 
            base=10000, 
            max_seq_len=256
        )
        
        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.encoder_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.encoder_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, x, encoder_inputs, attention_mask=None):
        B, L, D = x.size()
        Be, Le, De = encoder_inputs.size()
        assert Be == B, f"Batch size of encoder inputs {Be} does not match query inputs {B}"
        assert D == self.embed_dim, f"Input dimension {D} does not match embed_dim {self.embed_dim}"
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_inputs).view(Be, Le, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_inputs).view(Be, Le, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.RoPE(q) # Only apply RoPE to queries
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(bool)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)  # [B, L, embed_dim]
        attn_output = self.out_proj(attn_output)  # [B, n_heads, L, head_dim]
        return attn_output

class SlotAttention(nn.Module):
    def __init__(self, incoming_dim, slot_dim, n_heads, n_slots, dropout=0.1, layer_norm_eps=1e-5):
        super(SlotAttention, self).__init__()
        self.incoming_dim = incoming_dim
        self.n_heads = n_heads
        self.slot_dim = slot_dim
        self.head_dim = slot_dim // n_heads
        self.n_slots = n_slots
        assert self.slot_dim % self.n_heads == 0, "slot dimension must be divisible by n_heads"
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        
        # Rotary Positional Embeddings
        self.RoPE = RotaryPositionalEmbeddings(
            dim=self.head_dim, 
            base=10000, 
            max_seq_len=256
        )
        
        # Query slots that will attend to the incoming sequence
        self.q = nn.Parameter(torch.randn(n_slots, self.slot_dim))
        self.k_proj = nn.Linear(self.incoming_dim, self.slot_dim)
        self.v_proj = nn.Linear(self.incoming_dim, self.slot_dim)
        self.out_proj = nn.Linear(self.slot_dim, self.slot_dim)
        
    def forward(self, x, attention_mask = None):
        B, L, D = x.size()
        assert D == self.incoming_dim, f"Input dimension {D} does not match incoming_dim {self.incoming_dim}"
        
        q = self.q.view(1, self.n_slots, self.n_heads, self.head_dim).transpose(1, 2)  # [1, n_heads, n_slots, head_dim]
        q = q.expand(B, -1, -1, -1)  # [B, n_heads, n_slots, head_dim]
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(bool)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, self.n_slots, self.slot_dim)  # [B, n_slots, slot_dim]
        attn_output = self.out_proj(attn_output)  # [B, n_heads, L, head_dim]
        return attn_output
