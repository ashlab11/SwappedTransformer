import torch
import torch.nn as nn
from layers import DefaultEncoderLayerRoPE, SwappedTransformerLayerRoPE

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, n_layers, n_heads, activation_function=nn.ReLU, swapped = False,
               dropout=0.1, layer_norm_eps=1e-5):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.activation_function = activation_function
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.swapped = swapped
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lm = nn.Linear(embed_dim, vocab_size, bias=False)
        
        
        if self.swapped:
            self.layers = nn.ModuleList([
                SwappedTransformerLayerRoPE(
                    embed_dim=self.embed_dim,
                    ff_dim=self.ff_dim,
                    n_heads=self.n_heads,
                    activation_function=self.activation_function,
                    dropout=self.dropout,
                    layer_norm_eps=self.layer_norm_eps
                ) for _ in range(self.n_layers)  
            ])
        else:
            self.layers = nn.ModuleList([
                DefaultEncoderLayerRoPE(
                    embed_dim=self.embed_dim,
                    ff_dim=self.ff_dim,
                    n_heads=self.n_heads,
                    activation_function=self.activation_function,
                    dropout=self.dropout,
                    layer_norm_eps=self.layer_norm_eps
                ) for _ in range(self.n_layers)
            ])
    
    def forward(self,
                input_ids, 
                attention_mask=None,
                labels = None,
                **kwargs):
        
        x = self.embed(input_ids)  # [B, L, embed_dim]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask, **kwargs)
        
        logits = self.lm(x) # [B, L, vocab_size]
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100) #Ignore padding tokens
            
        return {
                "loss": loss,
                "logits": logits
            }