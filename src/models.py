import torch
import torch.nn as nn
from layers import DefaultTransformerLayerRoPE, SwappedTransformerLayerRoPE
from attention import Attention

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, n_layers, n_heads, activation_function=nn.ReLU, swapped = False,
               dropout=0.1, layer_norm_eps=1e-5):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        if swapped:
            self.layers = nn.ModuleList([
                SwappedTransformerLayerRoPE(
                    embed_dim=embed_dim,
                    ff_dim=ff_dim,
                    n_heads=n_heads,
                    activation_function=activation_function,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps
                ) for _ in range(self.n_layers)  
            ])
        else:
            self.layers = nn.ModuleList([
                DefaultTransformerLayerRoPE(
                    embed_dim=embed_dim,
                    ff_dim=ff_dim,
                    n_heads=n_heads,
                    activation_function=activation_function,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps
                ) for _ in range(n_layers)
            ])
    
    def forward(self,
                input_ids, 
                attention_mask=None,
                **kwargs):
        
        x = self.embed(input_ids)  # [B, L, embed_dim]
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, **kwargs)
            
        return x
        
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, encoder_dim, n_layers, n_heads,
                 activation_function=nn.ReLU, dropout=0.1, layer_norm_eps=1e-5):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.self_attention_layers = nn.ModuleList(
            [DefaultTransformerLayerRoPE(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                n_heads=n_heads,
                activation_function=activation_function,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(n_layers)])
    
        self.cross_attention_layers = nn.ModuleList(
            [DefaultTransformerLayerRoPE(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                n_heads=n_heads,
                activation_function=activation_function,
                encoder_dim=encoder_dim,  # Use encoder dimension for cross-attention
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(n_layers)])
        
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids, encoder_inputs, attention_mask=None):
        x = self.embed(input_ids)
        #Alternating between self-attention and cross-attention
        for i in range(len(self.self_attention_layers)):
            x = self.self_attention_layers[i](x, attention_mask=attention_mask)
            x = self.cross_attention_layers[i](x, encoder_inputs=encoder_inputs, attention_mask=attention_mask)
        x = self.output_layer(x)  # Final linear layer to map back to vocabulary size
        return x
        
        

class AutoEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim,
                 ff_dim, n_layers_enc, n_layers_dec, n_heads, 
                 n_slots, activation_function=nn.ReLU,
                 dropout=0.1, layer_norm_eps=1e-5):
        super(AutoEncoder, self).__init__()
        
        """
        First part is easy, with encoder and then cross-attention. 
        Then, we want to reconstruct the input from the encoder output.
        Decoder will have same structure as encoder, but with cross-attention and self-attention
        and a final linear layer to map back to the vocabulary size.
        """
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            n_layers=n_layers_enc,
            n_heads=n_heads,
            activation_function=activation_function,
            swapped=False,  # Set to True if you want to use the swapped architecture
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )
        
        #We swap the encoder and decoder dimensions -- initial encoder works in vocab dim, 
        #But we want the "between space" to be in the embedding dim
        self.slots = nn.Parameter(torch.randn(1, n_slots, encoder_dim)) # Slots for cross-attention
        self.cross_attn = Attention(
            embed_dim=encoder_dim, 
            encoder_dim=embed_dim, 
            n_heads=n_slots,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )
        
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            encoder_dim=encoder_dim,  # Use encoder dimension for cross-attention
            n_layers=n_layers_dec,
            n_heads=n_heads,
            activation_function=activation_function,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )
        
    def forward(self, input_ids, labels = None, attention_mask=None):
        # Encoder part
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        slots = self.slots.expand(input_ids.size(0), -1, -1)
        # Cross-attention part
        cross_attn_outputs = self.cross_attn(slots, encoder_inputs=encoder_outputs, attention_mask=attention_mask)
        
        # Decoder part
        logits = self.decoder(input_ids, encoder_inputs=cross_attn_outputs, attention_mask=attention_mask)
        
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
                
        
        
        