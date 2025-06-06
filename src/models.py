import torch
import torch.nn as nn
from layers import EncoderLayer, DecoderLayer
from attention import SlotAttention
class AutoEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, slot_dim,
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
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.ModuleList([EncoderLayer(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            n_heads=n_heads,
            activation_function=activation_function,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        ) for _ in range(n_layers_enc)])
        
        #We swap the encoder and decoder dimensions -- initial encoder works in vocab dim, 
        #But we want the "between space" to be in the embedding dim
        self.slot_attn = SlotAttention(
            incoming_dim = embed_dim,
            slot_dim = slot_dim,
            n_heads=n_heads,
            n_slots=n_slots,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )
        
        self.decoder = nn.ModuleList([DecoderLayer(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            n_heads=n_heads,
            encoder_dim=slot_dim,  # Cross-attention will use the encoder output
            activation_function=activation_function,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        ) for _ in range(n_layers_dec)])
        
    def forward(self, input_ids, labels = None, attention_mask=None):
        # Encoder part
        embedding = self.embed(input_ids)
        x = embedding
        for layer in self.encoder:
            x = layer(x, attention_mask=attention_mask)
        # Encoder outputs, which we will use for slot attention
        
        slots = self.slot_attn(x)
        
        # Decoder part
        x = embedding  # Start with the same input embeddings
        for layer in self.decoder:
            # We pass the slots as encoder inputs to the decoder
            x = layer(x, encoder_inputs=slots, attention_mask=attention_mask)
        
        # Final linear layer to map back to vocabulary size
        logits = nn.Linear(self.embed.embedding_dim, self.embed.num_embeddings)(x)
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
                
        
        
        