import torch
import torch.nn as nn
from layers import EncoderLayer, DecoderLayer
from attention import SlotAttention

class AutoEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, slot_dim,
                 ff_dim, n_layers_enc, n_layers_dec, n_heads,
                 n_slots, activation_function=nn.ReLU,
                 dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                n_heads=n_heads,
                activation_function=activation_function,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(n_layers_enc)
        ])

        self.slot_attn = SlotAttention(
            incoming_dim=embed_dim,
            slot_dim=slot_dim,
            n_heads=n_heads,
            n_slots=n_slots,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        self.decoder = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                n_heads=n_heads,
                encoder_dim=slot_dim,
                activation_function=activation_function,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(n_layers_dec)
        ])

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = self.embed(input_ids)
        for layer in self.encoder:
            x = layer(x, attention_mask=attention_mask)

        slots = self.slot_attn(x, attention_mask=attention_mask)

        x = self.embed(input_ids)
        for layer in self.decoder:
            x = layer(x, encoder_inputs=slots, attention_mask=attention_mask)

        logits = self.output_layer(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits}
