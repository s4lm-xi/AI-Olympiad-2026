import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, attention_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_size)
        self.attn = nn.MultiheadAttention(embedding_size, attention_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, layers=4, heads=4, context_length=256, dropout=0.1):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_length, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_size, heads, dropout) for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_size)
        self.head = nn.Linear(embedding_size, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        b, t = idx.shape
        device = idx.device
        pos = torch.arange(t, device=device).unsqueeze(0)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(pos))

        causal_mask = torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def create_model(vocab_size, embedding_size=128, layers=4, heads=4, context_length=256, dropout=0.1):
    return MiniGPT(vocab_size, embedding_size, layers, heads, context_length, dropout)


def add_transformer_layer(model, attention_heads=4, dropout=0.1):
    embedding_size = model.token_embedding.embedding_dim
    new_block = TransformerBlock(embedding_size, attention_heads, dropout)
    model.blocks.append(new_block)
    return model


def save_model(model, path="my_llm.pt", heads=None):
    torch.save({
        "state_dict": model.state_dict(),
        "vocab_size": model.token_embedding.num_embeddings,
        "embedding_size": model.token_embedding.embedding_dim,
        "layers": len(model.blocks),
        "context_length": model.context_length,
        "heads": heads or model.blocks[0].attn.num_heads,
    }, path)


def load_model(path="my_llm.pt"):
    ckpt = torch.load(path, map_location="cpu")
    model = create_model(
        vocab_size=ckpt["vocab_size"],
        embedding_size=ckpt["embedding_size"],
        layers=ckpt["layers"],
        heads=ckpt["heads"],
        context_length=ckpt["context_length"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model
