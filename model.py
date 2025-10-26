import torch
import torch.nn as nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, cache_max_seq_len=2048)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # qkv projection and reshape
        qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_length, 3 * embed_dim)
        qkv = rearrange(qkv, 'b s (qkv h d) -> qkv b h s d', qkv=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, num_heads, seq_length, head_dim)

        # positional encoding with rotary embeddings
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        
        # Reshape back - use reshape instead of view when possible
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        attn_output = self.attention(x)
        x = x + attn_output
        x = self.layernorm1(x)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.layernorm2(x)

        return x


class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout=0.1):
        super(Model, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embed(x)
        x = self.embed_dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = torch.matmul(x, self.embed.weight.T) # Weight tying
        return logits