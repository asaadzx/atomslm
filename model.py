import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, max_seq_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # CREATE THE CAUSAL MASK
        # This creates a lower-triangular grid of 1s, and 0s in the upper right (the future)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("bias", mask)

    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # APPLY THE CAUSAL MASK HERE
        # Anywhere our mask buffer is 0, we replace the score with -inf (negative infinity).
        # When softmax runs, e^(-inf) becomes exactly 0, meaning 0% attention to the future!
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        # In SwiGLU, the internal hidden dimension is typically expanded to 8/3 of d_model
        hidden_dim = int(2 * (4 * d_model) / 3)
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU activation formula: Silu(w1(x)) * w3(x) -> passed into w2
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, max_seq_len=512):
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len)
        
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        # Attention layer + Residual connection
        x = x + self.attention(self.attention_norm(x))
        # Feed-Forward layer + Residual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Calculate variance along the last dimension (d_model)
        variance = x.pow(2).mean(-1, keepdim=True)
        # Multiply by the scale factor
        return x * torch.rsqrt(variance + self.eps) * self.weight

class AtomLanguageModel(nn.Module):
    def __init__(self, vocab_size=256000, d_model=256, n_heads=8, n_layers=4, max_seq_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Create a stack of multiple transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len) for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        # The LM Head projects our vector back out to the size of our vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, idx):
        x = self.token_embedding(idx)
        
        # Pass the data through every stacked layer block
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        logits = self.lm_head(x) # Output shape: (Batch, Sequence Length, Vocab Size)
        return logits

# --- UPDATE OUR TEST CODE BLOCK ---
if __name__ == "__main__":
    print("Initializing complete AtomLanguageModel...")
    model = AtomLanguageModel(n_layers=4) # A 4-layer stacked mini transformer!
    
    fake_input_data = torch.randint(0, 1000, (4, 512))
    print(f"Input batch shape: {fake_input_data.shape}")
    
    logits = model(fake_input_data)
    print(f"Final output Logits shape: {logits.shape}")