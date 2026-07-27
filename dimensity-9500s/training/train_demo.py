# train_demo.py — small-scale on-device training demo (CPU) for POCO X8 Pro Max.
# Trains a tiny char-level transformer on tiny-shakespeare with PyTorch.
import math, os, time, urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
DEV = "cpu"
torch.set_num_threads(8)

DATA = os.path.expanduser("~/ai-bench/tinyshakespeare.txt")
if not os.path.exists(DATA):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", DATA)
text = open(DATA).read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_vocab = len(chars)
train = data[: int(0.9 * len(data))]

BLOCK, BATCH, EMB, HEADS, LAYERS = 128, 32, 128, 4, 4

def get_batch():
    ix = torch.randint(len(train) - BLOCK - 1, (BATCH,))
    x = torch.stack([train[i : i + BLOCK] for i in ix])
    y = torch.stack([train[i + 1 : i + BLOCK + 1] for i in ix])
    return x, y

class CausalSelfAttention(nn.Module):
    # Manual attention (nanoGPT-style) — numerically stable; avoids the fused
    # nn.MultiheadAttention kernel that emits nan on this ARM torch build.
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMB, 3 * EMB)
        self.proj = nn.Linear(EMB, EMB)
        self.hd = EMB // HEADS
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(EMB, dim=2)
        q = q.view(B, T, HEADS, self.hd).transpose(1, 2)
        k = k.view(B, T, HEADS, self.hd).transpose(1, 2)
        v = v.view(B, T, HEADS, self.hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(EMB), nn.LayerNorm(EMB)
        self.attn = CausalSelfAttention()
        self.mlp = nn.Sequential(nn.Linear(EMB, 4 * EMB), nn.GELU(), nn.Linear(4 * EMB, EMB))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(n_vocab, EMB)
        self.pos = nn.Embedding(BLOCK, EMB)
        self.blocks = nn.Sequential(*[Block() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(EMB)
        self.head = nn.Linear(EMB, n_vocab)
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        return self.head(self.ln(self.blocks(x)))

model = TinyGPT()
n_params = sum(p.numel() for p in model.parameters())
print(f"vocab={n_vocab} params={n_params/1e6:.2f}M device={DEV} threads={torch.get_num_threads()}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
WARMUP = 20
STEPS = 150

def lr_at(s):
    if s < WARMUP:
        return 1e-3 * s / WARMUP
    return 1e-3 * (0.1 + 0.9 * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP))) / 2)

t0 = time.time()
for step in range(1, STEPS + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, n_vocab), y.view(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # stability
    opt.step()
    if step % 15 == 0 or step == 1:
        dt = time.time() - t0
        print(f"step {step:4d} loss {loss.item():.4f} lr {lr_at(step):.1e} elapsed {dt:6.1f}s ({dt/step:.2f}s/step)", flush=True)

# sample some text
itos = {i: c for c, i in stoi.items()}
idx = torch.zeros((1, 1), dtype=torch.long)
model.eval()
with torch.no_grad():
    for _ in range(300):
        logits = model(idx[:, -BLOCK:])
        probs = F.softmax(logits[:, -1] / 0.8, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if probs.sum() <= 0:
            probs = torch.ones_like(probs) / probs.numel()
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
print("SAMPLE:\n" + "".join(itos[int(i)] for i in idx[0]))
print("TRAINING DEMO COMPLETE")
