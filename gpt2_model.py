import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
import math

# ------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|end_of_line|>
    n_layer: int = 6 # number of layers 
    n_head: int = 6 # number of heads 
    n_embd: int = 384 # embedding dims 



class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh") # GPT-2 paper use approximate version
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
       super().__init__()
       assert config.n_embd % config.n_head == 0 # check that we can divide n_emb into multihead attn
       # key, query, value projections for all heads, but in batch
       self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
       # output projection 
       self.c_proj = nn.Linear(config.n_embd, config.n_embd)
       # regularization 
       self.n_head = config.n_head
       self.n_embd = config.n_embd
       # not really a bias but more of a mask, but following the OpenAI/HF naming though
       self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
       B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
       # calculate query, key, values for all heads in batch and move head forward to batch
       # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
       # e.g: in GPT-2 (124M) n_head=12, hs=64 so nh*hs=C=768 channels in the Transformer
       qkv = self.c_attn(x)
       q, k, v = qkv.split(self.n_emb, dim=2)
       k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
       q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
       v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
       # attention (metrializes the large (T, T) matrix for all the queries and keys)
       att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
       att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
       att = F.softmax(att, dim=-1)
       y = att @ v
       # make tensor layout in contiguous memory ??? 
       y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side-by-side
       # output projection
       y = self.c_proj(y)
       return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # the slight different here versus
        # the architectur from original Attention is All You Need is that 
        # we directly add residual path to output for attention head amd mlp without 
        # normalization interrupt in between like original paper.
        # This will help to keep the gradient flow directly from input to 
        # mlp and help optimization a lot 

        # NOTE: we can think of attention is a weighted sum or a reduced function
        # and MLP is mapping function -> MAP/REDUCE block 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                
                # this reflects there are 12 layers in the transformer 
                # Example of 1 layer:
                # transformer.h.0.ln_1.weight torch.Size([768])
                # transformer.h.0.ln_1.bias torch.Size([768])
                # transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
                # transformer.h.0.attn.c_attn.bias torch.Size([2304])
                # transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
                # transformer.h.0.attn.c_proj.bias torch.Size([768])
                # transformer.h.0.ln_2.weight torch.Size([768])
                # transformer.h.0.ln_2.bias torch.Size([768])
                # transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
                # transformer.h.0.mlp.c_fc.bias torch.Size([3072])
                # transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
                # transformer.h.0.mlp.c_proj.bias torch.Size([768])
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

                ln_f = nn.LayerNorm(config.n_embd)
            )
        )

        #language model head to project from embedding space back to vocab 
        # GPT also use no bias 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx):
        # idx is of shape (B, T) --> T is the index of embeddings that is used for looking up
        B, T = idx.size()

        # forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos) # position embeddings (T, n_embd)
        tok_embd = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_embd + pos_embd
        # forward the blocks of the transformer:
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    @classmethod
    def from_pretrained(cls, model_type: str):
        """ Loads pretrained GPT2-model weights from hugging face """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights pretrained from gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
                'gpt2':             dict(n_layer=12, n_head=12, n_embd=768), # 124M params
                'gpt2-medium':      dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':       dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':          dict(n_layer=12, n_head=12, n_embd=768), # 124M params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer, not a param 

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy weights from huggingface model while ensuring all of the parameters
        # are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys() 
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] # ingore these masks
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")] # similar as above
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use "Conv1D" module, but we only want to use vinalla Linear
        # this means that we have to transpose these weights when we import them
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over other params
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model

# -------------------------------------------------------------------------------------------

model = GPT.from_pretrained('gpt2')
print(f"Loaded GPT-2 !!! YAYYAYAYAY")






