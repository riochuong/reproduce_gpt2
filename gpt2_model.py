import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import numpy as np
from hellwaswag import render_example, iterate_examples
import tiktoken

# ------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|end_of_line|>
    n_layer: int = 12 # number of layers 
    n_head: int = 12 # number of heads 
    n_embd: int = 768 # embedding dims 

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh") # GPT-2 paper use approximate version
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1

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
       # this to control the growth of weights in residual stream   
       self.c_proj.NANO_GPT_SCALE_INIT = 1.0 

       # regularization 
       self.n_head = config.n_head
       self.n_embd = config.n_embd
       # not really a bias but more of a mask, but following the OpenAI/HF naming though
       #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
       #                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
       B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
       # calculate query, key, values for all heads in batch and move head forward to batch
       # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
       # e.g: in GPT-2 (124M) n_head=12, hs=64 so nh*hs=C=768 channels in the Transformer
       qkv = self.c_attn(x)
       q, k, v = qkv.split(self.n_embd, dim=2)
       k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
       q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
       v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
       # attention (metrializes the large (T, T) matrix for all the queries and keys)
    #    att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
    #    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    #    att = F.softmax(att, dim=-1)
    #    y = att @ v
       # FLASH ATTENTION -- (torch.compile cannot find these to optimize just yet)
       y = F.scaled_dot_product_attention(q, k, v, is_causal=True)


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

        # weight sharing scheme -- works better and save a ton of parameters also 
        # what about the shape here 
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        # apply will call self._init_weights on all of module that is part of GPT
        self.apply(self._init_weights)
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # separate into groups that should have weight_decay and should not have weight_decay
        # mostly decay weight that participate in matmul and embeddings
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups, any parameters that is 2D will be weight decay otherwise NO
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,}")
        print(f"num nodecayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,}")

        # Create AdamW optimizer 
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        use_fused = fused_available and 'cuda' in device_type
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    
    def _init_weights(self, module):
        # this is to mimic how gpt2 model from openai initialized weight
        # std=0.02 is in the vincinity of Xavier initialization 
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANO_GPT_SCALE_INIT'):
                # we have 2 residual stream here one to attention and one to mlp
                # so we are scaling down the weights
                std *= (2 * self.config.n_layer) ** -0.5  
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
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
        loss = None
        if targets is not None:
            # need to flatten 3-dim x tensors into 2-dim (input) and 1-dim targets
            # x: (B, T, vocab_size) -> (B*T, vocab_size)
            # y: (B, T) ->  (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
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

        config_args['vocab_size'] = 50304 # 50304 better numbrer UGly number 50257 # always 50257 for GPT model checkpoints
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
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous() # B, T-1, vocab_size
    shift_tokens = (tokens[..., 1:]).contiguous() # B, T-1
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # B*(T-1), vocab_size
    flat_shift_tokens = shift_tokens.view(-1) # B*T-1

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # B*(T-1)
    shift_losses = shift_losses.view(tokens.size(0), -1)

    shift_mask = (mask[..., 1:]).contiguous() # (B, T-1)
    masked_shift_losses = shift_losses * shift_mask # (B*T-1) * tor(B, T-1) 
    sum_loss = torch.sum(masked_shift_losses, dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # convert uint16 to int32 before to long tensor
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
    
    def reset(self):
        # state, init at shard zero 
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T *self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + (B*T) +1]
        #buf = buf.to(device)
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        # update current position
        self.current_position += B * T * self.num_processes
        # if loading next batch would be out of bounds then we reset
        # need to check with B*T+1 to make sure every batch has correct
        # number of inputs vs targets
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y 


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # gpt papers mentions it warms up with 375e6 tokens / 2^19 (which is the batch_size) =  ~715 steps
max_steps = 19073 * 4 # we have about 10^9 tokens and 10^9 / 2^19 = ~19073
# learning rate with cosine decay ratio
# TODO: steps through in ipynb to gain some intuition for this  function  
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)



# ============= IMPLEMENT DISTRIBUTED DATA PARALLEL TO RUN ON MULTIPLE GPUS ==============
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORDL_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run ??? a hacky way to check if it's a ddp run
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "DDP might only support CUDA"
    init_process_group(backend='nccl')
    # important and unique for each GPU so we can use it to make sure they are not running on the same data # setup code
    ddp_rank = int(os.environ['RANK'])
    # this is used for multinode settings. it is the rank of the GPU in the single node 
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # this is number of GPU 
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device_type = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device_type)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing

else:
    # vanilla, non-DDP run 
    ddp_rank = 0
    ddp_local_rank= 0
    ddp_world_size = 1
    master_process = True
    device_type = "cpu"
    if torch.cuda.is_available():
        device_type = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
    print(f"using device: {device_type}")

# NOTES: 
# imagine there are 8 GPUS are running at the same time so we need to adjust BATCH, SIZE
# =========================================================================================

enc = tiktoken.get_encoding("gpt2")
# from GPT3 paper the Batchsize is 0.5M 
# in order to respect it with small GPU 
# we need to implement gradient accummulation step
total_batch_size = 524288 # 2**19, ~0.5M nice power of 2 number 
B = 64 # microbatch size 
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0 , "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # each process will do B*T but we have ddp_world_size of GPUS
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> caclculated gradient accumulation steps: {grad_accum_steps}")


# attempt to auto detect the device
device_type = "cpu"
if torch.cuda.is_available():
    device_type = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device_type = "mps"
print(f"Using device: {device_type} ")

num_return_sequences = 5
max_length = 30

# switch to default random model
model = GPT(GPTConfig)
print(f"Loaded GPT-2 !!! YAYYAYAYAY")
model.eval()
model.to(device_type)
# NOT using compile for now so we can evaluate hellwaswag
#model = torch.compile(model)
if ddp:
    # in the forward pass, it will be identical as single GPU
    # in the backward pass, when backward pass is complete 
    # DDP does an average gradient across on GPU rank and 
    # deposit it across the ranks
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# training loop
# optimizer: AdamW is the bug fixes of Adam per doc from Kaparthy
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type) 
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
# not sure if it's available on my old gpu --> so we wont get 8x speedup
torch.set_float32_matmul_precision('high')
import time


# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


# torchrun --standalone --nproc_per_node=8 gpt2_model.py

for step in range(max_steps):
    if master_process:
        print(f"Start step {step}")
    t0 = time.time() 
    last_step = (step == max_steps - 1)

    # once in a while evaluate validation loss
    if (step > 0) and  (step % 250 == 0):
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device_type), y.to(device_type)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss /val_loss_steps
                val_loss_accum += loss
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
    
    # once in a while evaluate hellaswag
    if ((step > 0) and (step % 250 == 0)) or last_step:
        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process example where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render example in to tokens 
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device_type)
            mask = mask.to(device_type)
            # get logits 
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device_type)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device_type)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
        

    # once in a while we sample from our model 
    if step > 0 and step % 100 == 0:
        model.eval()
        num_return_sequences = 4 
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device_type)
        sample_rng = torch.Generator(device=device_type)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_sie)
                # take the logit at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilitlies
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indics is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities 
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) 
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=-1)
        # print the generate
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # train loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device_type), y.to(device_type)
        # little hacky that we only allow sync at last step
        # disable syncrhonization between GPUs for microstep here as it is wasteful
        # we can wait until the gradient finish accumuated and then synchronize across GPUs
        # NOTE: added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # train with bfloat16 on some parts of the calculations 
        with torch.autocast(device_type=device_type , dtype=torch.bfloat16):
            _, loss = model(x, y) 
        # keep deposit gradient for each of the batch here 
        # the F.crossentropy loss using the "reduction" by "mean" to calculate the loss 
        # therefore, we need to add the mean factor here. Need to recover the original 
        # normalizer
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() 
        loss.backward()
    # currently loss_accum only printed the local loss of master process
    # we want loss_accum to be the average across all GPUS
    if ddp:
        # will create an average of loss_accum tensor across all the rank
        # and deposit to loss_accum on each GPUs
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # HERE we have average gradient across GPUs 
    # follow GPT3 paper to clip the gradient to prevent shocking the model learning process with shooting gradient
    norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
    # determine and set learning rate 
    lr = get_lr(step)
    # pytorch has the notion of group params so we must set the lr this way
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    # wait for gpu to execute all kernels/works before getting the time
    if device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    token_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = token_processed / dt
    #print(f"dt: {(t1 - t0)*1000}ms")
    # we want to see that we can crush this little batch 
    # step 41, loss: 0.0025385613553225994
    # step 42, loss: 0.0024354925844818354
    # step 43, loss: 0.002342457417398691
    # step 44, loss: 0.002258246298879385
    # step 45, loss: 0.002181940246373415
    # step 46, loss: 0.0021126489154994488
    # step 47, loss: 0.0020495580974966288
    # step 48, loss: 0.001992021454498172
    # step 49, loss: 0.0019393289694562554
    #if (i % 10) == 0:
    if master_process:
        print(f"step {step} | lr: {lr} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | dt: {dt*1000}ms | tok/sec: {tokens_per_sec}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum}\n")
        if step > 0 and (step % 5000 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
            # you might also want to add optimizer.state_dict() and
            # rng seeds etc., if you wanted to more exactly resume training
            torch.save(checkpoint, checkpoint_path)

if ddp:
    destroy_process_group()



 




