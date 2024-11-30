#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from mamba_ssm import Mamba
import torch.nn.init as init


# In[2]:


from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    print("causal_conv1d not found")
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    print("selective_state_update not found")
    selective_state_update = None


# In[3]:


import torch
class SampleGenerator(object):
    """Construct dataset for multi-class classification"""

    def __init__(self, ratings, n_test=1):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        # self.preprocessed_ratings = self._preprocess_ratings(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.ratings, n_test)

    def _split_loo(self, ratings, n_test=1):
        """
        Split dataset into train and test sets, with `n_test` interactions per user in the test set.

        Args:
            ratings: DataFrame, contains user-item interactions.
            n_test: int, number of recent interactions to include in the test set.

        Returns:
            train: DataFrame, contains the train set.
            test: DataFrame, contains the test set.
        """
        # Rank interactions by timestamp for each user
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

        # Test set contains the top `n_test` interactions for each user
        test = ratings[ratings['rank_latest'] <= n_test]
        
        # Train set contains all other interactions
        train = ratings[ratings['rank_latest'] > n_test]

        print(train['userId'].nunique())
        print(test['userId'].nunique())

        # Ensure every user is in both train and test sets
        # assert train['userId'].nunique() == test['userId'].nunique()

        # Return train and test dataframes
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def evaluate_data(self):
        """Create evaluate data for classification"""
        test_users = self.test_ratings['userId'].tolist()
        test_items = self.test_ratings['itemId'].tolist()
        test_ratings = self.test_ratings['rating'].tolist()

        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(test_ratings)]
    
    def get_train_test_dataframes(self):
        """
        Returns the train and test dataframes.
        """
        return self.train_ratings, self.test_ratings


# In[4]:


import pandas as pd
import numpy as np
from IPython.display import display

# Load Data
data_dir = '/raid/home/shyamsg/Final_Project/DecoderOnlyModel/Data/food.dat'
rating_df = pd.read_csv(data_dir, sep='\t', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

# i want to drop nulls
rating_df = rating_df.dropna()

# Reindex userId and itemId, start from 0
user_id = rating_df[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
rating_df = pd.merge(rating_df, user_id, on=['uid'], how='left')

# Reindex itemId, start from 0
item_id = rating_df[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
rating_df = pd.merge(rating_df, item_id, on=['mid'], how='left')

rating_df = rating_df[['userId', 'itemId', 'rating', 'timestamp']]

# i want to change 0 ratings to 1
rating_df['rating'] = rating_df['rating'].apply(lambda x: 1 if x == 0 else x)

print('Range of userId is [{}, {}]'.format(rating_df.userId.min(), rating_df.userId.max()))
print('Range of itemId is [{}, {}]'.format(rating_df.itemId.min(), rating_df.itemId.max()))


# n_test specifies number of most recent interactions to be used for testing for each user
sample_generator = SampleGenerator(rating_df, n_test=2)

# Get train and test dataframes
df1, df2 = sample_generator.get_train_test_dataframes()

print(df1.shape, df2.shape)


# In[5]:


# now create a new dataframe columns
#df['prompt'] = What star rating do you think userid_df[user_id] will give item_df[item_id]? 
#df['rating'] = df[rating]

train_df = pd.DataFrame()
train_df['prompt'] = "What star rating do you think user_" + df1['userId'].astype(str) + " will give item_" + df1['itemId'].astype(str) + "?"
train_df['rating'] = df1['rating']
print(train_df.shape)   

val_df = pd.DataFrame()
val_df['prompt'] = "What star rating do you think user_" + df2['userId'].astype(str) + " will give item_" + df2['itemId'].astype(str) + "?"
val_df['rating'] = df2['rating']
print(val_df.shape)

print(f"Train Size: {len(train_df)}, Val Size: {len(val_df)}")
          


# In[6]:


import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, GPT2ForSequenceClassification


################################################################################
# Transformer Model classes with Mamba layers
################################################################################

@dataclass
class MambaConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)   
        return x

class Mamba_custom(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Mamba_custom(config.n_embd)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MAMBA(nn.Module):
    def __init__(self, model_type='mamba', is_gen=False):
        super(MAMBA, self).__init__()
        
        # n_layer and n_embd are determined from model_type
        config_args = {
            'mamba':         dict(n_layer=12,n_embd=768),  # 124M params
            'mamba-small':  dict(n_layer=6,  n_embd=768), # 350M params
            'mamba-xsmall':   dict(n_layer=3, n_embd=768), # 774M params
            'mamba-singlelayer': dict(n_layer=1,n_embd=768), # 1558M (1.3B) params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True        # always True for GPT model checkpoints
        

        self.config = MambaConfig(**config_args)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))
        if is_gen:
            self.lm_head = nn.Linear(
                self.config.n_embd, self.config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
            self.score = None
        else:
            self.score = nn.Linear(self.config.n_embd, 5, bias=False)

        # TODO: Remove gradients from the embedding layers

        # added by me below
        # self.transformer.wte.weight.requires_grad = False
        # self.transformer.wpe.weight.requires_grad = False
        # added by me above 

        # Initialize weights with Xavier initialization\n",
        self.apply(self._init_weights)
    
        # print the total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {total_params / 1e6:.2f}M")
        if not is_gen:
            # print the number of trainable parameters
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
            # calculate the reduction in parameters
            reduction = 100 * (total_params - num_params) / total_params
            print(f"Reduction: {reduction:.2f}%")
    
    def _init_weights(self, module):
    # Check if the module is either a Linear or Embedding layer
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Apply Xavier (Glorot) uniform initialization to the weights
            init.xavier_uniform_(module.weight)
            
            # If the module has a bias, initialize it to zero
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

        # Check if the module is a LayerNorm
        elif isinstance(module, nn.LayerNorm):
            # Set the bias to zero and the weight to 1 for LayerNorm
            module.bias.data.fill_(0)
            module.weight.data.fill_(1.0)
        
        # If the module is a ModuleList, recursively initialize its weights
        elif isinstance(module, nn.ModuleList):
            for submodule in module:
                self._init_weights(submodule)


    def forward(self, idx, mask=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # if mask is provided, find the indices of the last tokens in each sequence
        if mask is not None:
            assert mask.size() == idx.size(), "Mask size must match input size"
            eos_idxs = mask.sum(1) - 1 # last non-pad token in each sequence

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if self.score is None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        else: # no need to preserve the time dimension for classification task
            if mask is not None:
                # if mask is provided, only return the logits for the last token in each sequence
                logits = self.score(x[torch.arange(b, device=device), eos_idxs])
            else:
                
                logits = self.score(x[:, -1, :])
               
        # return logits.squeeze(-1)
        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            
            # print(f"shape of idx in generate: {idx.shape}") # added by me 
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def save_trainable_params(self, path):
        trainable_params =\
            list(filter(lambda p: p.requires_grad, self.parameters()))
        torch.save(trainable_params, path)
    
    def load_trainable_params(self, path):
        trainable_params = torch.load(path)
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = trainable_params.pop(0)



# In[7]:


def get_data_loader(data, batch_size, tokenizer, shuffle=True, max_len=30):
    """
    Get a data loader for the training data.
    """
    X, y = data['prompt'], data['rating']
    y= y-1
    X = tokenizer.batch_encode_plus(
        X.tolist(), max_length=max_len, truncation=True, padding='max_length')
    X, mask = X['input_ids'], X['attention_mask']
    # convert them to tensors
    X = torch.tensor(X)
    mask = torch.tensor(mask)
    y = torch.tensor(y.values, dtype=int)
    data = torch.utils.data.TensorDataset(X, mask, y)
    
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader


# In[8]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


# In[9]:


import torch
import time
import matplotlib.pyplot as plt


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_pred = 0
    total_pred = 0
    for batch in train_loader:
        optimizer.zero_grad()
        X, mask, y = batch
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        output = model(X, mask) # return logits for last valid token
        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct_pred += (predicted == y).sum().item()
        total_pred += len(y)
    
    return total_loss / len(train_loader), correct_pred / total_pred 
  
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_pred = 0
    total_pred = 0
    with torch.no_grad():
        for batch in val_loader:
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            output = model(X, mask)
            loss = criterion(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_pred += (predicted == y).sum().item()
            total_pred += len(y)
             
    return total_loss / len(val_loader) , correct_pred / total_pred 

    
def plot_losses(train_losses, val_losses, mode, args):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{mode}_{args.mamba_variant} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'../plots_food/{args.mamba_variant}_loss.png')
    plt.close()
    print(f"Plots saved at plots/{args.mamba_variant}_loss.png")
    
def plot_metrics(train_accs, val_accs, mode, args):
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'{mode} Accuracy')
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'../plots_food/{args.mamba_variant}_acc.png')
    plt.close()
    print(f"Plots saved at plots/{args.mamba_variant}_acc.png")


# In[10]:


train_loader = get_data_loader(train_df, 256, tokenizer, shuffle=True)
val_loader = get_data_loader(val_df, 256, tokenizer, shuffle=False)

print("Length of train_loader: ", len(train_loader))
print("Length of val_loader: ", len(val_loader))


# In[ ]:


import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Mamba Model Training and Generation")
    parser.add_argument("--mode", type=str, default="mamba", choices=["mamba", "gen"], help="Mode of operation: 'mamba' for training, 'gen' for generation")
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID to use")
    parser.add_argument("--mamba_variant", type=str, default="mamba-singlelayer", choices=["mamba", "mamba-small", "mamba-xsmall", "mamba-singlelayer"], help="Variant of the Mamba model")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/mamba.pth", help="Path to save the trained model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    return parser.parse_args()

args = parse_args()
args.device = torch.device(
    f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else\
    "mps" if torch.backends.mps.is_available() else "cpu")

def main(args):
    if args.mode == "gen":
        model = MAMBA(args.mamba_variant, is_gen=True).to(args.device)
        model.eval()

        # TODO: You can add your super creative prompt here
        prompt = "My name is Inigo Montoya. You killed my father. Prepare to die. "
        # prompt = "Once upon a time, in a land far far away, there was a"

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "mamba":    
        model = MAMBA(args.mamba_variant).to(args.device)
        
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(args.epochs):
            # i want some seperation between print statements of epochs
            print('-' * 80)
            epoch_start = time.time()

            train_start= time.time()
            train_loss,train_acc = train(model, train_loader, optimizer, criterion, args.device)
            train_end= time.time()

            val_start= time.time()
            val_loss,val_acc = evaluate(model, val_loader, criterion, args.device)
            val_end= time.time()

            epoch_end = time.time()

            # Calculate the durations
            train_duration = train_end - train_start
            val_duration = val_end - val_start
            epoch_duration = epoch_end - epoch_start
    
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Time taken for epoch {epoch+1}: {epoch_duration:.4f} seconds (Training: {train_duration:.4f} seconds, Evaluation: {val_duration:.4f} seconds)")
            # save the model after every 10 epochs in model directory with different names use save_trainable_params
        #     if (epoch) % 25 == 0:
        #         model.save_trainable_params(f'models/{args.mamba_variant}_{epoch+1}.pth')
        # # save the train_losses and val_losses lists in some file so that we can plot them later
        with open(f'../results_food/{args.mamba_variant}_metrics.txt', 'w') as f:
            f.write('Train Losses:\n')
            f.write(str(train_losses))
            f.write('\nVal Losses:\n')
            f.write(str(val_losses))
            f.write('\nTrain Accuracies:\n')
            f.write(str(train_accs))
            f.write('\nVal Accuracies:\n')
            f.write(str(val_accs))
        print(f"Metrics saved at model/{args.mamba_variant}_metrics.txt")
        # save the plot in plots folder 
        plot_losses(train_losses, val_losses, args.mode, args)
        plot_metrics(train_accs, val_accs, args.mode, args)
        # model.save_trainable_params(args.model_path)
    else:
        print("Invalid mode")
        return

if __name__ == "__main__":
    main(args)

