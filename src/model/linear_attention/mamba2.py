"""
Linear attention classes
"""
import sys
#sys.path.insert(0, "/data/hongyuantao/data/SOLO")
from typing import List, Tuple, Optional
import copy
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from einops import rearrange, repeat
import torch.nn.functional as F
import math


from transformers.cache_utils import Cache  # starting at Transformers v4.36

# Causal linear attention dot product CUDA kernel from fast-transformers
try:
    from csrc import causal_dot_product as fast_causal_dot_product
except ImportError:
    fast_causal_dot_product = None

from src.model.feature_map import init_feature_map, init_learned_kernel
from src.model.rotary import get_rotary_embeddings, apply_rotary_pos_emb
from .utils import repeat_kv

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules.activations import ACT2FN

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def segsum(x):
    """More stable segment sum calculation."""
    # [1, 2, 3]
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    # [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    # [[0, 0, 0], [2, 0, 0], [3, 3, 0]]
    x_segsum = torch.cumsum(x, dim=-2)
    # [[0, 0, 0], [2, 0, 0], [5, 3, 0]]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd_minimal_discrete(X, A_log, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A_log: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
        block_len: int
        initial_states: (batch, n_heads, d_state, d_state) or None
    Return:
        Y: (batch, length, n_heads, d_head)
        final_state: (batch, n_heads, d_head, d_state)
    """
    assert X.dtype == A_log.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0
    batch_size, length, n_heads, d_head = X.shape
    d_state = B.shape[-1]
    assert A_log.shape == (batch_size, length, n_heads)
    assert B.shape == C.shape == (batch_size, length, n_heads, d_state)

    # Rearrange into blocks/chunks
    X, A_log, B, C = [
        rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A_log, B, C)
    ]

    A_log = rearrange(A_log, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A_log, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    length = torch.exp(segsum(A_log))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, length, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    else:
        # Rearrange initial states into blocks
        initial_states = rearrange(initial_states, "b h d s -> b 1 h d s")

    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state

def step(x, B, C, A_log, state):
    """
    Arguments:
        x: (batch, n_v_heads, dim)
        B: (batch, n_qk_heads, d_state)
        C: (batch, n_qk_heads, d_state)
        A_log: (batch, length, n_heads)
        state: dict
    Return:
        y: (batch, n_v_heads, dim)
        ssm_state: (batch, n_v_heads, d_state)
    """
    n_v_heads = x.shape[1]
    n_qk_heads = B.shape[1]
    assert n_v_heads % n_qk_heads == 0
    

    # Broadcast B and C across the head dimension (n_v_heads % n_qk_heads == 0)
    B = B.repeat_interleave(n_v_heads // n_qk_heads, dim=1)
    C = C.repeat_interleave(n_v_heads // n_qk_heads, dim=1)

    # SSM step
    Bx = torch.einsum("bhn,bhp->bhpn", B, x)
    Ah = torch.einsum(
        "bh,bhpn->bhpn",
        torch.sigmoid(-A_log).to(x.dtype),  # = torch.exp(-F.softplus(A_log))
        state["ssm"].to(x.dtype),
    )
    ssm_state = Ah + Bx
    y = torch.einsum("bhn,bhpn->bhp", C, ssm_state)
    return y, ssm_state

def materialize_mixer(A_log, B, C, D):
    """
    Since the transfer matrix will be equated to the attention matrix,
    we need to support the form: torch.matmul(attn_weights, value_states).
    Thus, y = torch.matmul(T, X)
    Arguments:
        A_log: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        T: (batch, n_heads, length, length)
    """
    batch_size, length, n_heads, d_state = B.shape
    assert A_log.shape == (batch_size, length, n_heads)
    assert B.shape == C.shape == (batch_size, length, n_heads, d_state)

    # Compute:
    A_log = rearrange(-F.softplus(A_log), "b l h -> b h l")
    powers = torch.exp(segsum(A_log))
    T = torch.einsum("blhn,bshn,bhls->bhsl", C, B, powers)

    # Add D:
    if D is not None:
        T[:, :, torch.arange(length), torch.arange(length)] += D.view(1, n_heads, 1)

    T = rearrange(T, "b h z l -> b h l z")
    return T


# -------------------
# Attention functions
# -------------------

def causal_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Causal linear attention dot product
    - If available, use CUDA kernel from fast-transformers
    """
    fast_causal_dot_product=None
    if fast_causal_dot_product is None:
        kv = torch.einsum('bhlf,bhld->bhlfd', k, v)
        return torch.einsum('bhlf,bhlfd->bhld', q, kv.cumsum(dim=2))
    return fast_causal_dot_product(q, k, v)

def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     fp32_attention: bool = False, eps: float = 1e-12,
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Compute linear attention with CUDA kernel implementation from fast-transformers
    - https://github.com/idiap/fast-transformers
    - Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim); 
      v is shape (b, h, l, head_dim)
    """
    dtype = q.dtype
    # Causal mask already applied
    y = causal_dot_product(q.contiguous().to(dtype=torch.float32),
                           k.contiguous().to(dtype=torch.float32),
                           v.contiguous().to(dtype=torch.float32))
    if fp32_attention:
        y = (y / (torch.einsum(
            "bhld,bhld->bhl", q.float(), k.float().cumsum(dim=2)
        ) + eps)[..., None]).to(dtype=dtype)
    else:
        y = y.to(dtype=dtype)
        k = k.float().cumsum(dim=2).to(dtype=dtype)
        y = y / (torch.einsum("bhld,bhld->bhl", q, k) + eps)[..., None]
    return y, None, None






# ---------------------
# Attention layer class
# ---------------------

class Mamba2_Attention(nn.Module):
    """
    LoLCATs attention implementation initialized from a 
    `LlamaAttention` or `MistralAttention` object (base_attn)

    Most of the arguments are directly tied to argparse args
    - For now we don't support padding.
    """
    def __init__(self,
                 base_attn: nn.Module,  # like LlamaAttention
                 layer_idx: Optional[int] = None,
                 max_layer_idx: Optional[int] = None,
                 tie_qk_kernels: Optional[bool] = False,
                 rotary_config: Optional[dict] = None,
                 train_attention: Optional[bool] = False,
                 remove_base_attn: Optional[bool] = True,
                 attention_type: Optional[str] = 'lolcats_llama',
                 gate_logit_normalizer: int = 16,
                 mask_value: int = 0,
                 eps: float = 1e-12,
                 fp32_attention: bool = False,
                 track_state_grads: bool = False,
                 rank: Optional[int] = 0,
                 use_conv: Optional[bool] = False,
                 use_qknorm: Optional[bool] = False,
                 use_gnorm: Optional[bool] = True,
                 use_A: Optional[bool] = True,
                 use_D: Optional[bool] = False,
                 inherit_qkv: Optional[bool] = False,
                 mimic_init: Optional[bool] = False,
                 stage1: Optional[bool] = False,
                 stage2: Optional[bool] = True,
                 **kwargs: any) -> None:
        super().__init__()
        self.base_config = getattr(base_attn, 'config', None)
        if self.base_config is not None:
            self.base_config = self.base_config.to_dict()
        self.gate_logit_normalizer=gate_logit_normalizer
        self.attention_type = attention_type
        self.mask_value = mask_value
        self.eps = eps
        self.layer_idx = (layer_idx if layer_idx is not None else base_attn.layer_idx)
        self.max_layer_idx = max_layer_idx
        self.tie_qk_kernels = tie_qk_kernels
        self.train_attention = train_attention
        self.base_inference = False
        self.fp32_attention = fp32_attention
        self.track_state_grads = track_state_grads

        self.use_conv = use_conv
        self.use_qknorm = use_qknorm
        self.use_D = use_D
        self.stage1 = stage1
        self.stage2 = stage2
        self.use_gnorm = use_gnorm
        self.use_A = use_A
        self.inherit_qkv = inherit_qkv
        self.mimic_init = mimic_init
        self.bias = False
        self.chunk_size = 128
        conv_bias = True
        self.conv_bias = conv_bias
        self.d_conv = 2
        self.activation="silu"
        self.train_stage="1"


        self.remove_base_attn = remove_base_attn

        # Rotary embeddings (patch for Llama 3.1, Transformer v4.43.0)
        self.rotary_config = rotary_config
        if isinstance(self.rotary_config, DictConfig):  # ensure dict
            self.rotary_config = OmegaConf.to_container(self.rotary_config)
        
        self.rotary_emb = None
        if self.base_config is not None and self.rotary_config is None:
            self.rotary_emb = base_attn.rotary_emb

        self.init_weights(base_attn, remove_base_attn)

    def init_weights(self, 
                      base_attn: nn.Module, 
                      remove_base_attn: bool = True,
                      elementwise_affine: Optional[bool] = True,
                      norm_eps: float = 1e-5,):
        """
        Initialize module layers, weights, positional dependencies, etc. 
        from original softmax attention layer (base_attn)
        """
        # Make other attributes accessible
        self.attention_dropout = 0  # We don't use dropout
        self.hidden_size = base_attn.hidden_size
        self.num_heads = base_attn.num_heads
        self.head_dim = base_attn.head_dim
        self.num_key_value_heads = base_attn.num_key_value_heads
        self.num_key_value_groups = base_attn.num_key_value_groups

        self.q_shape = [self.num_heads, self.head_dim]
        self.k_shape = [self.num_key_value_heads, self.head_dim]
        self.v_shape = [self.num_key_value_heads, self.head_dim]
        try:
            self.q_proj = base_attn.q_proj
            self.k_proj = base_attn.k_proj
            self.v_proj = base_attn.v_proj
            self.o_proj = base_attn.o_proj
            self.mode = None
        except:
            #self.wqkv = base_attn.wqkv
            # import ipdb; ipdb.set_trace()
            self.o_proj = base_attn.wo
            self.mode = "InternLM"
            wqkv_weight = base_attn.wqkv.weight

            # 计算每个部分的大小
            self.num_q_heads = self.num_heads * self.head_dim
            self.num_kv_heads = self.num_key_value_heads * self.head_dim

            wqkv_weight = rearrange(
                wqkv_weight,
                '(h gs d) q -> h gs d q',
                gs=2 + self.num_key_value_groups,
                d=self.head_dim,
            )

            self.q_weight = rearrange(wqkv_weight[:, : self.num_key_value_groups, :, :], 'h gs d q-> (h gs d) q')
            self.k_weight = rearrange(wqkv_weight[:, -2, :, :], 'h d q-> (h d) q')
            self.v_weight = rearrange(wqkv_weight[:, -1, :, :], 'h d q-> (h d) q')

            self.q_proj = nn.Linear(self.hidden_size, self.num_q_heads, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads, bias=False)

            if self.inherit_qkv:
                self.q_proj.weight.data = self.q_weight
                self.k_proj.weight.data = self.k_weight
                self.v_proj.weight.data = self.v_weight


        if self.mode == None:
            self.device = base_attn.q_proj.weight.device
            self.dtype = base_attn.q_proj.weight.dtype
        else:
            self.device = base_attn.wqkv.weight.device
            self.dtype = base_attn.wqkv.weight.dtype

        # Rotary embeddings
        if self.rotary_emb is None:
            self.max_position_embeddings = base_attn.max_position_embeddings
            scaling_factor = getattr(base_attn.rotary_emb, 'scaling_factor', 1.)
            if self.rotary_config is None:
                self.rotary_emb = get_rotary_embeddings(
                    rope_scaling_type=None,
                    head_dim=self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,  # base_attn.rotary_emb.max_position_embeddings,
                    rope_theta=base_attn.rotary_emb.base,
                    rope_scaling_factor=scaling_factor,  # base_attn.rotary_emb.scaling_factor,
                    device=self.device,
                )
            else:
                if 'device' not in self.rotary_config:
                    self.rotary_config['device'] = self.device
                self.rotary_emb = get_rotary_embeddings(**self.rotary_config)
        
        try:  # If wanting to use FA2 for ground-truth inference
            self._flash_attn_uses_top_left_mask = base_attn._flash_attn_uses_top_left_mask
        except AttributeError: 
            pass
        
        if self.use_conv:
            conv_dim = self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim
            
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=self.conv_bias,
                kernel_size=self.d_conv,
                groups=conv_dim,
                padding=self.d_conv - 1,
                device=self.device, 
                dtype=self.dtype
            )
            if self.mimic_init:
                with torch.no_grad():
                    self.conv1d.weight.zero_()  
                    self.conv1d.weight[:, 0, 1] = 1 
                    self.conv1d.bias.zero_()  

        # Activation after conv
        if self.activation == "identity":
            self.act = nn.Identity()
        elif self.activation in ["silu", "swish"]:
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation {self.activation}")

        self.in_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.num_heads,
            bias=self.bias,
        ).to(self.dtype).to(self.device)
        if self.mimic_init: 
            nn.init.zeros_(self.in_proj.weight)

        if self.use_gnorm:
            self.g_norm = RMSNorm(hidden_size=self.head_dim, elementwise_affine=elementwise_affine, eps=norm_eps).to(self.dtype).to(self.device)
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, device=self.device, dtype=self.dtype)
            # self.gate_fn = ACT2FN["swish"]
            self.gate_fn = ACT2FN["swish"]
            nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        #nn.init.zeros_(self.g_proj.weight)
        #nn.init.constant_(self.g_proj.bias, 1.28)

        dt = torch.exp(
            torch.rand(self.num_heads, dtype=self.dtype, device=self.device) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        dt = torch.clamp(dt, min=0.001)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        if self.mimic_init:
            self.dt_bias = nn.Parameter(inv_dt)
        else:
            self.dt_bias = nn.Parameter(torch.zeros_like(inv_dt))
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if self.use_A:
            if self.mimic_init:
                A_log = torch.ones(self.num_heads, dtype=self.dtype, device=self.device)
                self.A_log_bias = nn.Parameter(A_log)
                self.A_log_bias._no_weight_decay = True
            else:
                A_init_range = (1, 16)
                A = torch.empty(self.num_heads, dtype=torch.float32, device=self.device).uniform_(*A_init_range)
                A_log = torch.log(A).to(dtype=self.dtype)
                self.A_log_bias = nn.Parameter(A_log)
                self.A_log_bias._no_weight_decay = True

        if self.use_qknorm:
            self.q_norm = nn.LayerNorm(self.head_dim, device=self.device, dtype=self.dtype)
            self.k_norm = nn.LayerNorm(self.head_dim, device=self.device, dtype=self.dtype)

        if self.use_D:
            self.D = nn.Parameter(torch.ones(self.num_heads, device=self.device, dtype=self.dtype))
            self.D._optim = {"weight_decay": 0.0}

        if self.remove_base_attn or remove_base_attn:
            del base_attn  # We don't need to keep these around
        else:
            self.base_attn = base_attn  # For some training runs helpful to just call

    
    def softmax_attention(self, q: torch.Tensor, k: torch.Tensor, v: Optional[torch.Tensor] = None, 
                        causal: bool = True, fp32_attention: bool = True, stage1 = False, stage2 = True,
                        position_ids: Optional[torch.LongTensor] = None,
                        past_key_value: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Standard softmax attention; only compute outputs if v is not None
        -> Assume q, k, v are shape (batch_size, num_heads, seq_len, head_dim)
        """
        kv_seq_len = k.shape[-2]
        # Shape is (batch_size, seq_len, num_heads, head_dim)
        
        if past_key_value is not None:  #  and k.shape[2] > q.shape[2]:  # e.g., when generating
            past_key_value.window_size = getattr(self, 'decode_window_size', None)  # self.decode_window_size
            if isinstance(past_key_value, Cache):  # In Transformers v4.36+ this is a DynamicCache object
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        # Apply rotary embeddings and repeat for GQA
        if position_ids is not None and kv_seq_len <= position_ids[0, -1]:
            kv_seq_len = position_ids[0, -1] + 1  # hack for adjusting position ids
        try: # As in Transformers v4.36
            cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        except TypeError:  # As in Transformers v4.39+
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups) 
        v = repeat_kv(v, self.num_key_value_groups)

        a = torch.einsum('bhmd,bhnd->bhmn', q, k) * (k.shape[-1] ** -0.5) #/ math.sqrt(k.shape[-1])
        if causal:  # Apply causal mask
            m, n = a.shape[-2:]
            causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
            a = a.masked_fill(causal_mask, -torch.finfo(a.dtype).max)
        if fp32_attention:
            a = torch.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        else:
            a = torch.softmax(a, dim=-1)
        y = torch.einsum('bhmn,bhnd->bhmd', a, v)
        a = None
        return y, a

    def forward(self,
                hidden_states: torch.Tensor,
                vision_patch_indices: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,  # "legacy" cache approach
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass modified from transformers.models.mistral.modeling_mistral (v4.36)
        - Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        
        hidden_states = hidden_states.to(self.dtype)

        if self.train_stage == "1":
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q = q.view(b, l, *self.q_shape).transpose(1, 2)
            k = k.view(b, l, *self.k_shape).transpose(1, 2)
            v = v.view(b, l, *self.v_shape).transpose(1, 2)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q = q.view(b, l, *self.q_shape).transpose(1, 2)
            k = k.view(b, l, *self.k_shape).transpose(1, 2)
            v = v.view(b, l, *self.v_shape).transpose(1, 2)
            q_shadow = self.q_proj_shadow(hidden_states)
            k_shadow = self.k_proj_shadow(hidden_states)
            v_shadow = self.v_proj_shadow(hidden_states)
            q_shadow = q_shadow.view(b, l, *self.q_shape).transpose(1, 2)
            k_shadow = k_shadow.view(b, l, *self.k_shape).transpose(1, 2)
            v_shadow = v_shadow.view(b, l, *self.v_shape).transpose(1, 2)

        if self.base_inference:
            with torch.no_grad():
                # 1. Compute "ground-truth" attention output and weights
                y_true, attn_true= self.softmax_attention(q, k, v, causal=True, position_ids=position_ids, past_key_value=past_key_value)
                y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                y_true = self.o_proj(y_true)

        elif self.train_attention:  # Distilling / learning attentions
            # Note for now we assume no padding when distilling; attention masks only enforce causality
            # assert output_attentions is True, f'When training feature maps, output_attentions should be True but is {output_attentions}'
            with torch.no_grad():
                # 1. Compute "ground-truth" attention output and weights
                if self.train_stage == "1":
                    _y_true, attn_true = self.softmax_attention(q.clone(), k.clone(), v.clone(), causal=True, fp32_attention=False, stage1=self.stage1, stage2=self.stage2, position_ids=position_ids, past_key_value=past_key_value)
                    y_true = _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                    y_true = self.o_proj(y_true)

                else:
                    _y_true, attn_true = self.softmax_attention(q_shadow, k_shadow, v_shadow, causal=True, fp32_attention=False, stage1=self.stage1, stage2=self.stage2, position_ids=position_ids, past_key_value=past_key_value)
                    y_true = _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                    y_true = self.o_proj(y_true)


            u = hidden_states
            batch, seqlen, dim = u.shape

            v = rearrange(v, "b h l n -> b l h n", h=self.num_key_value_heads)
            q = rearrange(q, "b h l n -> b l h n", h=self.num_heads)
            k = rearrange(k, "b h l n -> b l h n", h=self.num_key_value_heads)

            if self.use_conv:
                v_flattened = rearrange(v, "b l h n -> b l (h n)")
                k_flattened = rearrange(k, "b l h n -> b l (h n)")
                q_flattened = rearrange(q, "b l h n -> b l (h n)")

                # 2. 在维度 2 上拼接三个张量，得到 (b, l, 3*(h*d))
                vkq = torch.cat([v_flattened, k_flattened, q_flattened], dim=2)

                vkq = self.convolutional_forward(vkq)

                v, k, q = torch.split(
                    vkq,
                    [
                        self.num_key_value_heads * self.head_dim,
                        self.num_key_value_heads * self.head_dim,
                        self.num_heads * self.head_dim,
                    ],
                    dim=-1,
                )

                v = rearrange(v, "b l (h n) -> b h l n", h=self.num_key_value_heads)
                k = rearrange(k, "b l (h n) -> b h l n", h=self.num_key_value_heads)
                q = rearrange(q, "b l (h n) -> b l h n", h=self.num_heads)
                k = repeat_kv(k, self.num_key_value_groups).transpose(1, 2)
                v = repeat_kv(v, self.num_key_value_groups).transpose(1, 2)


            else:
                # import ipdb; ipdb.set_trace()
                # q, k ,v = self.act(q), self.act(k), self.act(v)
                k = repeat_kv(k.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)
                v = repeat_kv(v.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)
            # 2. Compute "predicted" attention (just weights)

            #下面是mamba2的forward---------------------------------------------------------------------------------------

            if self.use_qknorm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            dt = self.in_proj(u)

            if self.use_A:
                A = -torch.exp(self.A_log_bias.float())
            else:
                A = -torch.ones(self.num_heads, device=dt.device)
            y = mamba_chunk_scan_combined(
                x =v,
                #x=v / F.softplus(A_log).to(v.dtype).unsqueeze(-1), 
                dt=dt,
                dt_softplus=True,
                A=A,
                #A=-torch.ones(self.num_heads, device=dt.device),
                B=k,
                C=q,
                chunk_size=self.chunk_size,
                dt_bias=self.dt_bias,
                # initial_states=(state["ssm"] if state is not None else None), # currently not supported by mamba_ssm.utils.generation
                return_final_states=False,
            )

            if self.use_D:
                Du = torch.einsum("h,blhp->blhp", self.D, v)
                y = y + Du
            if self.use_gnorm:
                y_pred = self.g_norm(y)
                g = self.g_proj(hidden_states)
                g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
                y_pred = y_pred * self.gate_fn(g)  
                
            else:
                y_pred = y
            y_pred = rearrange(y_pred, 'b l h d -> b h l d')

            attn_pred = None

        else:  # Stage3
            u = hidden_states
            batch, seqlen, dim = u.shape
            v = rearrange(v, "b h l n -> b l h n", h=self.num_key_value_heads)
            q = rearrange(q, "b h l n -> b l h n", h=self.num_heads)
            k = rearrange(k, "b h l n -> b l h n", h=self.num_key_value_heads)

            if self.use_conv:
                v_flattened = rearrange(v, "b l h n -> b l (h n)")
                k_flattened = rearrange(k, "b l h n -> b l (h n)")
                q_flattened = rearrange(q, "b l h n -> b l (h n)")

                # 2. 在维度 2 上拼接三个张量，得到 (b, l, 3*(h*d))
                vkq = torch.cat([v_flattened, k_flattened, q_flattened], dim=2)

                vkq = self.convolutional_forward(vkq)

                v, k, q = torch.split(
                    vkq,
                    [
                        self.num_key_value_heads * self.head_dim,
                        self.num_key_value_heads * self.head_dim,
                        self.num_heads * self.head_dim,
                    ],
                    dim=-1,
                )

                v = rearrange(v, "b l (h n) -> b h l n", h=self.num_key_value_heads)
                k = rearrange(k, "b l (h n) -> b h l n", h=self.num_key_value_heads)
                q = rearrange(q, "b l (h n) -> b l h n", h=self.num_heads)
                k = repeat_kv(k, self.num_key_value_groups).transpose(1, 2)
                v = repeat_kv(v, self.num_key_value_groups).transpose(1, 2)
            else:
                q, k ,v = self.act(q), self.act(k), self.act(v)

            if self.use_qknorm:
                q = self.q_norm(q)
                k = self.k_norm(k)
                
            dt = self.in_proj(u)
            if self.use_A:
                A = -torch.exp(self.A_log_bias.float())
            else:
                A = -torch.ones(self.num_heads, device=dt.device)
            y = mamba_chunk_scan_combined(
                x = v,
                #x = v / F.softplus(A_log).to(v.dtype).unsqueeze(-1),
                dt=dt,
                dt_softplus=True,
                A=A,
                B=k,
                C=q,
                chunk_size=self.chunk_size,
                dt_bias=self.dt_bias,
                # initial_states=(state["ssm"] if state is not None else None), # currently not supported by mamba_ssm.utils.generation
                return_final_states=False,
            )

            if self.use_D:
                Du = torch.einsum("h,blhp->blhp", self.D, v)
                y = y + Du
            if self.use_gnorm:
                y_true = self.g_norm(y)
                g = self.g_proj(hidden_states)
                g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
                y_true = y_true * self.gate_fn(g) 
            else:
                y_true = y
            y_true = rearrange(y_true, 'b l h d -> b h l d')

            attn_pred=None
            attn_true=None
            y_pred=None
            _y_true=None

            # Concatenate heads and apply output projection
            y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            y_true = self.o_proj(y_true)

        return y_true, (attn_pred, attn_true), (y_pred, _y_true), past_key_value
    
    def convolutional_forward(self, xBC):
        xBC = causal_conv1d_fn(
            xBC.transpose(1, 2),
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            activation=None if self.activation == "identity" else self.activation,
        ).transpose(1, 2)
        return xBC
    
    def create_shadowweight(self):
        self.q_proj_shadow = nn.Linear(self.hidden_size, self.num_q_heads, bias=False, dtype=self.dtype, device=self.device)
        self.k_proj_shadow = nn.Linear(self.hidden_size, self.num_kv_heads, bias=False, dtype=self.dtype, device=self.device)
        self.v_proj_shadow = nn.Linear(self.hidden_size, self.num_kv_heads, bias=False, dtype=self.dtype, device=self.device)
        # self.q_proj_shadow.weight.data = self.q_proj.weight.clone().detach()
        # self.k_proj_shadow.weight.data = self.k_proj.weight.clone().detach()
        # self.v_proj_shadow.weight.data = self.v_proj.weight.clone().detach()
        self.q_proj_shadow.weight.data = self.q_weight.detach()
        self.k_proj_shadow.weight.data = self.k_weight.detach()
        self.v_proj_shadow.weight.data = self.v_weight.detach()
        self.q_proj_shadow.weight.requires_grad = False
        self.k_proj_shadow.weight.requires_grad = False
        self.v_proj_shadow.weight.requires_grad = False

        self.q_proj.weight.requires_grad = True
        self.k_proj.weight.requires_grad = True
        self.v_proj.weight.requires_grad = True

        self.train_stage="2"

    def change_weight(self, norm_eps: float = 1e-5,):
        self.wvkqgdt = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads + self.num_heads) * self.head_dim + self.num_heads,
            bias=self.bias,
            dtype=self.dtype,
        )
        self.g_norm_swish_gate = FusedRMSNormSwishGate(hidden_size=self.head_dim, elementwise_affine=True, eps=norm_eps).to(self.dtype).to(self.device)

        with torch.no_grad():
            # 获取原始卷积层的权重和偏置
            q_weight = self.q_proj.weight
            k_weight = self.k_proj.weight
            v_weight = self.v_proj.weight
            g_weight = self.g_proj.weight
            dt_weight = self.in_proj.weight
            g_norm_weight = self.g_norm.weight
            
            self.wvkqgdt.weight.copy_(torch.cat([v_weight, k_weight, q_weight, g_weight, dt_weight], dim=0))

            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.g_proj = None
            self.in_proj = None

            self.g_norm_swish_gate.weight.copy_(g_norm_weight)
            self.g_norm = None

        


class LinearAttentionState(Cache):
    """
    Handle the KV and K states for linear attention
    - Adopts HF Transformers `past_key_values` convention
    - Inherits from `Cache` class
    - Modified from transformers.cache_utils.DynamicCache (v4.36)
    """
    def __init__(self) -> None:
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []
        self.kv_states: List[torch.Tensor] = []
        self.k_states:  List[torch.Tensor] = []

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states. A layer index can be optionally passed.
        """
        if len(self._seen_tokens_by_layer) <= layer_idx:  # Initializing kv and k states
            self._seen_tokens_by_layer.append(0)
        return self._seen_tokens_by_layer[layer_idx]

    def get_max_length(self) -> Optional[int]:
        """
        Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length.
        """
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: Optional[int] = None, cache_kwargs: Optional[any] = None,
               accumulate_in_fp32: bool = True, **kwargs: any,
              ) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad ():
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
            dtype = key_states.dtype
            if accumulate_in_fp32:
                key_states, value_states = key_states.float(), value_states.float()

            kv_state = torch.einsum('bhlf,bhld->bhfd', key_states, value_states).detach()
            k_state  = key_states.sum(dim=-2, keepdim=True).detach()  # b, h, 1, f; note the 1
            # Update the cache
            if len(self.k_states) <= layer_idx:  # Initializing kv and k states
                print('if len(self.k_states) <= layer_idx:  # Initializing kv and k states')
                self.kv_states.append(kv_state.to(dtype))
                self.k_states.append(k_state.to(dtype))
            else:
                kv_state = (self.kv_states[layer_idx].to(kv_state.dtype) + kv_state).to(dtype)
                k_state  = (self.k_states[layer_idx].to(kv_state.dtype) + k_state).to(dtype)
                self.kv_states[layer_idx] = kv_state
                self.k_states[layer_idx]  = k_state
            self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2] 
        return self.kv_states[layer_idx], self.k_states[layer_idx]

    def to_legacy_cache(self):
        """Hack, but just return self"""
        return self

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """
        Reorders the cache for beam search, given the selected beam indices.
        -> Copied from transformers/src/transformers/cache_utils.py
        """
        raise NotImplementedError('Reordering cache not implemented for LinearAttentionState')