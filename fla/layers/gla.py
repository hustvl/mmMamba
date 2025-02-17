# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.models.mistral.configuration_mistral import MistralConfig
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.activations import ACT2FN
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
import torch.nn.init as init
import math
import copy

if TYPE_CHECKING:
    from fla.models.utils import Cache


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
# TODO @Arthur no longer copied from LLama after static cache
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class FeatureMapMLP(nn.Module):
    """
    Learnable MLP in feature map.
    
    Full feature map is like f(xW + b)
    -> This is the `W` and (optional) `b` part
    """
    def __init__(self, 
                 num_heads: int,
                 head_dim: int,     # input dim
                 feature_dim: int,  # output dim
                 dtype: torch.dtype,
                 device: torch.device,
                 skip_connection: bool = False,
                 bias: bool = False,
                 zero_init: bool = False,
                 normal_init: bool = False,):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.device = device
        self.skip_connection = skip_connection
        self.bias = bias
        self.zero_init = zero_init
        self.normal_init = normal_init
        self.init_weights_()

        """if self.zero_init:  # Zero-out weights or set as identity post-initialization
            self.zero_init_with_skip_() if self.skip_connection else self.zero_init_()"""

        if self.normal_init:
            with torch.no_grad():
                nn.init.normal_(self.layer)
        
        if self.skip_connection:
            assertion_fail = f'If self.skip_connection we need self.head_dim == self.feature_dim but self.head_dim is {self.head_dim} != self.feature_dim is {self.feature_dim}'
            assert self.head_dim == self.feature_dim, assertion_fail

    def init_weights_(self):
        """
        Initialize (W)eights and (b)iases
        """
        layer = torch.empty(
            (self.num_heads, self.head_dim, self.feature_dim),
            dtype=self.dtype, device=self.device,
        )
        nn.init.kaiming_uniform_(layer)
        self.layer = nn.Parameter(layer)

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(
                (1, self.num_heads, 1, 1),  # self.feature_dim),
                dtype=self.dtype, device=self.device,
            ))
            nn.init.kaiming_uniform_(self.bias)
        else:
            self.bias = 0.  # hack

    def zero_init_with_skip_(self):
        """
        Initialize weights to zero matrix if skip connection
        """
        with torch.no_grad():
            nn.init.zeros_(self.layer)

    def zero_init_(self):
        """
        Initialize weights to identity matrix if no skip connection
        """
        with torch.no_grad():
            for i in range(self.layer.shape[0]):
                try:
                    nn.init.eye_(self.layer[i])
                except RuntimeError:
                    with torch.no_grad():
                        dtype = self.layer[i].dtype
                        weight = torch.eye(*self.layer[i].shape,
                                           requires_grad=self.layer[i].requires_grad,
                                           device=self.layer[i].device)
                        self.layer[i] = weight.to(dtype=dtype)

    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, num_heads, seq_len, head_dim)
        """
        _x = torch.einsum('hdf,bhld->bhlf', self.layer, x) + self.bias
        return x + _x if self.skip_connection else _x
    
# -----------------------
# Feature map activations
# -----------------------
class FeatureMapAct(nn.Module):
    """
    Base class for feature map activations
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, *args: any, **kwargs: any):
        """
        x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return x
    
class SoftmaxDim(FeatureMapAct):
    """
    Softmax activation as in https://arxiv.org/abs/2402.04347
    """
    def forward(self, x: torch.Tensor, *args: any, **kwargs: any):
        return torch.cat([
            torch.softmax(x, dim=-1), torch.softmax(-x, dim=-1)
        ], dim=-1).clamp(min=self.eps)
    
class FeatureMap(nn.Module):
    """
    Final 'activation' of feature map. Can probably be combined with
    `FeatureMapMLP` below

    Full feature map is like f(xW + b)
    -> This is the `f` part
    """
    def __init__(self,
                 head_dim_idx: int = -1, 
                 eps: float = 1e-12, 
                 mlp: nn.Module = None,
                 fullspace: bool = True,):
        super().__init__()
        self.head_dim_idx = head_dim_idx     
        self.eps = eps
        self.mlp = mlp if mlp is not None else nn.Identity()
        self.activation = SoftmaxDim()
        
    def forward(self, x: torch.Tensor, *mlp_args: any, **mlp_kwargs: any):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return self.activation(self.mlp(x, *mlp_args, **mlp_kwargs), x)

    def q_map(self, *args: any, **kwargs: any):
        """
        Use for inference in case q and k feature maps differ
        """
        return self.forward(*args, **kwargs)

    def k_map(self, *args: any, **kwargs: any):
        """
        Use for inference in case q and k feature maps differ
        """
        return self.forward(*args, **kwargs)
    



    


class GatedLinearAttention(nn.Module):
    r"""
    The layer implementaion for [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635).  # noqa

    Args:
        mode (str, Optional):
            Which GLA kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 0.5.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_kv_heads (int, Optional):
            The number of key/value heads, used for MQA. Default: None.
        feature_map (str, Optional):
            Feature map function applied to queries/keys. Default: None.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `False`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        gate_fn (str, Optional):
            The activation function for the output gate. Default: `swish`.
        elementwise_affine (bool, Optional):
            If `True`, applies elementwise affine to LayerNorm with learnable parameters. Default: `True`.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        gate_logit_normalizer (int, Optional):
            The normalizer for the gate logits, appied after `logsigmoid`. Default: 16.
        gate_low_rank_dim (int, Optional):
            The low rank dim for the gate projection. Default: 16.
        clamp_min (float, Optional):
            The minimum value for the gate logits. Default: None.
        fuse_norm (bool, Optional):
            Whether to fuse the norm and the output gate for better memory footprint. Default: `True`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
    """

    def __init__(
        self,
        config: MistralConfig,
        mode: str = 'fused_recurrent',
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        learned_kernel_kwargs: Optional[dict] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: int = None,
    ) -> GatedLinearAttention:
        super().__init__()

        self.training=True

        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        num_heads=config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mode = mode
        self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None

        
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate

        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_heads * self.head_dim
        self.key_dim_per_group = self.num_key_value_heads * self.head_dim
        self.value_dim_per_group = self.num_key_value_heads * self.head_dim
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        #self.gk_proj = nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        self.gk_proj = nn.Sequential(nn.Linear(self.hidden_size, 16, bias=False),
                                     nn.Linear(16, self.key_dim_per_group, bias=True))
        #nn.init.xavier_uniform_(self.gk_proj[0].weight, gain=2 ** -2.5)
        #nn.init.xavier_uniform_(self.gk_proj[1].weight, gain=2 ** -2.5)#
        self.gk_proj[0].weight.data.normal_(mean=0.0, std=0.05)
        self.gk_proj[1].weight.data.normal_(mean=0.0, std=0.05)
        nn.init.zeros_(self.gk_proj[1].bias)
        

        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        self.gate_fn = ACT2FN[gate_fn]
        
        self.return_matrix=False

        self.gate_logit_normalizer = gate_logit_normalizer

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        #learned_kernel_kwargs = {k: v for k, v in learned_kernel_kwargs.items()}
        """learned_kernel_kwargs['num_heads'] = self.num_heads
        learned_kernel_kwargs['head_dim']  = self.head_dim
        learned_kernel_kwargs['dtype']     = self.q_proj.weight.dtype
        learned_kernel_kwargs['device']    = self.q_proj.weight.device"""

        mlp=FeatureMapMLP(num_heads=num_heads, head_dim=self.head_dim, dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device, feature_dim=int(self.head_dim/2))
        self.feature_map_q = FeatureMap(mlp=mlp)
        self.feature_map_k = copy.deepcopy(self.feature_map_q)

        """for layer in self.gk_proj.children():
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)"""

        #self.apply(self._initialize_weights)

        #print('yes')


    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_chunk'
        self.use_output_gate=None
        output_attentions=False

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if self.use_short_conv:
            conv_state_q = last_state[0] if use_cache else None
            conv_state_k = last_state[1] if use_cache else None
            conv_state_v = last_state[2] if use_cache else None
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q = self.q_conv1d(q, attention_mask, conv_state_q)
            k = self.k_conv1d(k, attention_mask, conv_state_k)
            v = self.v_conv1d(v, attention_mask, conv_state_v)            
        else:
            hidden_states = hidden_states.to(self.q_proj.weight.dtype)
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)

        kv_seq_len = k.shape[-2]
        

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        if self.num_key_value_heads > 1:
            k, v, gk = (repeat(x, 'b l (h d) -> b (h g) l d', h=self.num_key_value_heads, g=self.num_key_value_groups) for x in (k, v, gk))
        else:
            k, v, gk = (rearrange(x, 'b l (h d) -> b h l d', h=self.num_key_value_heads) for x in (k, v, gk))

        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        q=self.feature_map_q.q_map(q)
        k=self.feature_map_q.k_map(k)

        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        recurrent_state = last_state[-1] if use_cache else None
        #o1, recurrent_state = fused_recurrent_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        #o2, recurrent_state = fused_chunk_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(q, k, v, gk, initial_state=recurrent_state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        if output_attentions:
            scale = q.shape[-1] ** -0.5
            
            g = torch.cumsum(gk, dim=2)

            attn_weights=torch.matmul((q*torch.exp(g)*scale), (k*torch.exp(-g)).transpose(2,3))

            """q_g, k_g=ComputeAttnParallelFunction.apply(q, k, gk)

            #contains_nan1 = torch.any(torch.isnan(q_g))
            #contains_nan2 = torch.any(torch.isnan(k_g))

            q_g = q_g.to(torch.float64)
            k_g = k_g.to(torch.float64)

            attn_weights = torch.matmul(q_g, k_g.transpose(2, 3))"""

            seq_len=q.shape[2]

            causal_mask_np = np.tril(np.ones((seq_len, seq_len)))

            causal_mask = torch.from_numpy(causal_mask_np).to(attn_weights.device)

            masked_attention = torch.where(causal_mask == 1, attn_weights, torch.tensor(0.0, device=attn_weights.device))

            """o = torch.matmul(masked_attention, v)

            recurrent_state=None"""
            
            masked_attention = torch.sum(masked_attention, dim=1)

        else:masked_attention=None

        if past_key_values is not None:
            if self.use_short_conv:
                last_state = (conv_state_q, conv_state_k, conv_state_v, recurrent_state)
            else:
                last_state = (recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, q.shape[2])

        o = rearrange(o, 'b h l d -> b l h d')
        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b l h d -> b l (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b l h d -> b l (h d)')

        o = self.o_proj(o)

        return o, masked_attention, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            state += (param.new_zeros(batch_size, self.key_dim, self.conv_size),
                        param.new_zeros(batch_size, self.key_dim, self.conv_size),
                        param.new_zeros(batch_size, self.value_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size
    
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
from packaging import version
from fla.utils import contiguous
from torch.cuda.amp import custom_bwd, custom_fwd


inv_ln2 = 1.44269504

@triton.jit
def fwd_decay_cumsum(
    g,
    g_o, 
    s_qk_h,
    s_qk_t,
    s_qk_d,
    B,
    H,
    T,
    BK: tl.constexpr,
    DK: tl.constexpr
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)#为什么不使用s_qk_t、s_qk_d
    p_go = g_o + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK

    for i in range(T):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        cum_decay += _g * inv_ln2
        tl.store(p_go, cum_decay.to(p_go.dtype.element_ty), mask=mask)
        p_g += DK
        p_go += DK

@triton.jit
def prepare_qg_kg_bwd(
    dq,
    dk,
    q,
    k,
    g,
    dg,
    dq_g,
    dk_g,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    B,
    H,
    T,
    scale,
    BK: tl.constexpr,
    DK: tl.constexpr
):

    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)+(T - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)+(T - 1) * DK
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)+(T - 1) * DK

    p_dq = dq + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)+(T - 1) * DK
    p_dg = dg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)+(T - 1) * DK
    p_dk = dk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)+(T - 1) * DK

    p_dq_g = dq_g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)+(T - 1) * DK
    p_dk_g = dk_g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)+(T - 1) * DK
    
    mask = (i_k * BK + tl.arange(0, BK)) < DK

    last_decay = tl.load(g + i_bh * s_qk_h + (T - 1) * DK + i_k * BK + tl.arange(0, BK))
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)

    for i in range(T-1, -1, -1):
        _dq = tl.load(p_dq_g, mask=mask, other=0)
        _dk = tl.load(p_dk_g, mask=mask, other=0)
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        _dq *= tl.math.exp2(_g) * scale
        _dk *= tl.math.exp2( - _g)
        _dg=_dq * _q - _dk * _k
        cum_grad_dg += _dg
        tl.store(p_dk, _dk.to(p_dk.dtype.element_ty), mask=mask)
        tl.store(p_dq, _dq.to(p_dq.dtype.element_ty), mask=mask)
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_dq -= DK
        p_dg -= DK
        p_dk -= DK
        p_q -= DK
        p_g -= DK
        p_k -= DK
        p_dk_g -= DK
        p_dq_g -= DK

@triton.jit
def prepare_qg_kg(
    q,
    k,
    g,
    qg,
    kg,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    B,
    H,
    T,
    scale,
    BK: tl.constexpr,
    DK: tl.constexpr
):

    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    
    mask = (i_k * BK + tl.arange(0, BK)) < DK

    last_decay = tl.load(g + i_bh * s_qk_h + (T - 1) * DK + i_k * BK + tl.arange(0, BK))

    for i in range(T):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        _q *= tl.math.exp2(_g) * scale
        _k *= tl.math.exp2( - _g)
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        p_q += DK
        p_g += DK
        p_k += DK
        p_kg += DK
        p_qg += DK

class ComputeAttnParallelFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, g):
        ctx.g_dtype = g.dtype
        g_original = g
        # cumulative decay should be in float32, otherwise the err will be accumulated and amplified.
        g = torch.empty_like(g, dtype=torch.float32)
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        scale = q.shape[-1] ** -0.5
        ctx.scale = scale

        BK= min(d_head_qk, 64)

        NK= triton.cdiv(d_head_qk, BK)
        q_g = torch.empty_like(q)
        k_g = torch.empty_like(k)
        grid = (NK, batch_size * n_heads)
        fwd_decay_cumsum[grid](#计算对数化的B矩阵
                g_original,
                g,
                q.stride(1), q.stride(2), q.stride(3),
                batch_size, n_heads, seq_len,
                BK=BK, DK=d_head_qk, num_warps=1
            )
        prepare_qg_kg[grid](
                q, k, g, q_g, k_g,
                q.stride(1), q.stride(2), q.stride(3),
                batch_size, n_heads, seq_len, scale,
                BK=BK, DK=d_head_qk, num_warps=1
            )

        CHECK = True
        if version.parse(triton.__version__) < version.parse('2.2.0'):
            import warnings
            warnings.warn(
                "Triton<2.2.0 detected for running this kernel, "
                "which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) "
                "that lead to significant precision loss. "
                "We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. "
                "For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
            )
            CHECK = True

        ctx.save_for_backward(q, k, g_original)
        ctx.CHECK = CHECK
        return q_g, k_g
    


    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, dq_g, dk_g):
        q, k, g_original= ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        scale = ctx.scale

        # recomputation
        # inter-chunk
        g = torch.empty_like(g_original, dtype=torch.float32)
        BK= min(d_head_qk, 64)
        NK= triton.cdiv(d_head_qk, BK)
        dq = torch.empty_like(dq_g)
        dk = torch.empty_like(dk_g)
        dg = torch.empty_like(g_original, dtype=torch.float32)
        grid = (NK, batch_size * n_heads)
        fwd_decay_cumsum[grid](#计算对数化的B矩阵
            g_original,
            g,
            q.stride(1), q.stride(2), q.stride(3),
            batch_size, n_heads, seq_len,
            BK=BK, DK=d_head_qk, num_warps=1
        )
        prepare_qg_kg_bwd[grid](
                dq, dk, q, k, g, dg, dq_g, dk_g,
                q.stride(1), q.stride(2), q.stride(3),
                batch_size, n_heads, seq_len, scale,
                BK=BK, DK=d_head_qk, num_warps=1
        )

        return dq.to(q), dk.to(k), dg.to(ctx.g_dtype)