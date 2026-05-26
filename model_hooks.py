from collections import OrderedDict
from dataclasses import dataclass, field
import types

import torch
import einops
from einops import rearrange
import torch.nn.functional as F

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from graph_registry import (
    get_attn_weights_cache, 
    clear_attn_weights_cache, 
    get_attn_weights_cache_info
)
import transformer_lens
from transformer_lens import HookedTransformer
import bitsandbytes.functional as bnbF

@dataclass
class Outputs:
    inputs_embeds: torch.FloatTensor = None
    position_embeds: torch.FloatTensor = None
    attn_outputs: tuple = ()
    values: tuple = ()
    attn_outs: tuple = ()
    head_inputs: tuple = ()
    head_outputs: tuple = ()
    mlp_pre_acts: tuple = ()
    mlp_gates: tuple = ()
    avg_mlp_gates: dict = field(default_factory=dict)
    mlp_outputs: tuple = ()
    ln_states: tuple = ()
    hidden_states: tuple = ()
    attentions: tuple = ()
    logits: torch.FloatTensor = None
    labels: torch.LongTensor = None
    loss: torch.FloatTensor = None
    attn_attr: OrderedDict = field(default_factory=OrderedDict)


def set_hooks(model):
    for block in model.model.layers:
        block.input_layernorm.variance = None
        block.self_attn.attn_out = None
        block.post_attention_layernorm.variance = None
        block.mlp_output = None
    model.model.norm.variance = None


def del_hooks(model):
    for block in model.model.layers:
        if hasattr(block.input_layernorm, 'variance'): delattr(block.input_layernorm, 'variance')
        if hasattr(block.self_attn, 'attn_out'): delattr(block.self_attn, 'attn_out')
        if hasattr(block.post_attention_layernorm, 'variance'): delattr(block.post_attention_layernorm, 'variance')
        if hasattr(block, 'mlp_output'): delattr(block, 'mlp_output')
    if hasattr(model.model.norm, 'variance'): delattr(model.model.norm, 'variance')


def _to_cpu_pinned(tensor):
    """Move tensor to CPU with pinned memory for faster GPU transfers."""
    return tensor.to('cpu').pin_memory()


def get_outputs(model, output, device='cpu', pin_memory=True):
    """
    Collect and offload model outputs to CPU.
    
    Args:
        pin_memory: If True and device='cpu', use pinned memory for ~2x faster 
                   GPU transfers when reloading (32ms vs 67ms for 486MB).
    """
    attn_outs, mlp_outputs, ln_states = [], [], []
    to_device = _to_cpu_pinned if (device == 'cpu' and pin_memory) else lambda t: t.to(device)
    
    for block in model.model.layers:
        ln_states.append((to_device(block.input_layernorm.variance),
                          to_device(block.post_attention_layernorm.variance)))
        delattr(block.input_layernorm, 'variance')
        delattr(block.post_attention_layernorm, 'variance')
        attn_outs.append(to_device(block.self_attn.attn_out))
        delattr(block.self_attn, 'attn_out')
        mlp_outputs.append(to_device(block.mlp_output))
        delattr(block, 'mlp_output')
    ln_states.append(to_device(model.model.norm.variance))
    delattr(model.model.norm, 'variance')
    hidden_states = [to_device(hs) for hs in output.hidden_states]
    return Outputs(hidden_states=hidden_states, ln_states=ln_states, attn_outs=attn_outs, mlp_outputs=mlp_outputs)

def get_outputs_from_cache(model, cache, device='cpu', pin_memory=True):
    if type(model) == HookedTransformer:
        install_qwen3_like_tl_attn(model)
    attn_outs, mlp_outputs, ln_states, hidden_states = [], [], [], []
    to_device = _to_cpu_pinned if (device == 'cpu' and pin_memory) else lambda t: t.to(device)
    
    for block in range(model.cfg.n_layers):
        ln_states.append((to_device(cache[f"blocks.{block}.ln1.hook_scale"]),
                          to_device(cache[f"blocks.{block}.ln2.hook_scale"])))
        attn_outs.append(to_device(cache[f"blocks.{block}.attn.hook_z"]))
        mlp_outputs.append(to_device(cache[f"blocks.{block}.hook_mlp_out"]))
        hidden_states.append(to_device(cache[f"blocks.{block}.hook_resid_pre"]))
    hidden_states.append(to_device(cache["ln_final.hook_normalized"]))
    ln_states.append(to_device(cache[f"ln_final.hook_scale"]))
    return Outputs(hidden_states=hidden_states, ln_states=ln_states, attn_outs=attn_outs, mlp_outputs=mlp_outputs)

def to_gpu(tensor, device, pos_ids=None):
    """Load tensor from CPU (pinned) memory to GPU efficiently."""
    if pos_ids is not None: tensor = tensor[:, pos_ids]
    return tensor.to(device, non_blocking=True)


def is_remote(model):
    """Check if model is a remote client (has compute_node_gradient method)."""
    return hasattr(model, 'compute_node_gradient')


def _qwen3_like_tl_attn_forward(
    self,
    hidden_states=None,
    position_embeddings=None,
    attention_mask=None,
    trim=True,
    no_sink=False,
    q_hidden_states=None,
    k_hidden_states=None,
    v_hidden_states=None,
    attn_weights=None,
    head_ids=None,
    past_key_values=None,
    cache_position=None,
    output_attentions=False,
    position_ids=None,
    q_position_ids=None,
    **kwargs,
):
    # Keep compatibility with TL legacy callsites (query_input/key_input/value_input).
    query_input = kwargs.pop("query_input", q_hidden_states if q_hidden_states is not None else hidden_states)
    key_input = kwargs.pop("key_input", k_hidden_states if k_hidden_states is not None else hidden_states)
    value_input = kwargs.pop("value_input", v_hidden_states if v_hidden_states is not None else hidden_states)
    past_kv_cache_entry = kwargs.pop("past_kv_cache_entry", None)
    if past_key_values is None:
        past_key_values = past_kv_cache_entry
    additive_attention_mask = kwargs.pop("additive_attention_mask", None)
    position_bias = kwargs.pop("position_bias", None)
    output_attentions = output_attentions or kwargs.pop("outputs_attention", False)
    _ = (position_embeddings, cache_position, position_ids, q_position_ids)

    q, k, v = self.calculate_qkv_matrices(query_input, key_input, value_input)

    if past_key_values is not None:
        kv_cache_pos_offset = past_key_values.past_keys.size(1)
        k, v = past_key_values.append(k, v)
    else:
        kv_cache_pos_offset = 0

    if self.cfg.positional_embedding_type == "rotary":
        # TL apply_rotary expects a 2D token mask (or None), not a 4D float causal mask.
        rotary_mask = None
        if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.dtype in (
            torch.bool, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            rotary_mask = attention_mask
        q = self.hook_rot_q(self.apply_rotary(q, kv_cache_pos_offset, rotary_mask))
        k = self.hook_rot_k(self.apply_rotary(k, 0, rotary_mask))

    if self.cfg.dtype not in [torch.float32, torch.float64]:
        q = q.to(torch.float32)
        k = k.to(torch.float32)

    if attn_weights is None:
        attn_scores = self.calculate_attention_scores(q, k)
        if self.cfg.positional_embedding_type == "alibi":
            query_ctx = attn_scores.size(-2)
            key_ctx = attn_scores.size(-1)
            if self.alibi is None or key_ctx > self.alibi.size(-1):
                self.alibi = self.create_alibi_bias(self.cfg.n_heads, key_ctx, self.cfg.device)
            attn_scores += self.alibi[:, -query_ctx:, :key_ctx]
        elif self.cfg.positional_embedding_type == "relative_positional_bias":
            if position_bias is None:
                if self.has_relative_attention_bias:
                    raise ValueError("Positional bias is required for relative_positional_bias")
                position_bias = torch.zeros(
                    1, self.cfg.n_heads, attn_scores.shape[2], attn_scores.shape[3], device=attn_scores.device
                )
            attn_scores += position_bias
        # HF/Qwen path often provides a 4D additive mask [b, 1, q, k].
        # TL apply_causal_mask expects 2D token masks and enforces q+offset==k.
        if attention_mask is not None and attention_mask.dim() == 4:
            attn_scores = attn_scores + attention_mask.to(attn_scores.dtype)
        elif self.cfg.attention_dir == "causal":
            attn_scores = self.apply_causal_mask(attn_scores, kv_cache_pos_offset, attention_mask)
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask
        if no_sink:
            attn_scores[:, :, :, 0] = -1e4
        attn_scores = self.hook_attn_scores(attn_scores)
        pattern = self.hook_pattern(F.softmax(attn_scores, dim=-1))
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
    else:
        pattern = attn_weights

    pattern = pattern.to(self.cfg.dtype).to(v.device)
    # Keep head dimension for downstream TL calculate_z_scores(v, pattern).
    # Upstream indexing like pattern[:, head] may collapse it to [b, q, k].
    if pattern.dim() == 3:
        pattern = pattern.unsqueeze(1)  # [b, 1, q, k]
    z = self.calculate_z_scores(v, pattern)  # [batch, pos, head_index, d_head]

    if head_ids is not None:
        if not isinstance(head_ids, (list, tuple)):
            head_ids = [head_ids]
        if trim and z.shape[2] == self.cfg.n_heads:
            z = z[:, :, head_ids, :]
        elif not trim and z.shape[2] == self.cfg.n_heads:
            mask = torch.zeros(self.cfg.n_heads, dtype=z.dtype, device=z.device)
            mask[head_ids] = 1.0
            z = torch.einsum("bphd,h->bphd", z, mask)
        if z.shape[2] != self.cfg.n_heads:
            full = torch.zeros(z.shape[0], z.shape[1], self.cfg.n_heads, z.shape[3], dtype=z.dtype, device=z.device)
            full[:, :, head_ids, :] = z
            z = full

    if not self.cfg.use_attn_result:
        w_o = _get_tl_w_o(self, z.dtype, z.device)
        out = torch.einsum("...f,fe->...e", z.reshape(z.shape[0], z.shape[1], -1), w_o)
        out = out + self.b_O.to(device=out.device, dtype=out.dtype)
    else:
        w_o = _get_tl_w_o(self, z.dtype, z.device)
        w_o = rearrange(w_o, "(h d) m -> h d m", h=self.cfg.n_heads)
        unhooked_result = torch.einsum("bphd,hdm->bphm", z, w_o)
        result = self.hook_result(unhooked_result)
        out = einops.reduce(result, "b p h m -> b p m", "sum")
        out = out + self.b_O.to(device=out.device, dtype=out.dtype)

    if output_attentions:
        return out, pattern
    return out


def install_qwen3_like_tl_attn(model):
    """Monkey-patch HookedTransformer attention forward with Qwen3-style interface."""
    if type(model) != HookedTransformer:
        return False
    if getattr(model, "_barc_qwen3_attn_patched", False):
        return False
    for block in model.blocks:
        attn = block.attn
        if not hasattr(attn, "_barc_original_forward"):
            attn._barc_original_forward = attn.forward
        attn.forward = types.MethodType(_qwen3_like_tl_attn_forward, attn)
    model._barc_qwen3_attn_patched = True
    return True


def restore_tl_attn(model):
    """Restore original HookedTransformer attention forward methods."""
    if type(model) != HookedTransformer:
        return False
    restored = False
    for block in model.blocks:
        attn = block.attn
        if hasattr(attn, "_barc_original_forward"):
            attn.forward = attn._barc_original_forward
            delattr(attn, "_barc_original_forward")
            restored = True
    if hasattr(model, "_barc_qwen3_attn_patched"):
        delattr(model, "_barc_qwen3_attn_patched")
    return restored


def _get_tl_w_o(attn, dtype, device):
    """
    Get HookedTransformer attention output projection matrix in shape
    [(n_heads * d_head), d_model], handling 4-bit quantized weights.
    """
    if getattr(attn.cfg, "load_in_4bit", False):
        # In 4-bit mode W_O is Params4bit (packed uint8), dequantize first.
        w_o = bnbF.dequantize_4bit(attn.W_O.t(), attn.W_O.quant_state)
    else:
        w_o = rearrange(attn.W_O, "n d e -> (n d) e")
    return w_o.to(device=device, dtype=dtype)


# =============================================================================
# Unified get_xxx functions - work with local model or remote client
# Usage: get_head_output(model, r, layer, head, pos_ids)
# =============================================================================

def get_hidden_states(model, r, layer, pos_ids=None, **_):
    if is_remote(model):
        return model.get_hidden_states(sample_id=r.index, layer=layer, pos_ids=pos_ids)
    return to_gpu(r.outputs.hidden_states[layer], model.device, pos_ids)


def get_head_output(model, r, layer, head, pos_ids=None, **_):
    if is_remote(model):
        return model.get_head_output(sample_id=r.index, layer=layer, head=head, pos_ids=pos_ids)
    outputs = r.outputs
    if type(model) != HookedTransformer:
        if head == model.config.num_attention_heads:  # mlp
            return to_gpu(outputs.mlp_outputs[layer], model.device, pos_ids).to(model.dtype)
    else:
        if head == model.cfg.n_heads:  # mlp
            return to_gpu(outputs.mlp_outputs[layer], model.device, pos_ids).to(model.dtype)
    attn_out = to_gpu(outputs.attn_outs[layer], model.device, pos_ids).to(model.dtype)
    if head is not None:
        m = torch.zeros(attn_out.shape[2]).to(model.device, dtype=model.dtype)
        m[head] = 1.
        masked_attn_out = torch.einsum('bind,n->bind', attn_out, m)
        masked_attn_out = rearrange(masked_attn_out, 'b i n d -> b i (n d)')
    else:  # all heads
        if type(model) != HookedTransformer:
            mask = torch.eye(model.config.num_attention_heads).to(model.device, dtype=attn_out.dtype) # n*n
        else:
            mask = torch.eye(model.cfg.n_heads).to(model.device, dtype=attn_out.dtype) # n*n
        masked_attn_out = torch.einsum('bind,mn->bimnd', attn_out, mask)
        masked_attn_out = rearrange(masked_attn_out, 'b i m n d -> b i m (n d)')
    if type(model) != HookedTransformer:
        return model.model.layers[layer].self_attn.o_proj(masked_attn_out)  # bie for one head or bine for all heads
    else:
        attn = model.blocks[layer].attn
        w_o = _get_tl_w_o(attn, masked_attn_out.dtype, masked_attn_out.device)
        b_o = attn.b_O.to(device=masked_attn_out.device, dtype=masked_attn_out.dtype)
        return torch.einsum('...f,fe->...e', masked_attn_out, w_o) + b_o  # bie for one head or bine for all heads


def get_attn_output(model, r, layer, pos_ids=None, **_):
    if is_remote(model):
        return model.get_attn_output(sample_id=r.index, layer=layer, pos_ids=pos_ids)
    outputs = r.outputs
    attn_out = to_gpu(outputs.attn_outs[layer], model.device, pos_ids)
    attn_out = rearrange(attn_out, 'b i n d -> b i (n d)')
    if type(model) != HookedTransformer:
        return model.model.layers[layer].self_attn.o_proj(attn_out)
    attn = model.blocks[layer].attn
    w_o = _get_tl_w_o(attn, attn_out.dtype, attn_out.device)
    b_o = attn.b_O.to(device=attn_out.device, dtype=attn_out.dtype)
    return torch.einsum('...f,fe->...e', attn_out, w_o) + b_o


def get_attn_kwargs(batch_size, seq_length, device, dtype, pos_ids=None):
    attn_mask_converter = AttentionMaskConverter(is_causal=True)
    attention_mask = attn_mask_converter.to_causal_4d(
        batch_size, seq_length, seq_length, dtype=dtype, device=device)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    q_position_ids = None
    if pos_ids is not None:
        q_position_ids = position_ids[:, pos_ids]  # [bsz(=1), seq_len[pos_ids]]
        attention_mask = attention_mask[:, :, pos_ids, :]  # [bsz, 1, q_seq_len[pos_ids], kv_seq_len]
    return {'attention_mask': attention_mask, 'position_ids': position_ids, 'q_position_ids': q_position_ids}


def get_attn_weights(model, r, layer, head, pos_ids=None, use_cache=True, **_):
    if is_remote(model):
        return model.get_attn_weights(sample_id=r.index, layer=layer, head=head, pos_ids=pos_ids)
    
    outputs = r.outputs
    head_key = head if isinstance(head, (int, type(None))) else tuple(head)
    pos_key = tuple(pos_ids) if pos_ids is not None else None
    cache_key = (r.index, layer, head_key, pos_key)
    
    cache = get_attn_weights_cache()  # From graph_registry (survives autoreload)
    if use_cache and cache_key in cache:
        return cache[cache_key].to(model.device)  # TODO: remove to(), slice already in gpu
    
    if type(model) != HookedTransformer:
        self = model.model.layers[layer]
        hidden_states = to_gpu(outputs.hidden_states[layer], model.device)
        hidden_states = self.input_layernorm(hidden_states)
        if head is None: head = list(range(model.config.num_attention_heads))
    else:
        self = model.blocks[layer]
        hidden_states = to_gpu(outputs.hidden_states[layer], model.device)
        hidden_states = self.ln1(hidden_states)
        if head is None: head = list(range(model.cfg.n_heads))
    q_hidden_states = hidden_states[:, pos_ids] if pos_ids is not None else None

    kwargs = get_attn_kwargs(hidden_states.shape[0], hidden_states.shape[1], model.device, model.dtype, pos_ids=pos_ids)
    if type(model) != HookedTransformer:
        result = self.self_attn(hidden_states=hidden_states, q_hidden_states=q_hidden_states, 
                                output_attentions=True, position_embeddings=self.position_embeddings, 
                                **kwargs)[1][:, head]#.to('cpu') # TODO: remove comment
    else:
        result = self.attn(
            query_input=hidden_states,
            key_input=hidden_states,
            value_input=hidden_states,
            output_attentions=True,
        )[1][:, head]
    
    if use_cache and pos_ids is not None:  # only cache slice in gpu mem
        cache[cache_key] = result
    return result

