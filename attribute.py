from dataclasses import dataclass, field
from collections.abc import Iterable
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
import transformer_lens
from transformer_lens import HookedTransformer

import sys
sys.path.append('/home/hushengchun/project/BARC-transformerlens/pptree')
from pptree import Node as TNode, print_tree

from common_utils import einsum, mr, topk_md
from dequant import get_gptq_weight_cached
from model_hooks import *
from graph_registry import get_current_graph, set_current_graph


@dataclass
class AttrData:
    nodes: list = None  # downstream nodes
    attr: torch.FloatTensor = None
    attn_weights_ds: dict = field(default_factory=dict)
    attn_attrs_ds: dict = field(default_factory=dict)
    top_heads: OrderedDict = None  # upstream heads with attribution scores
    groups: dict = None
    metrics: dict = None


def scaled_input(input, num_points, baseline=None, requires_grad=True):
    assert input.size(0) == 1
    if baseline is None: baseline = torch.zeros_like(input)
    if num_points == 0:
        # Detach and clone to create a fresh leaf tensor for gradient computation (for GPTQ models)
        res = input.detach().clone()  # for linear attribution (e.g. attn_v) where IG is not needed
    else:
        step = (input - baseline) / num_points
        res = torch.cat([baseline + step * i for i in range(num_points + 1)], dim=0)
        # res = torch.cat([baseline + step * (i + 1) for i in range(num_points)], dim=0)  # XD
        # alphas = list(0.5 * (1 + np.polynomial.legendre.leggauss(num_points)[0])) # copied from captum
        # res = torch.cat([baseline + alpha * (input - baseline) for alpha in alphas], dim=0)
    if requires_grad: res.requires_grad_(True)
    return res #, step


def compute_loss(logits, labels, reduction=None, return_logit_diff=False, logits_mask=None):
    # if return_logit_diff:  
    #     b = (labels != -100).squeeze(0)  # 1i->i
    #     _logits, _labels = logits[:, b], labels[:, b]  # biv->bkv, 1i->1k
    #     _logits_mask = (logits_mask == 0)[:, b]  # biv->bkv
    #     _logits_mean = (_logits * _logits_mask).sum(-1) / _logits_mask.sum(-1)  # bk = bkv->bk / bkv->bk
    #     _logits = _logits - _logits_mean.unsqueeze(-1)  # bkv - bk->bk1 = bkv
    #     # loss as the NEGATIVE of logit diff
    #     assert reduction in [None, 'per_example_mean'], reduction
    #     loss = -_logits[:, torch.arange(_logits.size(1)), _labels[0]].sum(-1)  # bkv->bk->b, per_example_mean
    #     return loss

    if reduction is None: reduction = 'per_example_mean'
    # print('in compute_loss, labels =', labels)
    if reduction == 'argmax':
        labels = labels.clone()
        labels[labels != -100] = logits[-1:].argmax(-1)[labels != -100]
        reduction = 'per_example_mean'
    if labels.size(0) < logits.size(0): # logits has been scaled
        labels = einops.repeat(labels, '1 i -> b i', b=logits.size(0))
    loss_fct = nn.CrossEntropyLoss(reduction='none' if reduction == 'per_example_mean' else reduction)
    # print(f'logits.size = {logits.size()}, labels.size = {labels.size()}')
    loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    if reduction != 'mean':
        loss = loss.view(labels.size(0), -1) #4,16
        if reduction == 'per_example_mean':
            # loss = einops.reduce(loss, 'b i -> b', 'sum') / einops.reduce(labels != -100, 'b i -> b', 'sum')
            loss = torch.einsum('bi->b', loss) / torch.einsum('bi->b', labels != -100)
    return loss


@contextmanager
def use_dequant_projections(module, layer, grad_inputs):
    """
    Temporarily replace QuantLinear projections with Linear using dequantized weights,
    enabling gradient computation for GPTQ models.
    
    Args:
        module: Qwen2Attention or Qwen2MLP module (auto-detected)
        layer: Layer index (for cache key)
        grad_inputs: Set indicating which inputs need gradients
            - Attention: {'q', 'k', 'v'} → q_proj, k_proj, v_proj + o_proj
            - MLP: {'i', 'g'} → up_proj (i), gate_proj (g) + down_proj
    
    Example:
        with use_dequant_projections(attn, layer=0, grad_inputs={'v'}):
            output = attn(...)  # Autograd works through v_proj and o_proj
    """
    if not grad_inputs:
        yield
        return
    
    # Auto-detect module type and configure
    is_attn = hasattr(module, 'o_proj')
    if is_attn:
        proj_map = {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj'}
        output_proj = 'o_proj'
        cache_type = 'attn'
    else:
        proj_map = {'i': 'up_proj', 'g': 'gate_proj'}
        output_proj = 'down_proj'
        cache_type = 'mlp'

    # Only run this replacement path for GPTQ QuantLinear modules.
    # For bitsandbytes Linear4bit (and other backends), skip dequant swap.
    if not hasattr(getattr(module, output_proj), 'qweight'):
        yield
        return
    
    # Determine projections to replace: input proj(s) + output proj
    projs_to_replace = {proj_map[g] for g in grad_inputs if g in proj_map}
    projs_to_replace.add(output_proj)
    
    originals = {}
    try:
        for proj_name in projs_to_replace:
            orig = getattr(module, proj_name)
            originals[proj_name] = orig
            
            # Get dequantized weight (cached)
            dequant_w = get_gptq_weight_cached(orig, f'layer{layer}_{proj_name}', cache_type)
            
            # Create Linear with dequantized weight
            new_proj = nn.Linear(orig.infeatures, orig.outfeatures, 
                                bias=orig.bias is not None, device=dequant_w.device, dtype=dequant_w.dtype)
            new_proj.weight.data = dequant_w
            if orig.bias is not None:
                new_proj.bias.data = orig.bias.clone()     
            setattr(module, proj_name, new_proj)
        yield
    finally:
        for proj_name, orig in originals.items():
            setattr(module, proj_name, orig)


def head_forward(model, layer, head, xq=None, xk=None, xv=None, aw=None, attention_mask=None, 
                 position_ids=None, q_position_ids=None, no_sink=False, trim=True,
                 grad_inputs=None):
    """
    Forward pass through attention head(s).
    
    Args:
        grad_inputs: Set of {'q', 'k', 'v', 'a'} indicating which inputs need gradients.
                    If provided, temporarily replaces QuantLinear with dequantized Linear.
    """
    if type(model) != HookedTransformer:
        decoder_layer = model.model.layers[layer]
        attn = decoder_layer.self_attn
        position_embeddings = decoder_layer.position_embeddings
    else:
        attn = model.blocks[layer].attn

    bsz = max(x.size(0) if x is not None else 1 for x in [xq, xk, xv])
    if attention_mask.size(0) != bsz:  # bsz may be expanded from 1 to steps when scaled_input
        assert attention_mask.size(0) == 1, f'{attention_mask.size()} mismatches {bsz}'
        attention_mask = attention_mask.expand(bsz, -1, -1, -1)
    if not isinstance(head, Iterable): head = [head]
    # Use xk as hidden_states (full sequence) - needed for input_shape in attention
    hidden_states = xk
    with use_dequant_projections(attn, layer, grad_inputs):
        if type(model) != HookedTransformer:
            attn_output = attn(hidden_states, q_hidden_states=xq, k_hidden_states=xk, v_hidden_states=xv, attn_weights=aw,
                attention_mask=attention_mask, position_ids=position_ids, q_position_ids=q_position_ids,
                position_embeddings=position_embeddings,
                no_sink=no_sink, head_ids=head, trim=trim, output_attentions=True)[0]
        else:
            attn_output = attn(hidden_states, q_hidden_states=xq, k_hidden_states=xk, v_hidden_states=xv, attn_weights=aw,
                attention_mask=attention_mask, position_ids=position_ids, q_position_ids=q_position_ids,
                no_sink=no_sink, head_ids=head, trim=trim, output_attentions=True)[0]
    
    return attn_output


def mlp_forward(model, layer, xi, xg=None, grad_inputs=None):
    """
    Forward pass through MLP.
    
    Args:
        xi: Input for up_proj path
        xg: Input for gate_proj path (usually same as xi)
        grad_inputs: Set of {'i', 'g'} indicating which inputs need gradients.
                    If provided, temporarily replaces QuantLinear with dequantized Linear.
    """
    if type(model) != HookedTransformer:
        mlp = model.model.layers[layer].mlp
    else:
        mlp = model.blocks[layer].mlp
        
    with use_dequant_projections(mlp, layer, grad_inputs):
        return mlp(xi, xg=xg)


def lm_head_forward(model, hidden_states, labels=None, logits_mask=None):
    if type(model) != HookedTransformer:
        logits = model.lm_head(hidden_states)
    else:
        logits = model.unembed(hidden_states)
    if logits_mask is not None: logits = logits + logits_mask
    return -compute_loss(logits, labels) if labels is not None else logits


def backward_layernorm(self, output_grad, variance):
    if type(self) == transformer_lens.components.rms_norm.RMSNorm:
        return (self.w * output_grad * torch.rsqrt(variance + self.eps)).to(output_grad.dtype)
    return (self.weight * output_grad * torch.rsqrt(variance + self.variance_epsilon)).to(output_grad.dtype)


def ig_backward(forward_fn, kwarg, gy=None, steps=None, debug=False):
    if steps is None: steps = 4
    key, x = kwarg.popitem()
    with torch.enable_grad():
        scaled_x = scaled_input(x, steps)
        scaled_y = forward_fn(**{key: scaled_x})
        if debug: 
            print(f'scaled_x: requires_grad={scaled_x.requires_grad}, grad_fn={scaled_x.grad_fn}, is_leaf={scaled_x.is_leaf}')
            print(f'scaled_y: requires_grad={scaled_y.requires_grad}, grad_fn={scaled_y.grad_fn}')
        if gy is not None:
            scaled_y = scaled_y * gy.detach()  # freeze upstream gradient
        gx, = torch.autograd.grad(scaled_y.sum(), scaled_x, retain_graph=False, create_graph=False)
    return gx.mean(dim=0, keepdim=True)


def get_lm_head_kwargs(r, model):
    if type(model) != HookedTransformer:
        vocab_size = model.config.vocab_size
    else:
        vocab_size = model.cfg.d_vocab
    logits_mask = (torch.ones(vocab_size) * (-1e4)).to(model.device, dtype=torch.float16)
    logits_mask[r.candidate_ids] = 0  # TODO
    labels = torch.tensor(r.labels)[None, :].to(model.device)
    return dict(labels=labels, logits_mask=logits_mask)


def get_examples_min_arc(r, last_k=2): 
    return r.puzzle['train'][-last_k:]

def get_examples_tom(r): 
    return r.index_map


def get_ranges_min_arc(example, span):
    item_tokens = ['(', 'x', ',', 'y', '):', 'c', ',']
    t2i = {t: i for i, t in enumerate(item_tokens)}
    # span = restore_span(span)
    if span.startswith('A'):  # ans
        indices = [qa.answer.value_ranges['c'][0] for qa in example.output]
    elif span.startswith('V'):  # val
        indices = [example.input_ranges.grid[*qa.answer.key, t2i['c'], 0] for qa in example.output]
    elif span.startswith('QV'):  # qval
        indices = [qa.query.value_ranges['c'][0] for qa in example.output]
    elif span.startswith('Qy'):  # qval
        indices = [qa.query.value_ranges['c'][0]-2 for qa in example.output]
    indices = torch.LongTensor(indices)
    if span.endswith('^'):
        indices[1:] = indices[:-1].clone()
    if span.endswith('-'): indices = indices - 1
    elif span.endswith('+'): indices = indices + 1
    ranges = [slice(i, i + 1) for i in indices] # TODO: assume all spans are of length 1
    return ranges, indices  


def get_ranges_tom(example, span):
    if span.startswith('A'):  # ans
        indices = [example['A-']]
    elif span.startswith('VK'):  # qval
        if span.endswith('C'):
            indices = [example['VK_C']]
        elif span.endswith('I'):
            indices = [example['VK_I']]
        else:
            assert False
    elif span.startswith('V'):  # val
        indices = [example['V']] #hsc
    elif span.startswith('QK_C'):  # qval
        indices = [example['QK_C']]
    elif span.startswith('QK_I'):  # qval
        indices = [example['QK_I']]
    if isinstance(indices, int):
        indices = torch.LongTensor([indices])
    else:
        indices = torch.LongTensor(indices)
    if span.endswith('^'):
        indices[1:] = indices[:-1].clone()
    if span.endswith('-'): indices = indices - 1
    elif span.endswith('+'): indices = indices + 1
    ranges = [slice(i, i + 1) for i in indices] # TODO: assume all spans are of length 1
    return ranges, indices  


# get_examples = get_examples_min_arc
# get_ranges = get_ranges_min_arc

get_examples = get_examples_tom
get_ranges = get_ranges_tom

def get_pos_ids_by_span(r, span):
    return torch.cat([get_ranges(e, span)[1] for e in get_examples(r)])


class Graph(object):
    def __init__(self, dataset_size, hidden_size, set_current=True):
        self.nodes = []
        self.edges = OrderedDict()
        self.dataset_size = dataset_size
        self.hidden_size = hidden_size
        if set_current:
            set_current_graph(self)  # Auto-set as current graph for Node interning
    
    def add_node(self, node):
        if node in self.nodes:   # Prevent duplicates
            print(f'{node} already in graph!'); return
        node.graph = self  # TODO: unnecessary?
        node.dataset_size = self.dataset_size
        node.hidden_size = self.hidden_size
        self.nodes.append(node)

    def remove_node(self, node):
        self.nodes.remove(node)
        for upstream, downstream in list(self.edges.keys()):
            assert downstream != node, f'Can not remove the downstream node of edge {upstream}->{downstream}.'
            if upstream == node: del self.edges[(upstream, downstream)]

    def add_edge(self, upstream, downstream, score):
        if upstream not in self.nodes: self.add_node(upstream)
        assert downstream in self.nodes, f'{downstream} not in graph'
        self.edges[(upstream, downstream)] = score
        # Freeze nodes to prevent modification of hash-relevant attributes
        upstream._frozen = True
        downstream._frozen = True

    def remove_edge(self, upstream, downstream):
        del self.edges[(upstream, downstream)]

    def get_downstreams(self, node):
        return [downstream for (upstream, downstream), _ in self.edges.items() if upstream == node]

    def get_upstreams(self, node):
        return [upstream for (upstream, downstream), _ in self.edges.items() if downstream == node]
    
    def topological_sort(self):
        """Sort nodes from leaves (no incoming edges) to root (lm_head).
        Returns nodes in order such that all upstreams of a node come before it.
        """
        # Build in-degree map
        in_degree = {node: 0 for node in self.nodes}
        for (upstream, downstream), _ in self.edges.items():
            if downstream in in_degree:
                in_degree[downstream] += 1
        
        # Start with leaves (in_degree == 0)
        queue = [n for n in self.nodes if in_degree[n] == 0]
        sorted_nodes = []
        
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for downstream in self.get_downstreams(node):
                if downstream in in_degree:
                    in_degree[downstream] -= 1
                    if in_degree[downstream] == 0:
                        queue.append(downstream)
        
        return sorted_nodes

    def get_reachable(self, start_nodes):
        """Get all nodes reachable downstream from start_nodes (BFS)."""
        reachable = set(start_nodes)
        queue = list(start_nodes)
        while queue:
            node = queue.pop(0)
            for ds in self.get_downstreams(node):
                if ds not in reachable:
                    reachable.add(ds)
                    queue.append(ds)
        return reachable

    def forward(self, r, model, leaf_nodes=None):
        if leaf_nodes is None: # graph leaves - all nodes reachable
            downstream_set = {ds for (_, ds) in self.edges.keys()}
            leaf_nodes = [n for n in self.nodes if n not in downstream_set]
            sorted_nodes = self.topological_sort()
        else: # custom leaves - filter to reachable nodes only
            reachable = self.get_reachable(leaf_nodes)
            sorted_nodes = [n for n in self.topological_sort() if n in reachable]
        
        node_outputs = {}
        for node in sorted_nodes:
            upstream_outputs = [node_outputs[up] for up in self.get_upstreams(node) if up in node_outputs]
            # Leaf nodes use cached hidden_states, others use upstream_sum
            upstream_sum = None if node in leaf_nodes else (sum(upstream_outputs) if upstream_outputs else None)
            output = node.forward(r, model, upstream_sum=upstream_sum)
            if node.is_lm_head(): return output
            node_outputs[node] = output
        raise ValueError("No lm_head node found in graph")


class Node(object):
    _HASH_ATTRS = ('layer', 'head', 'type')  # Attrs used in __eq__/__hash__, protected when frozen
    
    def __new__(cls, layer, head, type, attn_pattern=None, graph=None):
        """Return existing node from graph if found, otherwise create new instance."""
        g = graph or get_current_graph()
        if g is not None:
            for node in g.nodes:
                if node.layer == layer and node.head == head and node.type == type:
                    print(f'{node} already in graph!')
                    return node  # Return existing node
        return super().__new__(cls)
    
    def __setattr__(self, name, value):
        """Prevent modification of hash-relevant attributes after node is frozen (used as edge key)."""
        if name in Node._HASH_ATTRS and getattr(self, '_frozen', False) and getattr(self, name) != value:
            raise AttributeError(
                f"Cannot modify '{name}' from {getattr(self, name)} to {value} after {self} has been used as edge endpoint. "
                f"This would corrupt the graph's edges dict. Create a new Node instead."
            )
        super().__setattr__(name, value)
    
    def __init__(self, layer, head, type, attn_pattern=None, graph=None):
        # Skip re-init if this is an existing node returned by __new__
        if hasattr(self, '_initialized'):
            # Only update attn_pattern if provided and not already set
            if attn_pattern is not None and self.attn_pattern is None:
                self.attn_pattern = attn_pattern
            return
        self._initialized = True
        self._frozen = False  # Will be set True when used as edge endpoint
        self.layer = layer
        self.head = head
        self.type = type
        self.attn_pattern = attn_pattern
        self.graph = graph or get_current_graph()
        self.span = None
        self.src_span = None
        self.dataset_size = None
        self.hidden_size = None
        self.pos_ids = None
        self.output_grad = None
        self.grad = None
        self.dtype = torch.float16
        self.device = 'cpu'

    def __eq__(self, other):
        # Use duck typing to handle module reloads where isinstance fails
        return all(hasattr(other, a) for a in Node._HASH_ATTRS) and \
               all(getattr(self, a) == getattr(other, a) for a in Node._HASH_ATTRS)

    def __hash__(self): return hash(tuple(getattr(self, a) for a in Node._HASH_ATTRS))
    def __str__(self): return f"Node({self.layer},{self.head},'{self.type}')"
    def __repr__(self): return self.__str__()

    def is_mlp(self): return self.type.startswith('mlp')
    def is_attn(self): return self.type.startswith('attn')
    def is_lm_head(self): return self.type.startswith('lm_head')
    
    def get_downstreams(self): return self.graph.get_downstreams(self)

    def set_pos_ids(self, r):
        if self.is_lm_head(): #  'A-->A-'
            self.src_span = self.span = 'A-'
            self.src_pos_ids = self.pos_ids = torch.LongTensor(r.answer_indices) - 1
            return
        downstream = self.get_downstreams()[0]  # TODO: assert all downstreams have the same src_pos_ids
        if self.span is not None:
            assert self.span == downstream.src_span, f'{self.span} != {downstream.src_span}'
        else:
            self.span = downstream.src_span
        self.pos_ids = downstream.src_pos_ids
        if self.type in ['mlp', 'mlp_i', 'mlp_g', 'attn_q', 'attn_qns', 'attn_a']:
            self.src_span = self.span
            self.src_pos_ids = self.pos_ids
        elif self.type == 'attn_v':
            dst, src = self.attn_pattern.split('->')
            if src == dst:
                self.src_span = self.span
                self.src_pos_ids = self.pos_ids
            elif src in ['V', 'A^']:  # TODO: add other spans
                self.src_span = src
                self.src_pos_ids = get_pos_ids_by_span(r, src)
            else:  # TODO
                assert False
        elif self.type == 'attn_k':
            dst, src = self.attn_pattern.split('->')
            if src == dst:
                self.src_span = self.span
                self.src_pos_ids = self.pos_ids
            elif src in ['VK_C', 'VK_I', 'V']:  # TODO: add other spans
                self.src_span = src
                self.src_pos_ids = get_pos_ids_by_span(r, src)
            else:  # TODO
                assert False
        else:  # TODO
            assert False

    def accum_output_grad(self, index):
        if self.type not in ['lm_head']:
            grad = sum(downstream.grad[index] for downstream in self.get_downstreams())
            if self.output_grad is None: self.output_grad = torch.zeros(self.dataset_size, *grad.shape, dtype=self.dtype, device=self.device)
            self.output_grad[index] = grad

    def backward(self, r, model):
        """Unified backward - dispatches to local or remote based on model type.
        Args:
            r: Result object with index and outputs
            model: Either a local nn.Module or a ModelClient instance
        """
        grad = self._backward_remote(r, model) if is_remote(model) else self._backward_local(r, model)
        if self.type == 'attn_a': # attn_a returns (aw, grad) tuple for attribute_attn_weights
            assert isinstance(grad, tuple) and len(grad) == 2
            return grad
        if self.grad is None:
            self.grad = torch.zeros(self.dataset_size, *grad.shape, dtype=self.dtype, device=self.device)
        self.grad[r.index] = grad
    
    def _backward_local(self, r, model):
        outputs = r.outputs
        residual = outputs.hidden_states[self.layer][:, self.pos_ids].to(model.device)
        if self.is_mlp(): residual += get_attn_output(model, r, self.layer, self.pos_ids)
        if self.is_lm_head():
            if type(model) != HookedTransformer:
                ln = model.model.norm
            else:
                ln = model.ln_final
            ln_state = outputs.ln_states[self.layer][:, self.src_pos_ids].to(model.device)
            x = residual  # hidden_states[-1] is already post-layernorm, do NOT ln again
        else:
            if type(model) != HookedTransformer:
                ln_idx, ln_name = (0, 'input_layernorm') if not self.is_mlp() else (1, 'post_attention_layernorm')
                ln = getattr(model.model.layers[self.layer], ln_name)
            else:
                ln_idx, ln_name = (0, 'ln1') if not self.is_mlp() else (1, 'ln2')
                ln = getattr(model.blocks[self.layer], ln_name)
            ln_state = outputs.ln_states[self.layer][ln_idx][:, self.src_pos_ids].to(model.device)
            x = ln(residual)
            if self.is_attn():
                xkv = ln(outputs.hidden_states[self.layer].to(model.device))
                if self.type == 'attn_a':
                    aw = get_attn_weights(model, r, self.layer, self.head, self.pos_ids)#.to(model.device) # TODO: remove comment

        if self.type == 'lm_head':
            fwd_fn, kwarg = partial(lm_head_forward, model, **get_lm_head_kwargs(r, model)), dict(hidden_states=x)
        elif self.type == 'mlp_i':
            fwd_fn, kwarg = partial(mlp_forward, model, self.layer, xg=x, grad_inputs={'i'}), dict(xi=x)
        elif self.type == 'mlp_g':
            fwd_fn, kwarg = partial(mlp_forward, model, self.layer, xi=x, grad_inputs={'g'}), dict(xg=x)
        elif self.type == 'mlp':
            fwd_fn, kwarg = partial(mlp_forward, model, self.layer, xg=None, grad_inputs={'i', 'g'}), dict(xi=x)
        elif self.is_attn():
            attn_kwargs = get_attn_kwargs(*xkv.shape[:2], model.device, model.dtype, pos_ids=self.pos_ids)
            attn_kwargs['trim'] = False
            if self.type == 'attn_q':   frozen_kwargs, kwarg, grad_inputs = dict(xk=xkv, xv=xkv), dict(xq=x), {'q'}
            elif self.type == 'attn_k': frozen_kwargs, kwarg, grad_inputs = dict(xq=x, xv=xkv), dict(xk=xkv), {'k'}
            elif self.type == 'attn_v': frozen_kwargs, kwarg, grad_inputs = dict(xq=x, xk=xkv), dict(xv=xkv), {'v'}
            elif self.type == 'attn_a': frozen_kwargs, kwarg, grad_inputs = dict(xq=x, xk=xkv, xv=xkv), dict(aw=aw), {'a'}
            elif self.type == 'attn_qns': frozen_kwargs, kwarg, grad_inputs = dict(xk=xkv, xv=xkv, no_sink=True), dict(xq=x), {'q'}
            fwd_fn = partial(head_forward, model, self.layer, self.head, grad_inputs=grad_inputs,
                **frozen_kwargs, **attn_kwargs)
        else: assert False, f'Invalid node type: {self.type}'
        
        gy = self.output_grad[r.index].to(x.device) if self.output_grad is not None else None  # None for lm_head
        
        # Use steps=0 for attn_v/a (linear in the grad input), steps=None (default 4) for others
        steps = 0 if self.type in ['attn_v', 'attn_a'] else None
        grad = ig_backward(fwd_fn, kwarg, gy=gy, steps=steps, debug=False)
        if self.type == 'attn_a':
            return aw.to(self.device), grad.to(self.device)
        if self.type in ['attn_k', 'attn_v']: grad = grad[:, self.src_pos_ids]
        grad = backward_layernorm(ln, grad, ln_state).detach().to(self.device)
        return grad

    def _backward_remote(self, r, client):
        output_grad = self.output_grad[r.index] if self.output_grad is not None else None
        steps = 0 if self.is_attn() else None
        
        kwargs = dict(
            sample_id=r.index,
            node_type=self.type,
            layer=self.layer,
            pos_ids=self.pos_ids.tolist(),
            output_grad=output_grad,
            steps=steps,
        )
        
        if self.is_attn():
            kwargs['head'] = self.head
            if self.type in ['attn_k', 'attn_v'] and self.src_pos_ids is not None:
                kwargs['src_pos_ids'] = self.src_pos_ids.tolist()
        
        grad = client.compute_node_gradient(**kwargs)
        if self.type == 'attn_a': # attn_a returns tuple (aw, grad), move both to device
            assert isinstance(grad, tuple) and len(grad) == 2
            return (grad[0].to(self.device), grad[1].to(self.device))
        return grad.to(self.device)

    def forward(self, r, model, upstream_sum=None):
        return self._forward_remote(r, model, upstream_sum) if is_remote(model) else self._forward_local(r, model, upstream_sum)

    def _forward_local(self, r, model, upstream_sum=None):
        """Compute forward output for this node.
        upstream_sum: Optional tensor to REPLACE residual input. Shape bke (bje for attn_k/v).
        Returns: Output tensor bke
        """
        outputs = r.outputs
        
        if self.is_lm_head():
            x = upstream_sum if upstream_sum is not None else outputs.hidden_states[-1].to(model.device)
            return lm_head_forward(model, model.model.norm(x), **get_lm_head_kwargs(r, model))
        
        ln_name = 'input_layernorm' if not self.is_mlp() else 'post_attention_layernorm'
        ln = getattr(model.model.layers[self.layer], ln_name)
        
        if self.is_mlp():
            cached = get_hidden_states(model, r, self.layer, self.pos_ids) + \
                     get_attn_output(model, r, self.layer, self.pos_ids)
            xi = xg = ln(cached)
            if upstream_sum is not None:
                upstream_ln = ln(upstream_sum)
                if self.type == 'mlp_i': xi = upstream_ln
                elif self.type == 'mlp_g': xg = upstream_ln
                else: xi = xg = upstream_ln  # mlp
            return mlp_forward(model, self.layer, xi=xi, xg=xg)
        
        # Attention: For attn_i (i in q,k,v), only xi uses upstream_sum, others use cached
        xk = xv = ln(outputs.hidden_states[self.layer].to(model.device))
        xq = xk[:, self.pos_ids]
        
        if upstream_sum is not None:
            upstream_ln = ln(upstream_sum)
            if self.type == 'attn_k':
                xk = xk.clone(); xk[:, self.src_pos_ids] = upstream_ln
            elif self.type == 'attn_v':
                xv = xv.clone(); xv[:, self.src_pos_ids] = upstream_ln
            else:  # attn_q, attn_qns, attn_a
                xq = upstream_ln

        attn_kwargs = get_attn_kwargs(*xk.shape[:2], model.device, model.dtype, pos_ids=self.pos_ids)
        aw = get_attn_weights(model, r, self.layer, self.head, self.pos_ids).to(model.device) if self.type == 'attn_a' else None
        no_sink = (self.type == 'attn_qns')
        return head_forward(model, self.layer, self.head, xq=xq, xk=xk, xv=xv, aw=aw, no_sink=no_sink, **attn_kwargs)

    def _forward_remote(self, r, client, upstream_sum=None):
        kwargs = dict(
            sample_id=r.index,
            node_type=self.type,
            layer=self.layer,
            pos_ids=self.pos_ids.tolist(),
            upstream_sum=upstream_sum,
        )
        if self.is_attn():
            kwargs['head'] = self.head
            if self.type in ['attn_k', 'attn_v'] and self.src_pos_ids is not None:
                kwargs['src_pos_ids'] = self.src_pos_ids.tolist()
        
        return client.compute_node_forward(**kwargs)

def _attribute_residual(r, model, pos_ids, downstream_grads, downstream_layers, from_layer=0, to_layer=None):
    """Core residual attribution logic. Used by both local and server.
    Args:
        downstream_grads: Stacked gradients tensor [g, i, e]
        downstream_layers: List of layer indices (float for MLP: layer+0.5)
        from_layer: Start layer (inclusive)
        to_layer: End layer (exclusive)
    
    Returns:
        Attribution tensor [num_layers, num_heads+1, num_downstream_grads]
    """
    if to_layer is None:
        to_layer = int(math.floor(max(downstream_layers)))
        
    head_attr = torch.stack([
        einsum('ine,gie->ng', get_head_output(model, r, l, None, pos_ids)[0], downstream_grads)
        for l in range(from_layer, to_layer)])  # lng
    
    mlp_attr = torch.stack([
        einsum('ie,gie->g', r.outputs.mlp_outputs[l][:, pos_ids].to(model.device, dtype=model.dtype)[0], downstream_grads)
        for l in range(from_layer, to_layer)])  # lg
    
    # Layer masks (only attribute to layers before downstream node)
    grad_l = torch.as_tensor(downstream_layers, device=head_attr.device)
    output_l = torch.arange(from_layer, to_layer, device=head_attr.device)
    mask_attn = output_l[:, None] < grad_l[None, :]  # e.g. output_l=4, grad_l=4.5 is OK
    mask_mlp = output_l[:, None] < grad_l[None, :].floor()
    
    head_attr = head_attr * mask_attn.unsqueeze(1).to(head_attr.dtype)
    mlp_attr = mlp_attr * mask_mlp.to(mlp_attr.dtype)
    
    return torch.cat([head_attr, mlp_attr.unsqueeze(1)], dim=1)  # l(n+1)g


def attribute_residual(r, model, nodes, from_layer=0, to_layer=None):
    """Attribute gradients to upstream residual connections. Works with local model or remote client."""
    # downstream_layers: use fractional (layer+0.5) for MLP to distinguish from attention
    downstream_layers = [node.layer + (0.5 if node.is_mlp() else 0.0) for node in nodes]
    pos_ids = nodes[0].pos_ids
    
    if is_remote(model):
        downstream_grads = [node.grad[r.index] for node in nodes]
        return model.attribute_residual(
            sample_id=r.index,
            pos_ids=pos_ids,
            downstream_grads=downstream_grads,
            downstream_layers=downstream_layers,
            from_layer=from_layer,
            to_layer=to_layer,
        )
    
    downstream_grads = torch.cat([node.grad[r.index] for node in nodes]).to(model.device, dtype=model.dtype)  # g*[1ie]->gie
    return _attribute_residual(r, model, pos_ids, downstream_grads, downstream_layers, from_layer, to_layer)


def attribute_step(r, model, nodes):
    for node in nodes:
        node.set_pos_ids(r)
        node.accum_output_grad(r.index)
        node.backward(r, model)

    return attribute_residual(r, model, nodes)


def add_edges(g, upstream, downstreams, attr):
    for i, ds in enumerate(downstreams):
        score = attr[upstream.layer, upstream.head, i].item()
        if score > 0.: g.add_edge(upstream, ds, score)


def attribute_attn_weights(r, model, layer, head, downstreams):
    node = Node(layer, head, 'attn_a')  # create temporal node 
    for downstream in downstreams:  # add temporal edges for set_pos_ids and accum_output_grad
        node.graph.add_edge(node, downstream, 1.)  # any score is OK
        # downstream.set_pos_ids(r)  # 更新 downstream 的 pos_ids 以匹配当前样本
    node.set_pos_ids(r)
    node.accum_output_grad(r.index)
    node.graph.remove_node(node)  # remove temporal node and edges
    aw, ag = node.backward(r, model)
    return aw, ag


def get_attn_attrs_on_dataset(results, model, layer, head, downstreams, normalize=True):
    aw, ag = map(torch.cat, zip(*[attribute_attn_weights(r, model, layer, head, downstreams) for r in results]))
    aa = aw * ag.abs()
    if normalize: aa /= aa.sum(dim=-1, keepdim=True)
    return aw, aa


def data2str(data):
    nodes = data.nodes
    s = 'L' + ','.join(f'{l}' for l in sorted(set(n.layer for n in nodes), reverse=True))
    node = nodes[0]
    if node.is_lm_head(): return s
    s += f' {node.attn_pattern} x{len(nodes)}'
    return s


def get_top_heads(attr, k=30, H=None):
    attr_cpu = attr.detach().to(torch.float32).cpu()
    return OrderedDict(
        ((int(l), int(h)), attr_cpu[l, h].abs().sum().item())  # np.int64 -> int
        for l, h in zip(*topk_md(attr_cpu, k=k)[:2])
        if H is None or h < H
    )


def add_tree_node(results, model, nodes, parent=None):
    d = AttrData(nodes=nodes)
    tnode = TNode(data2str(d), parent=parent); tnode.data = d

    d.attr = mr(attribute_step)(results, model, nodes)  # nodes -> attr
    if type(model) != HookedTransformer:
        H = model.config.num_attention_heads
    else:
        H = model.cfg.n_heads
    d.top_heads = get_top_heads(d.attr, k=30, H=H)  # attr -> top_heads -> attn_attrs_ds
    for l, h in d.top_heads:
        d.attn_weights_ds[(l, h)], d.attn_attrs_ds[(l, h)] = get_attn_attrs_on_dataset(results, model, l, h, nodes)
    return tnode


add_tnode = add_tree_node  # TODO: rename function in notebook