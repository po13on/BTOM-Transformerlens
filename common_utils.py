from __future__ import annotations
import sys
from typing import Optional
import dataclasses
from dataclasses import dataclass, is_dataclass, asdict

import operator
from operator import getitem, setitem
from functools import partial
from collections import defaultdict, OrderedDict, Counter
from collections.abc import Iterable
import types
import math
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
# torch.nn.LayerNorm
# from transformers import GPT2Tokenizer

# cache_dir = '/nas/xd/.cache/torch/transformers/'
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

def join_lists(x, dedup=False):
    l = list(chain.from_iterable(x))
    if dedup: l = list(OrderedDict.fromkeys(l)) # list(set(l)) # to keep order
    return l

def list_diff(l1, l2):  # will preserve order of elements in l1 compared to list(set(l1) - set(l2))
    l2 = set(l2)
    return [x for x in l1 if x not in l2]

def reverse(l): return list(reversed(l))

def numpy(a, decimals=None):
    v = np.array(a) if isinstance(a, list) else a.detach().cpu().numpy()
    if decimals is not None: v = v.round(decimals)
    return v

def show_topk(values, indices, values_fn=lambda x: numpy(x, decimals=3), indices_fn=None, transpose=False):
    if indices_fn is None:
        indices_fn = show_topk.indices_fn if getattr(show_topk, 'indices_fn', None) is not None else lambda x: x
    return dict(OrderedDict(zip(indices_fn(indices), values_fn(values))))

def topk_md(tensor, k, largest=True, transpose=False):
    k = min(tensor.numel(), k)
    if tensor.ndim == 1:
        values, indices = tensor.topk(k, largest=largest)
        return indices.numpy(), values.numpy()
    values, indices = tensor.flatten().topk(k, largest=largest)
    values, indices = values.cpu(), indices.cpu()
    # https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    coords = np.unravel_index(indices.numpy(), tensor.shape)
    return (*coords, values.numpy()) if not transpose else list(zip(*coords, values.numpy()))

def topi_md(tensor, rows, cols):  # somewhat reverse of topk_md
    if not isinstance(rows, Iterable): return (tensor > tensor[rows, cols]).sum().item()
    all_sorted = torch.LongTensor(np.array(topk_md(tensor, tensor.numel())[:2]))
    rows_cols = torch.LongTensor(np.array([rows, cols]))
    if not isinstance(rows, Iterable): rows_cols = rows_cols.unsqueeze(1)
    a = ((all_sorted.T @ rows_cols) == 2).nonzero() # (ln)2,2k->(ln)k->k2
    topi = [x[0] for x in sorted(a.tolist(), key=lambda x: x[1])]
    if not isinstance(rows, Iterable): topi = topi[0]
    return topi

def to_df(*args): return pd.DataFrame(list(zip(*args)))

def norm(tensor, p=2): return tensor.norm(p=p, dim=-1).mean().round().item()

def prepare_inputs(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs

def convert_ids_to_tokens(ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    wrapped = False
    if type(tokens) == str: tokens, wrapped = [tokens], True
    out = [tokenizer.convert_tokens_to_string([token]) for token in tokens]
    if wrapped: out = out[0]  # unwrap
    return out

def mask_logits(logits, indices, kept_ids):
    mask = torch.ones_like(logits) * (-1e9) # biv
    for i, ids in zip(indices, kept_ids): mask[:, i, ids] = 0
    return logits + mask

def my_isinstance(obj, type_):  # to cope with autoreload
    # return isinstance(obj, type_)  # fail on autorelaod
    return obj.__class__.__name__ == type_.__name__ if not isinstance(type_, tuple) \
        else any(obj.__class__.__name__ == t.__name__ for t in type_)

def equal(a, b):
    assert a.size() == b.size(), f'{a.size()} != {b.size()}'
    return (a - b).abs().mean(), a.abs().mean(), b.abs().mean()

def einsum(
    eq: str,
    *tensors: torch.Tensor,
) -> torch.Tensor:
    """Drop dimensions of size 1 to allow broadcasting.
    from https://github.com/pytorch/pytorch/issues/48420"""
    lhs, rhs = eq.split("->")
    # squeeze
    mod_ops, mod_t = [], []
    for op, t in zip(lhs.split(","), tensors):
        mod_op = ""
        for i, c in reversed(list(enumerate(op))):
            if t.shape[i] == 1:
                t = t.squeeze(dim=i)
            else:
                mod_op = c + mod_op
        mod_ops.append(mod_op)
        mod_t.append(t)
    m_lhs = ",".join(mod_ops)
    r_keep_dims = set("".join(mod_ops))
    m_rhs = "".join(c for c in rhs if c in r_keep_dims)
    m_eq = f"{m_lhs}->{m_rhs}"
    mod_r = torch.einsum(m_eq, *mod_t)
    # unsqueeze
    for i, c in enumerate(rhs):
        if c not in r_keep_dims:
            mod_r = mod_r.unsqueeze(dim=i)
    return mod_r

def binarize(a, fraction=0.33):
    b = torch.zeros_like(a)
    prev_v = None
    for r, c, v in zip(*topk_md(a, 20)):
        if prev_v is not None and abs(v) < abs(prev_v) * fraction: break
        b[r, c] = 1; prev_v = v
    return b

# adapted from: https://dzone.com/articles/python-timer-class-context
# from timeit import default_timer
from datetime import datetime

class Printer(object):
    def __init__(self, fn, mode='w'):
        self.file = open(fn, mode)
        
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self.file
        
    def __exit__(self, *args):
        sys.stdout = self.original_stdout

class Timer(object):
    def __init__(self, msg='', verbose=True):
        self.verbose = verbose
        # self.timer = default_timer
        self.msg = msg
        
    def __enter__(self):
        if self.verbose: print(self.msg, '...', end=' ')
        self.start = datetime.now() # self.timer()
        return self
        
    def __exit__(self, *args):
        end = datetime.now() # self.timer()
        self.elapsed = str(end - self.start)#.split('.')[0]
        # self.elapsed_secs = end - self.start
        # self.elapsed = self.elapsed_secs #* 1000   # millisecs
        if self.verbose: print('done', self.elapsed)
            # print('elapsed time: %d s' % self.elapsed)

# https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
class TeeLogger(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, message): self.file.write(message); self.stdout.write(message)  
    def flush(self): self.file.flush()
    def __del__(self): sys.stdout = self.stdout; self.file.close()

# https://stackoverflow.com/questions/2461170/tree-implementation-in-python
# @dataclass
# class Node:
#     data: tuple = ()
#     parent: Optional[Node] = None
#     children: list[Node] = field(default_factory=list)

def traverse(node, fn):
    fn(node)
    for child in node.children: traverse(child, fn)

def reduce_objects(objs, fields=None, reduce_fn='mean'):
    if not isinstance(objs, list): objs = list(objs)
    denorm = len(objs) if reduce_fn == 'mean' else 1
    if isinstance(objs[0], torch.Tensor) or not isinstance(objs[0], dict) and not hasattr(objs[0], '__dict__'):
        # print('In reduce_objects:', [type(o) for o in objs])
        return sum(objs) / denorm
    
    obj, _fields, getter, setter = ({}, objs[0].keys(), getitem, setitem) if isinstance(objs[0], dict) else \
        (type(objs[0])(), objs[0].__dict__.keys(), getattr, setattr)   # dict or (data)class object
    _fields = [f for f in _fields if getter(objs[0], f) is not None]
    fields = set(_fields).intersection(set(fields)) if fields is not None else _fields
    for field in fields: setter(obj, field, sum(getter(o, field) for o in objs) / denorm)
    return obj

def mr(fn):
    def mapreduced_fn(data, *args, **kwargs):
        return reduce_objects([fn(d, *args, **kwargs) for d in data])
    return mapreduced_fn
    
def maybe_mr(bool_fn, reduce_fn=all):
    def wrapped_fn(x, *args, **kwargs):
        return reduce_fn([bool_fn(i, *args, **kwargs) for i in x]) \
            if isinstance(x, Iterable) else bool_fn(x, *args, **kwargs)
    return wrapped_fn

def iterable(item):
    return isinstance(item, Iterable) and not isinstance(item, (tuple, str, torch.Tensor))

def pad(input, dim, to_len, pad_left=False, **kwargs):
    if input.size(dim) == to_len: return input
    padding = [0] * ((input.ndim - dim) * 2)
    padding[-1-int(pad_left)] = to_len - input.size(dim)
    return F.pad(input, tuple(padding), **kwargs)

def sum_except(input, dims):
    if isinstance(dims, int): dims = [dims]
    dims = [d for d in range(input.ndim) if d not in dims]
    return input.sum(dims) if len(dims) > 0 else input

def maybe_map(fn, *iters):
    if iterable(iters[0]):
        res = list(map(fn, *iters))
        return {k: [d[k] for d in res] for k in res[0]} if isinstance(res[0], dict) else res
    return fn(*iters)

def lget(l, i, default=None): return l[i] if l is not None and len(l) > i else default

def list_get(l, i, reduce_fn=None):
    if reduce_fn is None: reduce_fn = lambda x: x
    if i is None: return reduce_fn(l) 
    if not isinstance(i, Iterable): return l[i]
    return reduce_fn([l[j] for j in i])

import inspect
def get_default_value(fn, name):
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    if isinstance(fn, types.FunctionType): return inspect.signature(fn).parameters[name].default
    assert isinstance(fn, partial); return fn.keywords[name]

def fn2str(fn, excluded_keys=[]):
    if isinstance(fn, types.FunctionType): return fn.__name__
    assert isinstance(fn, partial), str(fn)
    def convert_value(k, v):
        if k in excluded_keys: return '...'
        if isinstance(v, torch.Tensor): return v.size()
        if isinstance(v, types.FunctionType): return v.__name__
        return v
    return fn.func.__name__ + '[' + ','.join(f'{k}={convert_value(k, v)}' for k, v in fn.keywords.items()) + ']'

def fisher_discriminant_ratio(x, y, labels=['▁Yes', '▁No'], plot=True):
    x = np.array(x); y = np.array(y, dtype=np.float32)
    y0 = y[x == labels[0]]; y1 = y[x == labels[1]]
    m0 = y0.mean(0); m1 = y1.mean(0)
    fdr = np.square(m0 - m1).sum() / (np.var(y0, axis=0).sum() + np.var(y1, axis=0).sum())
    title = f'ratio = {fdr:.2f}'
    if plot:
        if y.ndim == 1:
            # plt.hist([Y0, Y1], label=labels)
            plt.hist(y0, alpha=0.5, label=labels[0])
            plt.hist(y1, alpha=0.5, label=labels[1])
        elif y.ndim == 2:
            plt.plot(y0[:, 0], y0[:, 1], 'gx', alpha=0.5, label=labels[0]);
            plt.plot(y1[:, 0], y1[:, 1], 'rx', alpha=0.5, label=labels[1]);
            # print('gx', y0[:, 1]); print('rx', y1[:, 1])  # nrk debug
            line_range = [min(np.min(y[:, 0]), np.min(y[:, 1])), max(np.max(y[:, 0]), np.max(y[:, 1]))]
            plt.plot(line_range, line_range, color='k', linestyle='-', alpha=0.2)
            logits_diff = np.concatenate([y0[:, 1] - y0[:, 0], y1[:, 0] - y1[:, 1]]).mean()
            title += f' logits_diff = {logits_diff:.2f}'
        plt.legend(loc='best')  # upper right
        plt.title(title)
        plt.show()
    return fdr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def move_model_files(model_name = 'gpt-neox-20b',
        dir0 = '/nas/xd/.cache/torch/transformers',
        dir1 = '/mnt/nvme1/xd/.cache/torch/transformers'):
    import glob
    import os, shutil
    from pathlib import Path

    for link in glob.glob(f'{dir0}/{model_name}*'):
        tgt = Path(link).resolve()
        with Timer(f'copying {link}'): shutil.copy(tgt, dir1)
        os.symlink(tgt, os.path.join(dir1, os.path.split(link)[-1]))
    
