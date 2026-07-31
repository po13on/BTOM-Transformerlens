import torch
import numpy as np
import transformer_lens
from transformer_lens import HookedTransformer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Iterable, Callable
from einops import rearrange

from common_utils import join_lists, numpy, show_topk
from model_hooks import get_head_output, get_attn_output, get_attn_weights
from attribute import attribute_attn_weights, get_examples, get_ranges


def get_ap_scores(aw, attn_labels, ranges_q):
    assert (aw >= 0).all()
    b = attn_labels.sum(1) > 0  # ij->i
    assert b.sum() == len(ranges_q), f'{b.sum()} != {len(ranges_q)}'
    if b.sum() == len(ranges_q):
        scores = (aw[..., b, :] * attn_labels[b, :]).sum(-1)  # ...kj,kj->...kj->...k
    else:
        scores = (aw * attn_labels).sum(-1)  # ...ij,ij->...ij->...i
        scores = torch.cat([scores[..., rq_slice].max(-1, keepdim=True).values if not isinstance(rq_slice, Iterable)
            else torch.cat([scores[..., _rq_slice].max(-1, keepdim=True).values for _rq_slice in rq_slice], dim=-1).mean(-1, keepdim=True)
            for rq_slice in ranges_q], dim=-1)  # k*[...1]->...k
    return scores#, scores.mean(-1)  # ...k->..., e.g. lnk->ln


def get_head_matching_scores(result, attn_patterns, model, layer, head, last_k=2):
    if isinstance(model, transformer_lens.HookedTransformer):
        is_mlp = head == model.cfg.n_heads
    else:
        is_mlp = head == model.config.num_attention_heads
    _attn_patterns = attn_patterns if isinstance(attn_patterns, list) else [attn_patterns]
    tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in model.tokenizer.tokenize(result.prompt)]
    # pos_ids = torch.LongTensor(result.answer_indices) - 1
    pos_ids = None
    if not is_mlp: attn_weights = get_attn_weights(model, result, layer, head, pos_ids=pos_ids)
    matching_scores = {}
    for attn_pattern in _attn_patterns:
        if is_mlp:
            dst, src = attn_pattern.split('->')
            matching_scores[attn_pattern] = torch.tensor(float(src == dst))  # == 1.
            continue
        attn_labels, ranges_q = attn_pattern2labels(get_examples(result), attn_pattern, tokens)
        if pos_ids is not None: attn_labels = attn_labels[pos_ids, :]  # [q_seq_len[pos_ids], kv_seq_len]
        ap_scores = get_ap_scores(attn_weights, attn_labels.to(attn_weights.device), ranges_q)
        matching_scores[attn_pattern] = ap_scores
    return matching_scores if isinstance(attn_patterns, list) else list(matching_scores.values())[0]


def attn_pattern2labels(examples, attn_pattern, tokens, normalize=True):
    seq_len = len(tokens)
    attn_labels = torch.zeros(seq_len, seq_len)
    ranges_q = []
    dst, src = attn_pattern.split('->')
    for example in examples:
        src_ranges, src_indices = get_ranges(example, src)
        dst_ranges, dst_indices = get_ranges(example, dst)
        attn_labels[dst_indices, src_indices] = 1.
        ranges_q.extend(dst_ranges)
    if normalize: attn_labels = attn_labels / (attn_labels.sum(1, keepdim=True) + 1e-9)
    return attn_labels, ranges_q


def eval_logit_lens(result, model, tokenizer, from_layer=0, to_layer=None, topk=3, verbose=True):
    if not verbose: print = lambda *args, **kwargs: None
    outputs = result.outputs
    pos_ids = torch.LongTensor(result.answer_indices) - 1
    if to_layer is None: to_layer = model.config.num_hidden_layers
    if hasattr(model.model.norm, 'variance'): delattr(model.model.norm, 'variance')
    acc, logprobs = [], []
    for l in range(from_layer, to_layer):
        residual = outputs.hidden_states[l][:, pos_ids].to(model.device)
        attn_output = get_attn_output(model, result, l, pos_ids)
        mlp_output = outputs.mlp_outputs[l][:, pos_ids].to(model.device)
        print(f'Layer {l}:')
        _acc, _logprobs = [], []
        for o in [0, attn_output, mlp_output]:
            residual += o
            logits = model.lm_head(model.model.norm(residual))
            labels = torch.tensor(result.labels).to(logits.device)
            _acc.append((logits[0].argmax(dim=-1) == labels).float().mean())
            _logprobs.append(logits[0].log_softmax(dim=-1)[torch.arange(logits.shape[0]), labels].mean())
            if not verbose: continue
            for val, ind in zip(*logits[0].softmax(dim=-1).topk(topk, dim=-1)):
                values_fn=lambda x: (numpy(x) * 100).astype(int)
                indices_fn=lambda ids: [t.replace('Ġ', ' ') for t in tokenizer.convert_ids_to_tokens(ids)]
                print(show_topk(val, ind, values_fn=values_fn, indices_fn=indices_fn), end=' ')
            print()
        acc.append(_acc)
        logprobs.append(_logprobs)
    return dict(acc=torch.tensor(acc), logprobs=torch.tensor(logprobs))


def show_logit_lens(acc_or_logprobs, from_layer=0):
    n_layers, n_bars = acc_or_logprobs.shape  # Should be [n_layers, 3]
    fig, ax = plt.subplots(figsize=(12, 6//2))
    bar_width = 0.25
    group_spacing = 0.3  # Extra space between layer groups
    layer_positions = np.arange(n_layers) * (n_bars * bar_width + group_spacing)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['resid', 'attn', 'mlp']

    for i in range(n_bars):
        positions = layer_positions + i * bar_width
        _ = ax.bar(positions, acc_or_logprobs[:, i], bar_width, label=labels[i], color=colors[i])

    # ax.set_xlabel('Layer', fontsize=12)
    # ax.set_ylabel('Accuracy', fontsize=12)
    # ax.set_title('Accuracy by Layer and Metric', fontsize=14, fontweight='bold')
    ax.set_xticks(layer_positions + bar_width)
    ax.set_xticklabels([f'{i+from_layer}' for i in range(n_layers)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def eval_head_lens(result, model, layer, head, strict=False, reduce='mean'):
    outputs = result.outputs
    pos_ids = torch.LongTensor(result.answer_indices) - 1
    output = get_head_output(model, result, layer, head, pos_ids) \
        if head < (model.cfg.n_heads if isinstance(model, transformer_lens.HookedTransformer) else model.config.num_attention_heads) else \
        outputs.mlp_outputs[layer][:, pos_ids].to(model.device)
    if isinstance(model, transformer_lens.HookedTransformer):
        logits = model.unembed(model.ln_final(output))
    else:
        logits = model.lm_head(model.model.norm(output))
    labels = torch.tensor(result.labels).to(logits.device)
    predictions = logits[0].argmax(dim=-1)
    pred_tokens = [t.replace('Ġ', ' ') for t in model.tokenizer.convert_ids_to_tokens(predictions)]
    is_correct = predictions == labels if strict else \
        torch.BoolTensor([p.strip().lower() == a.strip().lower() for p, a in zip(pred_tokens, result.answers)])
    return is_correct.float().mean() if reduce == 'mean' else is_correct  # 'none'


def eval_nodes(result, model, nodes):
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    outputs = result.outputs
    pos_ids = torch.LongTensor(result.answer_indices) - 1
    if len(nodes) == 1 and (node := nodes[0]).is_lm_head():
        output = outputs.hidden_states[node.layer][:, pos_ids].to(model.device)
    else:
        output = sum(get_head_output(model, result, node.layer, node.head, pos_ids) for node in nodes)
        output = model.model.norm(output)
    logits = model.lm_head(output)
    labels = torch.tensor(result.labels).to(logits.device)
    acc = (logits[0].argmax(dim=-1) == labels).float().mean()
    return acc


def show_attn(r, model, layer, head, downstreams=None, last_k=1, start=None, stop=None, pos_ids=None):
    for resp, b in zip(r.responses, r.is_corrects): print(resp+('✓' if b else '✗'), end=' ')
    aw = get_attn_weights(model, r, layer, head)
    actw = aw.clone()  # activation weights
    actw[:, :, 0] = 0  # clear attn to bos (attn sink)
    actw = actw.sum(-1).unsqueeze(1)  # 1ij->1i->11i

    # answers = r.answers[-r.Q_train*last_k:]
    # pos_ids = torch.LongTensor(r.answer_indices[-r.Q_train*last_k:]) - 1
    answers = r.answers[-last_k:]
    pos_ids = torch.LongTensor([pos_ids]) if pos_ids is not None else torch.LongTensor(r.answer_indices[-last_k:]) - 1
    aw = aw[:, pos_ids]  # 1ij-1kj
    if downstreams is not None:
        ag = attribute_attn_weights(r, model, layer, head, downstreams)[1].to(aw.device)
        ag = ag[:, -len(pos_ids):]
        aa = aw * ag.abs()
        aa = aa / aa.sum(-1, keepdim=True)
    qpos_weights = torch.zeros_like(aw); qpos_weights[:, torch.arange(len(pos_ids)), pos_ids] = 1
    aws = [qpos_weights, aw] + ([aa] if downstreams is not None else [])
    aw = rearrange(torch.cat(aws), 'b k j -> 1 (k b) j')  # 1kj+1kj->2kj->(k2)j
    aw = torch.cat([actw, aw], dim=1)  # 11j+1(k2)j->1(1+k2)j
    labels = ['act'] + join_lists([['q->', a+('✓' if b else '✗')] + (['aa'] if downstreams is not None else [])
        for a, b in zip(answers, eval_head_lens(r, model, layer, head, reduce='none'))])
    tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in model.tokenizer.tokenize(r.prompt)]
    start = start if start is not None else r.puzzle['train'][-last_k].input_ranges.prefix['grid'][0] - 5
    stop = stop if stop is not None else len(tokens)
    print(f'start={start}, stop={stop}')
    return tokens[start:stop], aw[0, :, start:stop].T, labels

def compute_cosine_matrix(
    data_dict: Dict[Tuple[int, int], torch.Tensor],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute pairwise cosine distance matrix between vectors.
    
    Cosine distance = 1 - cosine_similarity
    
    Suitable for attribution scores (attn_weights * attn_weights_grad) which are
    NOT probability distributions and can be negative.
    
    Args:
        data_dict: Dict mapping (layer, head) -> tensor of shape (batch, query_len, kv_len)
                   or any shape that can be flattened to a vector per head
    
    Returns:
        dist_matrix: NxN numpy array of cosine distances (0 = identical, 2 = opposite)
        head_keys: List of (layer, head) tuples in matrix order
    """
    head_keys = list(data_dict.keys())
    n = len(head_keys)
    
    # Flatten each tensor to a 1D vector and stack
    vectors = []
    for k in head_keys:
        v = data_dict[k].float().flatten()
        vectors.append(v)
    vectors = torch.stack(vectors)  # [n_heads, flattened_dim]
    
    # Normalize to unit vectors
    norms = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    vectors_normed = vectors / norms
    
    # Cosine similarity matrix: [n, n]
    cos_sim = torch.mm(vectors_normed, vectors_normed.T)
    
    # Convert to distance: 1 - similarity (range [0, 2])
    dist_matrix = (1 - cos_sim).cpu().numpy()
    
    # Ensure diagonal is exactly 0 and matrix is symmetric
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    
    return dist_matrix, head_keys


def compute_js_matrix(aw_dict: Dict[Tuple[int, int], torch.Tensor]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute pairwise Jensen-Shannon Divergence matrix between attention patterns.
    
    JS(P||Q) = (KL(P||M) + KL(Q||M)) / 2, where M = (P + Q) / 2
    
    Args:
        aw_dict: Dict mapping (layer, head) -> attention weights tensor of shape (batch, query_len, kv_len)
    
    Returns:
        js_matrix: NxN numpy array of JS divergences
        head_keys: List of (layer, head) tuples in matrix order
    """
    head_keys = list(aw_dict.keys())
    n = len(head_keys)
    js_matrix = np.zeros((n, n))
    
    # Only compute upper triangle (JS is symmetric)
    eps = 1e-8
    for i, k1 in enumerate(head_keys):
        for j, k2 in enumerate(head_keys):
            if i >= j:
                continue
            # Convert to float32 and add smoothing to avoid log(0)
            p = aw_dict[k1].float() + eps
            q = aw_dict[k2].float() + eps
            # Renormalize after smoothing
            p = p / p.sum(dim=-1, keepdim=True)
            q = q / q.sum(dim=-1, keepdim=True)
            m = (p + q) / 2
            # Use torch.xlogy for safe computation: xlogy(x, y) = x * log(y), with xlogy(0, 0) = 0
            kl_pm = (p * torch.log(p / m)).sum(dim=-1).mean().item()
            kl_qm = (q * torch.log(q / m)).sum(dim=-1).mean().item()
            js_val = (kl_pm + kl_qm) / 2
            js_matrix[i, j] = js_val
            js_matrix[j, i] = js_val  # Mirror for symmetry
    
    # Ensure symmetry even with NaN (NaN != NaN breaks scipy's symmetry check)
    # js_matrix = np.nan_to_num(js_matrix, nan=0.0)
    # js_matrix = (js_matrix + js_matrix.T) / 2
    # np.fill_diagonal(js_matrix, 0)
    # print(js_matrix)
    return js_matrix, head_keys


def cluster_heads(
    data_dict: Dict[Tuple[int, int], torch.Tensor],
    dist_fn: Callable = None,
    threshold: float = None,
    method: str = 'average',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 3),
    width_ratios: Tuple[float, float] = (2, 1),
    bar_height: float = 0.9,  # Max bar height as fraction of threshold (0.1-1.0)
    show_plot: bool = True,
    leaf_font_size: int = 9,
    leaf_rotation: int = 0,  # 0 = horizontal, 90 = vertical
    # node_type: str = 'attn_v',
    strengths: Optional[Dict[Tuple[int, int], float]] = None,
    model_config=None,  # If provided, color by GQA group
) -> Tuple[Dict[int, List], Dict[int, Dict[str, float]], np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering of attention heads.
    
    Args:
        data_dict: Dict mapping (layer, head) -> tensor (attention weights or attribution scores)
        dist_fn: Distance function, either compute_js_matrix or compute_cosine_matrix.
                 Default: compute_js_matrix
        threshold: Distance threshold for cutting dendrogram. Default depends on dist_fn:
                   - JS: 0.8 (range ~[0, 1])
                   - Cosine: 0.3 (range [0, 2])
        method: Linkage method ('average', 'complete', 'single', 'ward').
                Note: 'ward' works best with Euclidean-like distances (JS).
        save_path: Path to save the plot
        figsize: Base figure size (width, height). Height auto-extends when strengths provided.
        width_ratios: Relative widths of (dendrogram/bar column, heatmap column). Default (2, 1).
        bar_height: Max bar height as fraction of threshold (0.1-1.0). Default 0.9.
        show_plot: Whether to display the plot
        leaf_font_size: Font size for leaf labels in dendrogram
        leaf_rotation: Rotation angle for leaf labels (0=horizontal, 90=vertical). Default 0.
        # node_type: Type string for created Node objects  # TODO: remove
        strengths: Optional dict mapping (layer, head) -> strength (e.g., sum of outgoing edge scores).
                   If provided, displays a bar chart superposed on the dendrogram.
        model_config: Optional model config with num_attention_heads and num_key_value_heads.
                      If provided, colors leaf labels and bars by GQA group.
    
    Returns:
        groups: Dict mapping cluster_id -> list of (layer, head) tuples  # TODO: remove Node objects
        metrics: Dict mapping cluster_id -> {
            'n_heads': int,
            'intra_mean': float,  # Cohesion: avg within-cluster distance
            'intra_max': float,   # Cohesion: max within-cluster distance
            'sep_min': float,     # Separation: min distance to other clusters
            'sep_mean': float,    # Separation: avg nearest-neighbor distance
        }
        Also includes '_overall' key with global metrics:
            {'n_clusters', 'n_heads', 'avg_cohesion', 'avg_separation', 'sep_cohesion_ratio'}
        dist_matrix: NxN distance matrix
        linkage_matrix: Scipy linkage matrix
    """
    # Set defaults based on distance function
    if dist_fn is None:
        dist_fn = compute_js_matrix
    
    is_js = dist_fn == compute_js_matrix
    if threshold is None:
        threshold = 0.8 if is_js else 0.3
    metric_name = 'JS' if is_js else 'Cosine'
    vmax_default = 1.0 if is_js else 2.0
    
    # Compute distance matrix
    dist_matrix, head_keys = dist_fn(data_dict)
    head_labels = [f'{layer}-{head}' for layer, head in head_keys]
    # print(dist_matrix.shape, head_keys, dist_matrix)
    # Hierarchical clustering
    dist_condensed = squareform(dist_matrix)
    linkage_matrix = linkage(dist_condensed, method=method)
    
    # Cut tree to get clusters
    cluster_ids = fcluster(linkage_matrix, t=threshold, criterion='distance')
    
    # Group heads by cluster
    groups_tuples = {}
    for idx, cluster_id in enumerate(cluster_ids):
        if cluster_id not in groups_tuples:
            groups_tuples[cluster_id] = []
        groups_tuples[cluster_id].append(head_keys[idx])
    
    # Compute cluster metrics (intra + inter)
    key_to_idx = {k: i for i, k in enumerate(head_keys)}
    metrics = {}
    
    # Build index sets per cluster for inter-cluster computation
    cluster_indices = {cid: [key_to_idx[h] for h in heads] for cid, heads in groups_tuples.items()}
    
    for cluster_id, heads in groups_tuples.items():
        indices = cluster_indices[cluster_id]
        
        # Intra-cluster: cohesion (lower = tighter)
        if len(heads) == 1:
            intra_mean, intra_max = 0.0, 0.0
        else:
            pairwise_intra = [dist_matrix[i, j] for i in indices for j in indices if i < j]
            intra_mean = round(np.mean(pairwise_intra), 3)
            intra_max = round(np.max(pairwise_intra), 3)
        
        # Inter-cluster: separation (higher = better separated)
        # For each point in this cluster, find min distance to any point in other clusters
        other_indices = [idx for cid, idxs in cluster_indices.items() if cid != cluster_id for idx in idxs]
        if other_indices:
            sep_distances = []
            for i in indices:
                min_dist_to_other = min(dist_matrix[i, j] for j in other_indices)
                sep_distances.append(min_dist_to_other)
            sep_min = round(np.min(sep_distances), 3)  # Closest point to another cluster
            sep_mean = round(np.mean(sep_distances), 3)  # Average nearest-neighbor distance
        else:
            sep_min, sep_mean = float('inf'), float('inf')  # Only one cluster
        
        metrics[cluster_id] = {
            'n_heads': len(heads),
            'intra_mean': intra_mean,  # Cohesion: avg within-cluster distance
            'intra_max': intra_max,    # Cohesion: max within-cluster distance
            'sep_min': sep_min,        # Separation: min distance to other clusters
            'sep_mean': sep_mean,      # Separation: avg nearest-neighbor to other clusters
        }
    
    # Overall summary metrics
    if len(groups_tuples) > 1:
        all_intra = [m['intra_mean'] for m in metrics.values()]
        all_sep = [m['sep_mean'] for m in metrics.values() if m['sep_mean'] != float('inf')]
        # Ratio: separation / cohesion (higher = better clustering)
        avg_cohesion = np.mean(all_intra) if all_intra else 0
        avg_separation = np.mean(all_sep) if all_sep else 0
        metrics['_overall'] = {
            'n_clusters': len(groups_tuples),
            'n_heads': len(head_keys),
            'avg_cohesion': round(avg_cohesion, 3),
            'avg_separation': round(avg_separation, 3),
            'sep_cohesion_ratio': round(avg_separation / max(avg_cohesion, 1e-6), 2),
        }
    
    # Convert to Node objects (interned from current graph if set)
    groups = groups_tuples  # {cid: [Node(*h, node_type) for h in heads] for cid, heads in groups_tuples.items()}
    
    # Compute GQA group colors if model_config provided
    # GQA group = (layer, kv_group) - only heads in same layer AND same kv_group share color
    gqa_colors = None  # (layer, head) -> color or None (if singleton)
    if model_config is not None:
        if isinstance(model_config, transformer_lens.HookedTransformerConfig):
            num_heads = model_config.n_heads
            num_kv_heads = model_config.n_key_value_heads
            heads_per_group = num_heads // num_kv_heads
        else:
            num_heads = model_config.num_attention_heads
            num_kv_heads = model_config.num_key_value_heads
            heads_per_group = num_heads // num_kv_heads
        
        # Group by (layer, kv_group)
        from collections import defaultdict
        group_members = defaultdict(list)  # (layer, kv_group) -> list of (layer, head)
        for l, h in head_keys:
            kv_group = h // heads_per_group
            group_members[(l, kv_group)].append((l, h))
        
        # Only color groups with 2+ members (singletons stay default)
        multi_member_groups = {k: v for k, v in group_members.items() if len(v) >= 2}
        unique_groups = sorted(multi_member_groups.keys())
        
        # Use a colormap with enough distinct colors, skipping gray (index 7 in tab10)
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10') if len(unique_groups) <= 10 else cm.get_cmap('tab20')
        # Skip index 7 (gray) in tab10 - too similar to default black text
        skip_indices = {7} if cmap.N == 10 else set()
        def get_color(i):
            idx = i
            for skip in sorted(skip_indices):
                if idx >= skip: idx += 1
            return cmap(idx % cmap.N)
        group_to_color = {g: get_color(i) for i, g in enumerate(unique_groups)}
        
        # Map each head to its color (None for singletons)
        gqa_colors = {}
        for (l, h) in head_keys:
            kv_group = h // heads_per_group
            group_key = (l, kv_group)
            if group_key in multi_member_groups:
                gqa_colors[(l, h)] = group_to_color[group_key]
            else:
                gqa_colors[(l, h)] = None  # Singleton - use default color
    
    # Visualization
    if show_plot or save_path:
        # Simple 2-column layout (strength bars overlaid on dendrogram if provided)
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': width_ratios})
        ax1, ax2 = axes

        # Dendrogram
        dendro = dendrogram(linkage_matrix, labels=head_labels, leaf_font_size=leaf_font_size, leaf_rotation=leaf_rotation, ax=ax1)
        ax1.set_ylabel(f'{metric_name} Distance')
        ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label=f'Cut @ {threshold}')
        ax1.legend()
        
        # Reorder matrix by clustering
        order = dendro['leaves']
        dist_ordered = dist_matrix[order][:, order]
        labels_ordered = [head_labels[i] for i in order]

        # Color dendrogram leaf labels by GQA group (skip singletons - keep black)
        if gqa_colors is not None:
            for tick_label, leaf_idx in zip(ax1.get_xticklabels(), order):
                color = gqa_colors[head_keys[leaf_idx]]
                if color is not None:
                    tick_label.set_color(color)
        
        # Heatmap
        vmax = min(vmax_default, np.percentile(dist_matrix[dist_matrix > 0], 95))
        im = ax2.imshow(dist_ordered, cmap='RdYlBu', vmin=0, vmax=vmax)
        ax2.set_xticks(range(len(labels_ordered)))
        ax2.set_yticks(range(len(labels_ordered)))
        ax2.set_xticklabels(labels_ordered, rotation=90, fontsize=7, ha='center')
        ax2.set_yticklabels(labels_ordered, fontsize=7)
        # plt.colorbar(im, ax=ax2, label=f'{metric_name} Distance')
        
        # Strength bars (overlaid on dendrogram with transparency)
        if strengths is not None:
            # Get x-coordinates matching dendrogram leaf positions (5, 15, 25, ...)
            leaf_x = dendro['leaves']
            x_coords = [5 + 10 * i for i in range(len(leaf_x))]
            
            # Get strengths in leaf order and normalize for display
            strength_vals = [strengths.get(head_keys[i], 0) for i in leaf_x]
            max_strength = max(abs(v) for v in strength_vals) if strength_vals else 1
            
            # Scale bars to fit in bottom portion of dendrogram
            y_max = threshold * bar_height  # bar_height controls max height (0.1-1.0)
            scaled_vals = [v / max_strength * y_max for v in strength_vals]
            
            # Color by GQA group if available (singletons use grey)
            if gqa_colors is not None:
                colors = []
                for i in leaf_x:
                    c = gqa_colors[head_keys[i]]
                    colors.append(c if c is not None else '#888888')
            else:
                colors = ['#888888'] * len(strength_vals)
            
            # Draw semi-transparent bars on the dendrogram
            ax1.bar(x_coords, scaled_vals, width=8, color=colors, alpha=0.3, zorder=0)
            
            # Add secondary y-axis on right for strength scale
            ax1_right = ax1.twinx()
            ax1_right.set_ylim(0, max_strength * (ax1.get_ylim()[1] / y_max))
            ax1_right.set_ylabel('Strength', fontsize=8, color='#666666')
            ax1_right.tick_params(axis='y', labelcolor='#666666', labelsize=7)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    print_cluster_summary(groups, metrics)
    return groups, metrics, dist_matrix, linkage_matrix


def print_cluster_summary(
    groups: Dict[int, List],  # List of Node objects or (layer, head) tuples
    metrics: Optional[Dict[int, Dict[str, float]]] = None,
    aw_dict: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
) -> None:
    """
    Print a summary of clustered attention heads.
    
    Args:
        groups: Dict mapping cluster_id -> list of Node objects or (layer, head) tuples
        metrics: Optional cluster metrics from cluster_heads
        aw_dict: Optional attention weights dict for computing statistics
    """
    n_groups = len(groups)
    n_heads = sum(len(v) for v in groups.values())
    
    # print('=' * 60)
    print(f'CLUSTERING SUMMARY: {n_groups} groups, {n_heads} heads')
    
    # Print overall metrics if available
    if metrics and '_overall' in metrics:
        m = metrics['_overall']
        print(f'  Cohesion (avg intra): {m["avg_cohesion"]:.3f}')
        print(f'  Separation (avg sep): {m["avg_separation"]:.3f}')
        print(f'  Sep/Cohesion ratio:   {m["sep_cohesion_ratio"]:.2f} (higher = better)')
    # print('=' * 60)
    
    def get_lh(head):
        """Extract (layer, head) from Node or tuple."""
        if hasattr(head, 'layer'):
            return head.layer, head.head
        return head
    
    for cluster_id in sorted(k for k in groups.keys() if k != '_overall'):
        heads = groups[cluster_id]
        head_str = ', '.join([f'{l}-{h}' for l, h in [get_lh(h) for h in heads]])
        
        # Show metrics if available
        metric_str = ''
        if metrics and cluster_id in metrics:
            m = metrics[cluster_id]
            # Handle both old format (mean/max) and new format (intra_mean/sep_min)
            if 'intra_mean' in m:
                metric_str = f' | intra={m["intra_mean"]:.3f}, sep={m["sep_min"]:.3f}'
            elif 'mean' in m:
                metric_str = f' | mean={m["mean"]:.3f}, max={m["max"]:.3f}'
        
        print(f'📦 GROUP {cluster_id} ({len(heads)} heads){metric_str}', end='. ')
        print(f'   Heads: {head_str}')
        
        if aw_dict:
            # Compute some statistics for this group
            for head in heads:
                layer, h = get_lh(head)
                if (layer, h) in aw_dict:
                    aw = aw_dict[(layer, h)]
                    # Self-attention (diagonal)
                    diag_attn = torch.diagonal(aw, dim1=-2, dim2=-1).mean().item()
                    # Token 0 attention
                    tok0_attn = aw[:, :, 0].mean().item()
                    print(f'   L{layer}H{h}: self-attn={diag_attn:.1%}, tok0={tok0_attn:.1%}')


def visualize_group_patterns(
    groups: Dict[int, List],  # List of Node objects or (layer, head) tuples
    aw_dict: Dict[Tuple[int, int], torch.Tensor],
    last_n_pos: int = 150,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Visualize average attention pattern for each cluster group.
    
    Args:
        groups: Dict mapping cluster_id -> list of Node objects or (layer, head) tuples
        aw_dict: Dict mapping (layer, head) -> attention weights tensor
        last_n_pos: Number of positions to show (from end)
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    def get_lh(head):
        """Extract (layer, head) from Node or tuple."""
        if hasattr(head, 'layer'):
            return head.layer, head.head
        return head
    
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4))
    
    if n_groups == 1:
        axes = [axes]
    
    for idx, cluster_id in enumerate(sorted(groups.keys())):
        ax = axes[idx]
        group_heads = groups[cluster_id]
        
        # Average attention pattern across heads in this group
        patterns = []
        for head in group_heads:
            l, h = get_lh(head)
            if (l, h) in aw_dict:
                patterns.append(aw_dict[(l, h)].mean(dim=0))
        if not patterns:
            continue
        avg_attn = torch.stack(patterns).mean(dim=0)
        
        # Show last N positions
        avg_attn_vis = avg_attn[:, -last_n_pos:].cpu().numpy()
        
        im = ax.imshow(avg_attn_vis, aspect='auto', cmap='Blues')
        head_str = ', '.join([f'L{l}H{h}' for l, h in [get_lh(x) for x in group_heads[:3]]])
        if len(group_heads) > 3:
            head_str += '...'
        ax.set_title(f'Group {cluster_id}\n({len(group_heads)} heads: {head_str})', fontsize=9)
        ax.set_xlabel(f'KV pos (last {last_n_pos})')
        ax.set_ylabel('Query pos')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Average Attention Pattern per Group', fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def merge_gqa_groups(g, model_config, verbose=True):
    """
    Merge attention heads in the same (layer, KV-group) into a single virtual node.
    
    If nodes in a GQA group don't share type/attn_pattern, picks the strongest node
    (max sum of outgoing edge scores) and discards the others.
    
    Args:
        g: Graph object with .nodes and .edges
        model_config: Model config with num_attention_heads, num_key_value_heads
        verbose: If True, print info about merges and picks
    
    Returns:
        New Graph object with merged virtual nodes (does not modify original g)
    """
    from collections import defaultdict
    from attribute import Graph, Node
    
    num_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    heads_per_group = num_heads // num_kv_heads
    
    # Create new graph without setting as current (preserves global g)
    vg = Graph(g.dataset_size, g.hidden_size, set_current=False)
    
    # Compute outgoing edge strength for each node: sum of |scores| to downstream
    node_strength = defaultdict(float)
    for (upstream, downstream), score in g.edges.items():
        node_strength[id(upstream)] += abs(score)
    
    # Find nodes with at least one edge (ingoing or outgoing)
    nodes_with_edges = set()
    for upstream, downstream in g.edges.keys():
        nodes_with_edges.add(id(upstream))
        nodes_with_edges.add(id(downstream))
    
    # Group attention nodes by (layer, kv_group), excluding isolated nodes
    attn_groups = defaultdict(list)  # (layer, kv_group) -> [node, ...]
    non_attn_nodes = []
    
    for node in g.nodes:
        if id(node) not in nodes_with_edges:
            continue  # Skip isolated nodes
        if node.is_attn():
            kv_group = node.head // heads_per_group
            attn_groups[(node.layer, kv_group)].append(node)
        else:
            non_attn_nodes.append(node)
    
    # Map original node -> virtual node (for edge remapping)
    # Discarded nodes won't be in node_map
    node_map = {}
    
    # Create virtual nodes for attention groups
    for (layer, kv_group), group_nodes in sorted(attn_groups.items()):
        # Check if nodes share type and attn_pattern
        types = set(n.type for n in group_nodes)
        patterns = set(n.attn_pattern for n in group_nodes)
        
        if len(types) > 1 or len(patterns) > 1:
            # Conflict: pick strongest node, discard others
            nodes_with_strength = [(n, node_strength[id(n)]) for n in group_nodes]
            nodes_with_strength.sort(key=lambda x: -x[1])  # Descending by strength
            
            picked_node, picked_strength = nodes_with_strength[0]
            discarded = nodes_with_strength[1:]
            
            if verbose:
                print(f"L{layer}-G{kv_group}: CONFLICT - pick H{picked_node.head} ({picked_node.type}, {picked_node.attn_pattern}) strength={picked_strength:.4f}")
                for n, s in discarded:
                    print(f"    discard H{n.head} ({n.type}, {n.attn_pattern}) strength={s:.4f}")
            
            # Only include the picked node
            group_nodes = [picked_node]
        
        node_type = group_nodes[0].type
        attn_pattern = group_nodes[0].attn_pattern
        heads = sorted([n.head for n in group_nodes])
        
        if verbose and len(group_nodes) > 1:
            total_strength = sum(node_strength[id(n)] for n in group_nodes)
            heads_str = ','.join(str(h) for h in heads)
            print(f"L{layer}-G{kv_group}: MERGE H[{heads_str}] ({node_type}, {attn_pattern}) total_strength={total_strength:.4f}")
        
        # Create virtual node with representative head (first in group for node_id stability)
        # Use kv_group as "head" for unique identification
        virtual_node = object.__new__(Node)  # Bypass Node.__new__ interning
        virtual_node._initialized = True
        virtual_node._frozen = False
        virtual_node.layer = layer
        virtual_node.head = kv_group  # Use kv_group as head for virtual node
        virtual_node.type = node_type
        virtual_node.attn_pattern = attn_pattern
        virtual_node.graph = vg
        virtual_node.heads = heads  # Store real head IDs
        virtual_node.span = None
        virtual_node.src_span = None
        virtual_node.dataset_size = g.dataset_size
        virtual_node.hidden_size = g.hidden_size
        virtual_node.pos_ids = None
        virtual_node.output_grad = None
        virtual_node.grad = None
        virtual_node.dtype = torch.float16
        virtual_node.device = 'cpu'
        
        vg.nodes.append(virtual_node)
        for orig_node in group_nodes:
            node_map[id(orig_node)] = virtual_node
    
    # Copy non-attention nodes (create new objects to avoid sharing)
    for orig_node in non_attn_nodes:
        new_node = object.__new__(Node)  # Bypass Node.__new__ interning
        new_node._initialized = True
        new_node._frozen = False
        new_node.layer = orig_node.layer
        new_node.head = orig_node.head
        new_node.type = orig_node.type
        new_node.attn_pattern = orig_node.attn_pattern
        new_node.graph = vg
        new_node.heads = None  # Not a virtual node
        new_node.span = orig_node.span
        new_node.src_span = orig_node.src_span
        new_node.dataset_size = g.dataset_size
        new_node.hidden_size = g.hidden_size
        new_node.pos_ids = orig_node.pos_ids
        new_node.output_grad = None
        new_node.grad = None
        new_node.dtype = orig_node.dtype
        new_node.device = orig_node.device
        
        vg.nodes.append(new_node)
        node_map[id(orig_node)] = new_node
    
    # Merge edges: sum scores for edges pointing to/from same virtual node
    # Skip edges involving discarded nodes (not in node_map)
    merged_edges = defaultdict(float)
    for (upstream, downstream), score in g.edges.items():
        if id(upstream) not in node_map or id(downstream) not in node_map:
            continue  # Skip edges involving discarded nodes
        v_upstream = node_map[id(upstream)]
        v_downstream = node_map[id(downstream)]
        merged_edges[(v_upstream, v_downstream)] += score
    
    # Add merged edges to vg
    for (v_upstream, v_downstream), score in merged_edges.items():
        vg.edges[(v_upstream, v_downstream)] = score
        v_upstream._frozen = True
        v_downstream._frozen = True
    
    return vg


def minimize_crossings(g, max_iterations=20):
    """
    Reorder nodes within each layer to minimize edge crossings using barycenter heuristic.
    
    Args:
        g: Graph object with .nodes and .edges
        max_iterations: Maximum number of up/down sweep iterations
    
    Returns:
        Dict mapping node id() -> x_position (float)
    """
    from collections import defaultdict
    
    # Group nodes by layer
    nodes_by_layer = defaultdict(list)
    for node in g.nodes:
        nodes_by_layer[node.layer].append(node)
    
    if not nodes_by_layer:
        return {}
    
    # Build adjacency: for each node, list of (neighbor, layer_diff)
    # In attribution graph: edge (upstream, downstream) means upstream -> downstream
    # upstream is in lower layer, downstream in higher layer
    neighbors_above = defaultdict(list)  # node_id -> [nodes in higher layers]
    neighbors_below = defaultdict(list)  # node_id -> [nodes in lower layers]
    
    for (upstream, downstream), score in g.edges.items():
        # upstream.layer < downstream.layer (typically)
        neighbors_above[id(upstream)].append(downstream)
        neighbors_below[id(downstream)].append(upstream)
    
    # Initialize x-positions: spread evenly within each layer
    x_pos = {}
    for layer, nodes in nodes_by_layer.items():
        for i, node in enumerate(nodes):
            x_pos[id(node)] = float(i)
    
    # Sort layers
    layers = sorted(nodes_by_layer.keys())
    
    def barycenter(node, neighbor_list):
        """Compute barycenter (mean x) of neighbors."""
        neighbors = neighbor_list[id(node)]
        if not neighbors:
            return x_pos[id(node)]  # Keep current position if no neighbors
        return sum(x_pos[id(n)] for n in neighbors) / len(neighbors)
    
    def reorder_layer(layer_nodes, neighbor_list):
        """Reorder nodes in a layer based on barycenter of neighbors."""
        # Compute barycenter for each node
        bary = [(node, barycenter(node, neighbor_list)) for node in layer_nodes]
        # Sort by barycenter
        bary.sort(key=lambda x: x[1])
        # Assign new integer positions
        for i, (node, _) in enumerate(bary):
            x_pos[id(node)] = float(i)
    
    def count_crossings():
        """Count total edge crossings (for convergence check)."""
        total = 0
        # For each pair of adjacent layers
        for i in range(len(layers) - 1):
            lower_layer = layers[i]
            upper_layer = layers[i + 1]
            
            # Get edges between these layers
            edges = []
            for (up, down), score in g.edges.items():
                if up.layer == lower_layer and down.layer == upper_layer:
                    edges.append((x_pos[id(up)], x_pos[id(down)]))
            
            # Count crossings: two edges cross if their x-orders are reversed
            for j in range(len(edges)):
                for k in range(j + 1, len(edges)):
                    x1_low, x1_high = edges[j]
                    x2_low, x2_high = edges[k]
                    # Crossing if (x1_low < x2_low) != (x1_high < x2_high)
                    if (x1_low - x2_low) * (x1_high - x2_high) < 0:
                        total += 1
        return total
    
    # Iterate: sweep down then up
    prev_crossings = count_crossings()
    
    for iteration in range(max_iterations):
        # Sweep down (top to bottom): use neighbors above
        for layer in reversed(layers[:-1]):  # Skip top layer
            reorder_layer(nodes_by_layer[layer], neighbors_above)
        
        # Sweep up (bottom to top): use neighbors below
        for layer in layers[1:]:  # Skip bottom layer
            reorder_layer(nodes_by_layer[layer], neighbors_below)
        
        # Check convergence
        crossings = count_crossings()
        if crossings >= prev_crossings:
            break  # No improvement
        prev_crossings = crossings
    
    # After optimization: spread nodes evenly within each layer,
    # centered around their barycenter to reduce edge length
    for layer, nodes in nodes_by_layer.items():
        # Sort nodes by their current x_pos (which encodes the optimized order)
        sorted_nodes = sorted(nodes, key=lambda n: x_pos[id(n)])
        n = len(sorted_nodes)
        
        if n == 1:
            # Single node: position at barycenter of neighbors
            node = sorted_nodes[0]
            all_neighbors = neighbors_above[id(node)] + neighbors_below[id(node)]
            if all_neighbors:
                center = sum(x_pos[id(nb)] for nb in all_neighbors) / len(all_neighbors)
            else:
                center = 0
            x_pos[id(node)] = center
        else:
            # Multiple nodes: compute layer center from neighbor barycenters
            centers = []
            for node in sorted_nodes:
                all_neighbors = neighbors_above[id(node)] + neighbors_below[id(node)]
                if all_neighbors:
                    centers.append(sum(x_pos[id(nb)] for nb in all_neighbors) / len(all_neighbors))
            layer_center = sum(centers) / len(centers) if centers else 0
            
            # Spread evenly around the layer center with unit spacing
            start = layer_center - (n - 1) / 2
            for i, node in enumerate(sorted_nodes):
                x_pos[id(node)] = start + i
    
    return x_pos


def visualize_graph(
    g,  # Graph object
    model_config,  # model.config for GQA info
    height: str = '800px',
    width: str = '100%',
    node_size: int = 6,
    save_path: str = None,  # Defaults to os.getcwd()/graph.html
    open_browser: bool = True,
    reduce_crossings: bool = True,  # Use barycenter heuristic to minimize edge crossings
):
    """
    Visualize attribution graph with Pyvis.
    
    Supports virtual nodes from merge_gqa_groups() - node size reflects
    number of merged heads (via node.heads attribute).
    
    Args:
        g: Graph object with .nodes and .edges
        model_config: Model config with num_attention_heads, num_key_value_heads
        height: Figure height (CSS string)
        width: Figure width (CSS string)
        node_size: Base node size
        save_path: Path to save HTML file
        open_browser: If True, open in browser tab
        reduce_crossings: If True, reorder nodes to minimize edge crossings
    
    Returns:
        Path to saved HTML file
    """
    import os
    from pyvis.network import Network
    
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'graph.html')
    
    # GQA config
    num_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    heads_per_group = num_heads // num_kv_heads
    
    # Node type colors
    type_colors = {
        'attn_v': '#3498db',   # Blue
        'attn_q': '#9b59b6',   # Purple
        'mlp_i': '#2ecc71',    # Green
        'mlp_g': '#f1c40f',    # Yellow
        'lm_head': '#e74c3c',  # Red
    }
    
    # Create stable node ID from (layer, head, type) - not Python id()
    def node_id(node):
        return f"{node.layer}-{node.head}-{node.type}"
    
    # Find nodes with at least one edge (ingoing or outgoing)
    nodes_with_edges = set()
    for upstream, downstream in g.edges.keys():
        nodes_with_edges.add(id(upstream))
        nodes_with_edges.add(id(downstream))
    
    # Collect nodes by layer for layout calculation, excluding isolated nodes
    nodes_by_layer = {}
    for node in g.nodes:
        if id(node) not in nodes_with_edges:
            continue  # Skip isolated nodes
        nodes_by_layer.setdefault(node.layer, []).append(node)
    
    if not nodes_by_layer:
        print("Graph has no nodes with edges")
        return None
    
    # Calculate tight layout bounds
    min_layer = min(nodes_by_layer.keys())
    max_layer = max(nodes_by_layer.keys())
    layer_range = max(1, max_layer - min_layer)
    
    # Create network with fixed layout (no physics)
    net = Network(height=height, width=width, directed=True, notebook=False,
                  cdn_resources='in_line')
    net.toggle_physics(False)
    
    # Flatten filtered nodes for iteration
    filtered_nodes = [n for nodes in nodes_by_layer.values() for n in nodes]
    
    # Calculate x positions
    if reduce_crossings:
        # Use barycenter heuristic to minimize edge crossings
        x_pos = minimize_crossings(g)
        def get_x_position(node):
            return x_pos.get(id(node), 0)
    else:
        # Use GQA-based positioning
        def get_x_position(node):
            """Calculate x position based on node type and head number."""
            if node.is_attn():
                heads = getattr(node, 'heads', None)
                if heads:
                    # Virtual node: head is kv_group, position at center of group
                    kv_group = node.head
                    group_spacing = heads_per_group + 2
                    return kv_group * group_spacing + heads_per_group // 2
                else:
                    # Real node: group by KV group
                    head = node.head
                    kv_group = head // heads_per_group
                    pos_in_group = head % heads_per_group
                    group_spacing = heads_per_group + 2
                    return kv_group * group_spacing + pos_in_group
            elif node.is_mlp():
                # MLP nodes on the right side
                return num_kv_heads * (heads_per_group + 2) + 5
            else:  # lm_head
                return num_kv_heads * (heads_per_group + 2) // 2  # Center
    
    # Find x bounds for tight layout
    all_x = [get_x_position(n) for n in filtered_nodes]
    min_x, max_x = min(all_x), max(all_x)
    x_range = max(1, max_x - min_x)
    
    # Scale factors for display
    x_scale = 800 / x_range if x_range > 0 else 1
    y_scale = 400 / layer_range if layer_range > 0 else 1
    
    # Add nodes
    for node in filtered_nodes:
        # Position
        raw_x = get_x_position(node)
        x = (raw_x - min_x) * x_scale
        y = -(node.layer - min_layer) * y_scale  # Negative so higher layers are at top
        
        # Color
        color = type_colors.get(node.type, '#95a5a6')
        
        # Node size: scale by number of merged heads (for virtual nodes)
        heads = getattr(node, 'heads', None)
        num_merged = len(heads) if heads else 1
        size = node_size * (1 + 0.5 * (num_merged - 1))  # 1 head -> 1x, 2 heads -> 1.5x, etc.
        
        # Hover title
        if node.is_attn():
            if heads:
                # Virtual node: show all merged heads
                heads_str = ','.join(str(h) for h in heads)
                title = f"{node.layer}-[{heads_str}] {node.type}"
            else:
                title = f"{node.layer}-{node.head} {node.type}"
            if node.attn_pattern:
                title += f"\n{node.attn_pattern}"
        elif node.is_mlp():
            title = f"{node.layer}-mlp"
            if node.type != 'mlp_i':
                title = f"{node.layer}-{node.type.split('_')[1]}"
        else:
            title = "lm_head"
        
        net.add_node(
            node_id(node),
            label=" ",  # Space to hide label (empty shows node ID)
            title=title,
            x=x,
            y=y,
            color=color,
            size=size,
            physics=False,
        )
    
    # Add edges - normalize widths relative to max score
    max_score = max(abs(s) for s in g.edges.values()) if g.edges else 1.0
    min_width, max_width = 0.5, 6.0
    
    for (upstream, downstream), score in g.edges.items():
        # Width proportional to score, scaled to [min_width, max_width]
        normalized = abs(score) / max_score  # 0 to 1
        width = min_width + normalized * (max_width - min_width)
        
        # Color: green for positive, red for negative
        color = '#27ae60' if score > 0 else '#c0392b'
        
        # Hover shows score
        title = f"score: {score:.4f}"
        
        net.add_edge(
            node_id(upstream),
            node_id(downstream),
            width=width,
            color=color,
            title=title,
            arrows='to',
        )
    
    # Configure options for fixed layout
    net.set_options('''
    {
        "nodes": {
            "font": {"size": 10}
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"scaleFactor": 0.5}}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true
        },
        "physics": {
            "enabled": false
        }
    }
    ''')
    
    # Save HTML file
    net.save_graph(save_path)
    print(f"Graph saved to {save_path}")
    
    if open_browser:
        # For Remote SSH: serve via HTTP (Cursor auto-forwards ports)
        import threading
        import http.server
        import socketserver
        
        directory = os.path.dirname(os.path.abspath(save_path))
        filename = os.path.basename(save_path)
        port = 8765
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)
            def log_message(self, format, *args):
                pass  # Suppress logs
        
        # Check if server already running
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', port))
            s.close()
            # Port available, start server
            server = socketserver.TCPServer(('', port), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            print(f"Server started on port {port}")
        except OSError:
            pass  # Server likely already running
        
        url = f"http://localhost:{port}/{filename}"
        print(f"Open in browser: {url}")
    
    return save_path


def iter_attribution_tnodes(root):
    """Depth-first walk over pptree attribution nodes starting from root."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        children = getattr(node, 'children', None) or []
        stack.extend(children)


def collect_tree_attn_heads(root):
    """Collect (layer, head) pairs present on the attribution tree (skip lm_head)."""
    heads = set()
    for tnode in iter_attribution_tnodes(root):
        data = getattr(tnode, 'data', None)
        if data is None:
            continue
        for n in getattr(data, 'nodes', None) or []:
            if getattr(n, 'head', None) is None:
                continue
            if hasattr(n, 'is_lm_head') and n.is_lm_head():
                continue
            heads.add((int(n.layer), int(n.head)))
    return heads


def _model_n_layers_n_heads(model):
    if isinstance(model, transformer_lens.HookedTransformer):
        return int(model.cfg.n_layers), int(model.cfg.n_heads)
    return int(model.config.num_hidden_layers), int(model.config.num_attention_heads)


def _patch_plotly_figurewidget_uid_bug():
    """Skip trace-delta sync messages missing ``uid`` (known Plotly FigureWidget bug)."""
    try:
        from plotly.basewidget import BaseFigureWidget
    except Exception:
        return
    if getattr(BaseFigureWidget, '_btom_uid_patch', False):
        return
    _orig = BaseFigureWidget._handler_js2py_traceDeltas

    def _safe_handler(self, change):
        msg_data = change.get('new')
        if not msg_data:
            self._js2py_traceDeltas = None
            return
        deltas = msg_data.get('trace_deltas') or []
        fixed = [d for d in deltas if isinstance(d, dict) and 'uid' in d]
        if len(fixed) != len(deltas):
            msg_data = dict(msg_data)
            msg_data['trace_deltas'] = fixed
            change = {
                'name': change.get('name'),
                'old': change.get('old'),
                'new': msg_data,
                'owner': change.get('owner'),
                'type': change.get('type'),
            }
        if not fixed:
            self._js2py_traceDeltas = None
            return
        return _orig(self, change)

    BaseFigureWidget._handler_js2py_traceDeltas = _safe_handler
    BaseFigureWidget._btom_uid_patch = True


def visualize_model_heads(
    root,
    model,
    results,
    sample=None,
    downstreams=None,
    start=None,
    tree_color=None,
    marker_size=11,
):
    """Interactive L×H head map: click a circle to render show_attn on the right.

    - Default circles are white.
    - Heads that appear on the attribution tree share one random (or given) color.
    - ``sample`` is optional (defaults to ``results[0]``); change via the dropdown.

    Args:
        root: Attribution tree root (pptree node with ``.data.nodes``).
        model: HookedTransformer or HF model used by ``show_attn``.
        results: List of result objects (same as ``_results``).
        sample: Optional initial sample; defaults to ``results[0]``.
        downstreams: Passed to ``show_attn``; defaults to ``root.data.nodes``.
        start: Token window start for ``show_attn``; defaults to
            ``sample.index_map[0]['start']`` when available.
        tree_color: CSS/plotly color for on-tree heads; random if None.
        marker_size: Scatter marker size.

    Returns:
        ipywidgets HBox UI. Assign with ``ui = visualize_model_heads(...)`` so the
        notebook cell does not render a second orphan figure.
    """
    import random
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import plotly.graph_objects as go
    from circuitsvis.tokens import colored_tokens_multi

    _patch_plotly_figurewidget_uid_bug()

    n_layers, n_heads = _model_n_layers_n_heads(model)
    on_tree = collect_tree_attn_heads(root)

    if tree_color is None:
        tree_color = 'rgb({},{},{})'.format(
            random.randint(40, 200),
            random.randint(40, 200),
            random.randint(40, 200),
        )

    # Numeric colors + colorscale avoids some FigureWidget color-list sync issues.
    xs, ys, color_vals, texts, custom = [], [], [], [], []
    for l in range(n_layers):
        for h in range(n_heads):
            xs.append(h)
            ys.append(l)
            marked = (l, h) in on_tree
            color_vals.append(1 if marked else 0)
            texts.append(f'L{l} H{h}' + (' · on tree' if marked else ''))
            custom.append([l, h])

    scatter = go.Scatter(
        x=xs,
        y=ys,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color_vals,
            cmin=0,
            cmax=1,
            colorscale=[[0, 'white'], [1, tree_color]],
            line=dict(width=1, color='#555555'),
            symbol='circle',
            showscale=False,
        ),
        text=texts,
        hoverinfo='text',
        customdata=custom,
        uid='btom-head-grid',
    )
    fig = go.FigureWidget(data=[scatter])
    fig.update_layout(
        title=f'Heads {n_layers}×{n_heads}  |  on-tree: {tree_color}  |  {len(on_tree)} marked',
        xaxis=dict(
            title='head',
            dtick=max(1, n_heads // 8),
            range=[-0.8, n_heads - 0.2],
            zeroline=False,
        ),
        yaxis=dict(
            title='layer',
            dtick=max(1, n_layers // 8),
            range=[-0.8, n_layers - 0.2],
            zeroline=False,
        ),
        width=max(520, int(n_heads * 16 + 80)),
        height=max(420, int(n_layers * 16 + 80)),
        plot_bgcolor='#d9d9d9',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=48, b=40),
        showlegend=False,
    )

    if sample is None:
        sample = results[0]
    try:
        init_idx = list(results).index(sample)
    except ValueError:
        init_idx = 0
        sample = results[0]

    state = {'sample': sample, 'sel': None}

    sample_dd = widgets.Dropdown(
        options=[(f'sample[{i}]', i) for i in range(len(results))],
        value=init_idx,
        description='sample:',
        layout=widgets.Layout(width='220px'),
    )
    status = widgets.HTML(value='<i>Click a circle to inspect attention</i>')
    attn_out = widgets.Output(
        layout=widgets.Layout(max_height='720px', overflow_y='auto', min_width='480px')
    )

    def _resolve_start(s):
        if start is not None:
            return start
        try:
            return s.index_map[0]['start']
        except Exception:
            return None

    def _resolve_downstreams():
        if downstreams is not None:
            return downstreams
        return root.data.nodes

    def render_head(l, h):
        state['sel'] = (l, h)
        s = state['sample']
        status.value = f'<b>Selected</b> layer={l}, head={h}'
        with attn_out:
            clear_output(wait=True)
            display(colored_tokens_multi(*show_attn(
                s, model, int(l), int(h),
                downstreams=_resolve_downstreams(),
                start=_resolve_start(s),
            )))

    def on_click(trace, points, selector):
        if not points.point_inds:
            return
        i = points.point_inds[0]
        l, h = trace.customdata[i]
        render_head(int(l), int(h))

    # Bind after the trace is owned by FigureWidget (callbacks are not copied on add).
    fig.data[0].on_click(on_click)

    def on_sample_change(change):
        state['sample'] = results[change['new']]
        if state['sel'] is not None:
            render_head(*state['sel'])

    sample_dd.observe(on_sample_change, names='value')

    left = widgets.VBox([sample_dd, fig])
    right = widgets.VBox([status, attn_out])
    # Return the UI only — do NOT also display(fig)/display(ui).
    # Returning FigureWidget used to make the notebook cell render a second empty grid.
    ui = widgets.HBox([left, right])
    return ui
