from collections import OrderedDict
import torch


def extract_gptq_weight(layer, device=None):
    """
    Extract effective weight matrix from GPTQ quantized layer by probing.
    This works by passing identity matrices through the layer.
    
    Returns:
        weight: [out_features, in_features] tensor matching nn.Linear convention
    """
    if device is None:
        device = layer.scales.device
    
    in_features = layer.infeatures
    out_features = layer.outfeatures
    dtype = layer.scales.dtype
    
    # Probe with identity matrix in chunks to save memory
    chunk_size = min(1024, in_features)
    weights_extracted = []
    
    for i in range(0, in_features, chunk_size):
        end = min(i + chunk_size, in_features)
        identity_chunk = torch.zeros(end - i, in_features, device=device, dtype=dtype)
        identity_chunk[:, i:end] = torch.eye(end - i, device=device, dtype=dtype)
        
        with torch.no_grad():
            output = layer(identity_chunk.unsqueeze(0))  # [1, chunk_size, out_features]
            if layer.bias is not None:
                output = output - layer.bias
            weights_extracted.append(output.squeeze(0))  # [chunk_size, out_features]
    
    # [in_features, out_features] -> [out_features, in_features]
    effective_weight = torch.cat(weights_extracted, dim=0).T
    return effective_weight


# LRU Cache for extracted GPTQ weights with hybrid strategy
class GPTQWeightCache:
    """
    Hybrid LRU cache for extracted GPTQ weights.
    
    Benchmark results (A800 80GB, Qwen2.5-72B):
    - Extraction time scales with INFEATURES (not weight size)
    - Small infeatures (≤8192): extract ~2-20 ms, faster than CPU transfer
    - Large infeatures (>8192): extract ~70 ms, slower than CPU transfer (~32 ms)
    
    Strategy:
    - GPU LRU cache: Fast access (~0.01 ms) for frequently used weights
    - For cache miss:
      - Small infeatures: extract on-fly (faster)
      - Large infeatures: load from CPU pinned cache (faster)
    """
    
    # Threshold: if infeatures > this, use CPU cache instead of extract
    LARGE_INFEATURES_THRESHOLD = 8192
    
    def __init__(self, max_gpu_mb=4000):
        """
        Args:
            max_gpu_mb: Max GPU memory for this cache
        """
        self.max_gpu_bytes = max_gpu_mb * 1024 * 1024
        self.gpu_cache = OrderedDict()  # layer_key -> (weight, size_bytes)
        self.cpu_cache = {}  # layer_key -> weight (pinned) for large-infeatures projections
        self.current_gpu_bytes = 0
        
        # Stats
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.extracts = 0
    
    def get(self, layer, layer_key):
        """
        Get weight using hybrid strategy:
        1. Check GPU cache (instant)
        2. For cache miss:
           - Small infeatures: extract on-fly
           - Large infeatures: load from CPU cache (or extract once and cache)
        """
        # Check GPU cache first
        if layer_key in self.gpu_cache:
            self.gpu_hits += 1
            self.gpu_cache.move_to_end(layer_key)
            return self.gpu_cache[layer_key][0]
        
        # GPU miss - strategy depends on infeatures
        is_large = layer.infeatures > self.LARGE_INFEATURES_THRESHOLD
        
        if is_large:
            # Large infeatures: use CPU cache
            if layer_key in self.cpu_cache:
                self.cpu_hits += 1
                weight = self.cpu_cache[layer_key].to(layer.scales.device, non_blocking=True)
                torch.cuda.synchronize()
            else:
                # First access: extract and store in CPU cache
                self.extracts += 1
                weight = extract_gptq_weight(layer)
                self.cpu_cache[layer_key] = weight.to('cpu').pin_memory()
        else:
            # Small infeatures: extract on-fly (faster than CPU transfer)
            self.extracts += 1
            weight = extract_gptq_weight(layer)
        
        # Add to GPU cache
        self._add_to_gpu_cache(layer_key, weight)
        return weight
    
    def _add_to_gpu_cache(self, key, weight):
        """Add weight to GPU cache, evicting LRU entries if necessary."""
        size_bytes = weight.numel() * weight.element_size()
        
        # Evict LRU entries until we have space
        while self.current_gpu_bytes + size_bytes > self.max_gpu_bytes and self.gpu_cache:
            evicted_key, (evicted_weight, evicted_size) = self.gpu_cache.popitem(last=False)
            self.current_gpu_bytes -= evicted_size
            del evicted_weight
        
        # Add to GPU cache
        self.gpu_cache[key] = (weight, size_bytes)
        self.current_gpu_bytes += size_bytes
    
    def clear(self):
        """Clear all caches."""
        self.gpu_cache.clear()
        self.cpu_cache.clear()
        self.current_gpu_bytes = 0
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.extracts = 0
        torch.cuda.empty_cache()
    
    def stats(self):
        """Return cache statistics."""
        total = self.gpu_hits + self.cpu_hits + self.extracts
        return {
            'gpu_hits': self.gpu_hits,
            'cpu_hits': self.cpu_hits,
            'extracts': self.extracts,
            'hit_rate': f'{(self.gpu_hits + self.cpu_hits) / total:.1%}' if total > 0 else '0%',
            'gpu_cache_mb': self.current_gpu_bytes / 1024 / 1024,
            'gpu_entries': len(self.gpu_cache),
            'cpu_entries': len(self.cpu_cache),
        }


class DualGPTQWeightCache:
    """
    Manages separate caches for attention and MLP with different size limits.
    
    Attention weights are smaller (~150 MB/layer for v+o), MLP are larger (~1.5 GB/layer).
    down_proj has large infeatures (29696) so uses CPU cache.
    """
    
    def __init__(self, attn_max_gpu_mb=4000, mlp_max_gpu_mb=2000):
        """
        Args:
            attn_max_gpu_mb: Max GPU memory for attention cache (v_proj, o_proj, etc.)
            mlp_max_gpu_mb: Max GPU memory for MLP cache (gate_proj, up_proj only; down_proj uses CPU)
        """
        self.attn_cache = GPTQWeightCache(max_gpu_mb=attn_max_gpu_mb)
        self.mlp_cache = GPTQWeightCache(max_gpu_mb=mlp_max_gpu_mb)
    
    def get_attn(self, layer, layer_key):
        """Get attention projection weight."""
        return self.attn_cache.get(layer, layer_key)
    
    def get_mlp(self, layer, layer_key):
        """Get MLP projection weight."""
        return self.mlp_cache.get(layer, layer_key)
    
    def clear(self):
        """Clear all caches."""
        self.attn_cache.clear()
        self.mlp_cache.clear()
    
    def stats(self):
        """Return combined cache statistics."""
        attn_stats = self.attn_cache.stats()
        mlp_stats = self.mlp_cache.stats()
        return {
            'attn': attn_stats,
            'mlp': mlp_stats,
            'total_gpu_mb': attn_stats['gpu_cache_mb'] + mlp_stats['gpu_cache_mb'],
        }


# Global dual cache instance
_gptq_weight_cache = DualGPTQWeightCache(attn_max_gpu_mb=4000, mlp_max_gpu_mb=2000)


def get_gptq_weight_cached(layer, layer_key, cache_type='attn'):
    """
    Get extracted GPTQ weight using appropriate cache.
    
    Args:
        layer: The QuantLinear layer
        layer_key: Cache key (e.g., 'layer0_v_proj')
        cache_type: 'attn' or 'mlp'
    """
    if cache_type == 'attn':
        return _gptq_weight_cache.get_attn(layer, layer_key)
    else:
        return _gptq_weight_cache.get_mlp(layer, layer_key)


def clear_gptq_weight_cache():
    """Clear all GPTQ weight caches."""
    _gptq_weight_cache.clear()


def get_gptq_cache_stats():
    """Get cache statistics."""
    return _gptq_weight_cache.stats()


def configure_gptq_cache(attn_max_gpu_mb=4000, mlp_max_gpu_mb=2000):
    """
    Configure cache sizes. Call before using the cache.
    
    Args:
        attn_max_gpu_mb: Max GPU memory for attention weights
        mlp_max_gpu_mb: Max GPU memory for MLP weights (gate/up only)
    """
    global _gptq_weight_cache
    _gptq_weight_cache = DualGPTQWeightCache(
        attn_max_gpu_mb=attn_max_gpu_mb,
        mlp_max_gpu_mb=mlp_max_gpu_mb
    )