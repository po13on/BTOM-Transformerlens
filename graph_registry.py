# Graph registry - excluded from autoreload to persist state
# Usage: %aimport -graph_registry  (in notebook, before other imports)

_current_graph = None

def set_current_graph(graph):
    global _current_graph
    _current_graph = graph

def get_current_graph():
    return _current_graph

# Attention weights cache - avoids SDPA non-determinism
_attn_weights_cache = {}

def get_attn_weights_cache():
    return _attn_weights_cache

def clear_attn_weights_cache():
    _attn_weights_cache.clear()

def get_attn_weights_cache_info():
    total_bytes = sum(t.numel() * t.element_size() for t in _attn_weights_cache.values())
    return {
        'entries': len(_attn_weights_cache),
        'memory_mb': total_bytes / (1024 * 1024),
        'keys': list(_attn_weights_cache.keys())
    }

