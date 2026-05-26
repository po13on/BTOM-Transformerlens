# Model Server API Documentation

A FastAPI server for LLM interpretability research. Manages model lifecycle, caches activations, and computes gradients for Integrated Gradients (IG) attribution.

## Quick Start

```bash
# Start server
cd /home/xd/projects/BARC
conda activate tune
python model_server.py --port 8000 --device cuda:0
```

```python
from model_client import ModelClient
client = ModelClient("http://localhost:8000")

# Load model
client.load_model("/data0/modelscope/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4")

# Forward and cache activations
client.forward("my_dataset", sample_id=0, prompt="Hello world")

# Get tensors
attn = client.get_attn_weights("my_dataset", 0, layer=60, head=10)
```

---

## API Reference

### Server Management

| Method | Description |
|--------|-------------|
| `status()` → `dict` | Server status: model_loaded, device, cached_datasets |
| `load_model(path, device="cuda:0", quantization="gptq")` → `dict` | Load model |
| `empty_cuda_cache()` → `dict` | Free GPU memory |

### Forward Pass & Caching

| Method | Description |
|--------|-------------|
| `forward(dataset_id, sample_id, prompt)` | Run forward, cache activations |
| `batch_forward(dataset_id, prompts, show_progress=True)` | Forward multiple samples |
| `forward_results(dataset_id, results, prompt_attr="prompt")` | Forward Result objects |

**Cache Structure**: `dataset_id` → `sample_id` → activations (hidden_states, attn_weights, etc.)

### Get Cached Tensors

| Method | Returns |
|--------|---------|
| `get_hidden_states(dataset_id, sample_id, layer, pos_ids=None)` | `[1, seq_len, hidden_dim]` |
| `get_attn_weights(dataset_id, sample_id, layer, head=None, pos_ids=None)` | `[1, (heads), q_len, kv_len]` |
| `get_head_output(dataset_id, sample_id, layer, head, pos_ids=None)` | `[seq_len, hidden_dim]` |
| `get_attn_output(dataset_id, sample_id, layer, pos_ids=None)` | `[1, seq_len, hidden_dim]` |

All tensor methods accept `device='cpu'` (default) or `'cuda:X'`.

### Cache Management

| Method | Description |
|--------|-------------|
| `list_cached_samples(dataset_id)` → `{"sample_ids": [...], "count": N}` | List samples |
| `clear_cache(dataset_id)` | Delete dataset cache |
| `clear_all_cache()` | Delete all caches |

---

## Gradient Computation (IG Attribution)

### `compute_node_gradient`

Compute IG gradient for a single node in the computation graph.

```python
grad = client.compute_node_gradient(
    dataset_id="puzzles",
    sample_id=0,              # None = average over all samples
    node_type="attn_v",       # See node types below
    layer=60,
    head=10,                  # Required for attn_* nodes
    pos_ids=[100, 101],       # Query positions
    output_grad=downstream_grad,  # From downstream node (None for lm_head)
    steps=0,                  # IG steps: None=4, 0=plain backward
)
# Returns: [1, len(pos_ids), hidden_dim]
```

**Node Types**:
| Type | Description | Required Args |
|------|-------------|---------------|
| `lm_head` | Language model head (starting point) | `labels`, `candidate_ids` |
| `attn_q` | Query projection | `head` |
| `attn_k` | Key projection | `head`, `src_pos_ids` (optional) |
| `attn_v` | Value projection | `head`, `src_pos_ids` (optional) |
| `mlp_i` | MLP up_proj path | - |
| `mlp_g` | MLP gate_proj path | - |

**IG Steps**:
- `steps=None` (default 4): Integrated Gradients with interpolation
- `steps=0`: Plain backward (exact for linear operations like attention)

### `attribute_residual`

Attribute gradients to all attention heads and MLPs in the residual stream.

```python
attr = client.attribute_residual(
    dataset_id="puzzles",
    sample_id=0,
    pos_ids=[100, 101],
    downstream_grads=[grad1, grad2],    # List of gradient tensors
    downstream_layers=[60.0, 60.5],     # 60.0=attn, 60.5=MLP
    from_layer=0,
    to_layer=60,
)
# Returns: [num_layers, num_heads+1, num_downstream_nodes]
# Last column (index -1) in dim 1 is MLP attribution
```

**Layer Convention**:
- `layer` (int/float): Attention at layer `L` → `L.0`, MLP at layer `L` → `L.5`
- Example: `downstream_layers=[60.0, 60.5]` means attn@60 and mlp@60

---

## Full Attribution Example

```python
from model_client import ModelClient
client = ModelClient("http://localhost:8000")

# Setup
client.load_model("/path/to/model")
client.forward("data", 0, "The capital of France is")

# Step 1: Compute lm_head gradient (starting point)
pos_ids = [10]  # Position of interest
grad_lm = client.compute_node_gradient(
    dataset_id="data",
    sample_id=0,
    node_type="lm_head",
    layer=79,  # num_layers - 1
    pos_ids=pos_ids,
    labels=[1234],  # Target token ID
    candidate_ids=list(range(1000)),  # Valid token IDs
    steps=0,
)

# Step 2: Compute gradient for attention head
grad_attn = client.compute_node_gradient(
    dataset_id="data",
    sample_id=0,
    node_type="attn_v",
    layer=60,
    head=10,
    pos_ids=pos_ids,
    output_grad=grad_lm,
    steps=0,
)

# Step 3: Attribute to residual stream
attr = client.attribute_residual(
    dataset_id="data",
    sample_id=0,
    pos_ids=pos_ids,
    downstream_grads=[grad_attn],
    downstream_layers=[60.0],
    from_layer=0,
    to_layer=60,
)
# attr[layer, head, 0] = attribution score for head at layer
# attr[layer, -1, 0] = attribution score for MLP at layer
```

---

## Integration with min_arc.py

Remote versions of local functions for seamless server integration:

```python
from min_arc import Node, rattribute_step, rattribute_residual
from model_client import ModelClient

client = ModelClient("http://localhost:8000")

# Forward samples to server
client.forward_results("puzzles", results)

# Remote attribution step (replaces attribute_step)
for r in results:
    attr = rattribute_step(r, client, "puzzles", nodes)
    # attr shape: [num_layers, num_heads+1, num_nodes]
```

| Local Function | Remote Version | Description |
|----------------|----------------|-------------|
| `node.backward(r, model)` | `node.rbackward(r, client, dataset_id)` | Compute node gradient |
| `attribute_residual(...)` | `rattribute_residual(...)` | Residual stream attribution |
| `attribute_step(r, model, nodes)` | `rattribute_step(r, client, dataset_id, nodes)` | Full attribution step |

---

## Notes

- **Tensor device**: All tensor-returning methods accept `device='cpu'` (default) or `'cuda:X'`
- **GPTQ support**: Server automatically dequantizes GPTQ weights for gradient computation
- **Memory**: Use `empty_cuda_cache()` and `clear_cache()` to manage memory
- **Server log**: Check `server.log` for debugging

