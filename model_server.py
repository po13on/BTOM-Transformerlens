"""
Model Server for shared model state and activation caching.

Usage:
    # Start server
    python model_server.py --port 8000 --device cuda:0
    
    # Or with uvicorn directly
    uvicorn model_server:app --host 0.0.0.0 --port 8000
"""

import sys
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from types import SimpleNamespace
import numpy as np
import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch

# Add paths
sys.path.append('/home/xd/projects/transformers/notebooks')
sys.path.append('/home/xd/projects/BARC')

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from model_hooks import set_hooks, get_outputs, get_hidden_states, get_head_output, get_attn_output, get_attn_weights
from attribute import Node, _attribute_residual

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Global State (simple module-level globals)
# =============================================================================

model = None
tokenizer = None
device = "cuda:0"

# dataset_id -> {sample_id -> {outputs, seq_len, prompt, labels, candidate_ids}}
datasets: Dict[str, Dict[int, Any]] = {}

# Lock to prevent concurrent requests (model hooks store state on model object)
# uvicorn coroutine and blocking torch ops already serialize requests on single event loop.
# Lock is defensive: protects if awaits are added or thread pool is used.
import asyncio
_lock = asyncio.Lock()

# =============================================================================
# Pydantic Models for API
# =============================================================================

class LoadModelRequest(BaseModel):
    model_path: str
    device: str = "cuda:0"
    quantization: str = "gptq"  # "gptq", "awq", "none"

class ForwardRequest(BaseModel):
    dataset_id: str
    sample_id: int
    prompt: str
    # Task metadata - stored in cache, used by compute_node_gradient for lm_head
    labels: Optional[List[int]] = None
    candidate_ids: Optional[List[int]] = None

class BatchForwardRequest(BaseModel):
    dataset_id: str
    prompts: List[Dict[str, Any]]  # [{sample_id, prompt, labels?, candidate_ids?}, ...]

class GetTensorRequest(BaseModel):
    dataset_id: str
    sample_id: int
    layer: int
    pos_ids: Optional[List[int]] = None
    head: Optional[int] = None  # For attention weights/head output

class NodeGradientRequest(BaseModel):
    """Request for compute_node_gradient endpoint."""
    dataset_id: str
    sample_id: Optional[int] = None  # None = average over entire dataset
    node_type: str  # 'attn_q', 'attn_k', 'attn_v', 'mlp_i', 'mlp_g', 'lm_head'
    layer: int
    head: Optional[int] = None  # Required for attention nodes
    pos_ids: List[int]
    src_pos_ids: Optional[List[int]] = None  # For k/v nodes
    output_grad: Optional["TensorResponse"] = None
    steps: Optional[int] = None  # IG steps: None=4, 0=plain backward
    # labels/candidate_ids removed - server gets from cached sample

class NodeForwardRequest(BaseModel):
    """Request for compute_node_forward endpoint."""
    dataset_id: str
    sample_id: int
    node_type: str  # 'attn_q', 'attn_k', 'attn_v', 'mlp_i', 'mlp_g', 'lm_head'
    layer: int
    head: Optional[int] = None  # Required for attention nodes
    pos_ids: List[int]
    src_pos_ids: Optional[List[int]] = None  # For k/v nodes
    upstream_sum: Optional["TensorResponse"] = None  # Replaces cached input

class ResidualRequest(BaseModel):
    """Request for attribute_residual endpoint."""
    dataset_id: str
    sample_id: int
    pos_ids: List[int]
    downstream_grads: List["TensorResponse"]
    downstream_layers: List[float]  # float for mlp: layer+0.5
    from_layer: int = 0
    to_layer: Optional[int] = None

class TensorResponse(BaseModel):
    shape: List[int]
    dtype: str
    data: str  # base64 encoded

class TensorPairResponse(BaseModel):
    """Response containing two tensors (e.g., attn_a returns aw, grad)."""
    first: TensorResponse
    second: TensorResponse

# =============================================================================
# Tensor Serialization
# =============================================================================

def tensor_to_response(t: torch.Tensor) -> TensorResponse:
    """Convert tensor to API response."""
    arr = t.cpu().numpy()
    return TensorResponse(
        shape=list(arr.shape),
        dtype=str(arr.dtype),
        data=base64.b64encode(arr.tobytes()).decode('utf-8')
    )

def response_to_tensor(resp: TensorResponse, dev: str = 'cpu') -> torch.Tensor:
    """Convert API response back to tensor."""
    arr = np.frombuffer(base64.b64decode(resp.data), dtype=resp.dtype).reshape(resp.shape)
    return torch.from_numpy(arr.copy()).to(dev)

# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Model server starting...")
    yield
    logger.info("Model server shutting down...")
    global model
    if model is not None:
        del model
        torch.cuda.empty_cache()

app = FastAPI(
    title="Model Server",
    description="Shared model state and activation caching for interpretability research",
    lifespan=lifespan
)

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/status")
async def get_status():
    """Get server status."""
    return {
        "model_loaded": model is not None,
        "model_name": model.config._name_or_path if model else "",
        "device": device,
        "datasets": list(datasets.keys()),
        "dataset_sizes": {k: len(v) for k, v in datasets.items()},
        "gpu_memory_mb": torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
    }

@app.post("/load_model")
async def load_model(req: LoadModelRequest):
    """Load a model."""
    global model, tokenizer, device
    logger.info(f"Loading model: {req.model_path}")
    
    if model is not None:
        del model
        torch.cuda.empty_cache()
    
    try:
        if req.quantization == "gptq":
            gptq_config = GPTQConfig(
                bits=4, group_size=128, desc_act=False, 
                use_exllama=True, exllama_config={"version": 2}
            )
            model = AutoModelForCausalLM.from_pretrained(
                req.model_path,
                torch_dtype=torch.float16,
                quantization_config=gptq_config,
                device_map=req.device,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                req.model_path,
                torch_dtype=torch.float16,
                device_map=req.device,
                local_files_only=True
            )
        
        tokenizer = AutoTokenizer.from_pretrained(req.model_path)
        device = req.device
        
        config = {
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "hidden_size": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
        }
        logger.info(f"Model loaded successfully: {config}")
        return {"status": "success", "config": config}
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forward")
async def forward_sample(req: ForwardRequest):
    """Run forward pass and cache activations."""
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    async with _lock:  # Prevent concurrent forward passes (hooks store state on model)
        try:
            inputs = tokenizer([req.prompt], return_tensors="pt").to(model.device)
            
            set_hooks(model)
            with torch.no_grad():
                output = model(**inputs, output_hidden_states=True)
            
            if req.dataset_id not in datasets:
                datasets[req.dataset_id] = {}
            
            datasets[req.dataset_id][req.sample_id] = {
                "outputs": get_outputs(model, output),
                "seq_len": inputs.input_ids.shape[1],
                "prompt": req.prompt,
                "labels": req.labels,
                "candidate_ids": req.candidate_ids,
            }
            
            return {"status": "success", "dataset_id": req.dataset_id, 
                    "sample_id": req.sample_id, "seq_len": inputs.input_ids.shape[1]}
            
        except Exception as e:
            logger.error(f"Forward failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_forward")
async def batch_forward(req: BatchForwardRequest):
    """Run forward pass for multiple samples."""
    results = []
    for item in req.prompts:
        single_req = ForwardRequest(
            dataset_id=req.dataset_id,
            sample_id=item["sample_id"],
            prompt=item["prompt"],
            labels=item.get("labels"),
            candidate_ids=item.get("candidate_ids"),
        )
        result = await forward_sample(single_req)
        results.append(result)
    return {"status": "success", "results": results}

# =============================================================================
# Helpers
# =============================================================================

def check(req):
    """Validate model loaded and sample exists."""
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    if req.dataset_id not in datasets or req.sample_id not in datasets[req.dataset_id]:
        raise HTTPException(status_code=404, detail="Sample not found")

def sample(req):
    """Get cached sample dict."""
    return datasets[req.dataset_id][req.sample_id]

def rpc(fn, req):
    """RPC helper: validate, build fake Result, call fn(model, r, ...), return tensor."""
    check(req)
    r = SimpleNamespace(
        index=req.sample_id,
        outputs=sample(req)["outputs"],
    )
    return tensor_to_response(fn(model, r,
        layer=req.layer,
        head=req.head,
        pos_ids=torch.LongTensor(req.pos_ids) if req.pos_ids else None,
    ))

# =============================================================================
# Tensor Endpoints
# =============================================================================

@app.post("/hidden_states")
async def hidden_states(req: GetTensorRequest): return rpc(get_hidden_states, req)

@app.post("/attn_weights")
async def attn_weights(req: GetTensorRequest): return rpc(get_attn_weights, req)

@app.post("/head_output")
async def head_output(req: GetTensorRequest): return rpc(get_head_output, req)

@app.post("/attn_output")
async def attn_output(req: GetTensorRequest): return rpc(get_attn_output, req)

@app.delete("/datasets/{dataset_id}")
async def clear_dataset(dataset_id: str):
    """Clear a dataset."""
    if dataset_id in datasets:
        del datasets[dataset_id]
        torch.cuda.empty_cache()
        return {"status": "success", "message": f"Dataset '{dataset_id}' cleared"}
    raise HTTPException(status_code=404, detail="Dataset not found")

@app.delete("/datasets")
async def clear_all_datasets():
    """Clear all datasets."""
    datasets.clear()
    torch.cuda.empty_cache()
    return {"status": "success", "message": "All datasets cleared"}

@app.get("/datasets/{dataset_id}/samples")
async def list_samples(dataset_id: str):
    """List sample IDs in a dataset."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "dataset_id": dataset_id,
        "sample_ids": list(datasets[dataset_id].keys()),
        "count": len(datasets[dataset_id])
    }

@app.post("/empty_cache")
async def empty_cuda_cache():
    """Empty CUDA cache to free GPU memory."""
    torch.cuda.empty_cache()
    return {
        "status": "success",
        "gpu_memory_mb": torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    }

@app.post("/compute_node_gradient")
async def compute_node_gradient(req: NodeGradientRequest):
    """Compute IG gradient for a node."""
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    if req.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found")
    
    dataset = datasets[req.dataset_id]
    if req.sample_id not in dataset:
        raise HTTPException(status_code=404, detail=f"Sample {req.sample_id} not found")
    sample = dataset[req.sample_id]
    
    async with _lock:  # Prevent concurrent model access
        # Create a temporary Node (bypass __new__ interning and __init__)
        node = object.__new__(Node)
        node.layer = req.layer
        node.head = req.head
        node.type = req.node_type
        node.pos_ids = torch.LongTensor(req.pos_ids)
        node.src_pos_ids = torch.LongTensor(req.src_pos_ids) if req.src_pos_ids else node.pos_ids
        node.dataset_size = 1
        node.dtype = torch.float16
        node.device = 'cpu'
        node.grad = None
        node.output_grad = response_to_tensor(req.output_grad, model.device).unsqueeze(0) if req.output_grad else None
        
        r = SimpleNamespace(
            index=0,
            outputs=sample["outputs"],
            labels=sample.get("labels"),
            candidate_ids=sample.get("candidate_ids"),
            answer_indices=None,
        )
        
        grad = node._backward_local(r, model)
        
        if req.node_type == 'attn_a':  # attn_a returns tuple (aw, grad)
            assert isinstance(grad, tuple) and len(grad) == 2
            return TensorPairResponse(first=tensor_to_response(grad[0]), second=tensor_to_response(grad[1]))
        return tensor_to_response(grad)

@app.post("/compute_node_forward")
async def compute_node_forward(req: NodeForwardRequest):
    """Compute forward output for a node."""
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    if req.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found")
    
    dataset = datasets[req.dataset_id]
    if req.sample_id not in dataset:
        raise HTTPException(status_code=404, detail=f"Sample {req.sample_id} not found")
    
    async with _lock:  # Prevent concurrent model access
        sample_cache = dataset[req.sample_id]
        
        # Create a temporary Node (bypass __new__ interning and __init__)
        node = object.__new__(Node)
        node.layer = req.layer
        node.head = req.head
        node.type = req.node_type
        node.pos_ids = torch.LongTensor(req.pos_ids)
        node.src_pos_ids = torch.LongTensor(req.src_pos_ids) if req.src_pos_ids else node.pos_ids
        
        r = SimpleNamespace(
            index=0,
            outputs=sample_cache["outputs"],
            labels=sample_cache.get("labels"),
            candidate_ids=sample_cache.get("candidate_ids"),
            answer_indices=None,
        )
        
        upstream_sum = response_to_tensor(req.upstream_sum, model.device) if req.upstream_sum is not None else None
        output = node._forward_local(r, model, upstream_sum=upstream_sum)
        return tensor_to_response(output)

@app.post("/attribute_residual")
async def attribute_residual(req: ResidualRequest):
    """Attribute gradients to upstream heads/MLPs. Returns [layers, heads+1, grads]."""
    check(req)
    
    async with _lock:  # Prevent concurrent model access
        r = SimpleNamespace(index=req.sample_id, outputs=sample(req)["outputs"])
        pos_ids = torch.LongTensor(req.pos_ids)
        grads = torch.cat([response_to_tensor(g, model.device) for g in req.downstream_grads]).to(model.dtype)
        
        attr = _attribute_residual(r, model, grads, pos_ids, req.downstream_layers, req.from_layer, req.to_layer)
        return tensor_to_response(attr)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    device = args.device
    uvicorn.run(app, host=args.host, port=args.port)
