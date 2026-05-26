"""
Model Server Client - Python SDK for LLM interpretability research.

See MODEL_SERVER_API.md for full documentation.

Quick Start:
    from model_client import ModelClient
    client = ModelClient("http://localhost:8000")
    
    client.load_model("/path/to/model")
    
    # Forward with task metadata (labels/candidate_ids stored on server)
    client.forward_results("my_dataset", results)
    
    # Get activations
    attn = client.get_attn_weights(0, layer=60, head=10)
    
    # Compute gradients - no need to pass labels/candidate_ids, server has them
    grad = client.compute_node_gradient(0, "lm_head", layer=80, pos_ids=[100])
"""

import requests
import torch
import numpy as np
import base64
from typing import Optional, List, Dict, Any
from tqdm import tqdm


class ModelClient:
    """Client for interacting with the model server.
    
    After calling forward_results(), the dataset_id is remembered and used
    for subsequent calls. You can override it by passing dataset_id explicitly.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", device: str = "cuda:0"):
        self.base_url = base_url.rstrip("/")
        self._config = None
        self.dataset_id = None  # Set by forward_results
        self.device = device
    
    def _get_dataset_id(self, dataset_id: Optional[str]) -> str:
        """Get dataset_id, using stored value if not provided."""
        if dataset_id is not None:
            return dataset_id
        if self.dataset_id is None:
            raise ValueError("No dataset_id set. Call forward_results() first or pass dataset_id explicitly.")
        return self.dataset_id
    
    def _post(self, endpoint: str, data: dict) -> dict:
        """Make POST request."""
        resp = requests.post(f"{self.base_url}{endpoint}", json=data)
        if resp.status_code != 200:
            raise RuntimeError(f"Server error: {resp.status_code} - {resp.text}")
        return resp.json()
    
    def _get(self, endpoint: str) -> dict:
        """Make GET request."""
        resp = requests.get(f"{self.base_url}{endpoint}")
        if resp.status_code != 200:
            raise RuntimeError(f"Server error: {resp.status_code} - {resp.text}")
        return resp.json()
    
    def _delete(self, endpoint: str) -> dict:
        """Make DELETE request."""
        resp = requests.delete(f"{self.base_url}{endpoint}")
        if resp.status_code != 200:
            raise RuntimeError(f"Server error: {resp.status_code} - {resp.text}")
        return resp.json()
    
    def _response_to_tensor(self, resp: dict, device: str = 'cpu') -> torch.Tensor:
        """Convert API response to tensor."""
        arr = np.frombuffer(
            base64.b64decode(resp['data']), 
            dtype=resp['dtype']
        ).reshape(resp['shape'])
        return torch.from_numpy(arr.copy()).to(device)
    
    def _tensor_to_payload(self, t: torch.Tensor) -> dict:
        """Convert tensor to payload for API request."""
        arr = t.cpu().numpy()
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data": base64.b64encode(arr.tobytes()).decode('utf-8')
        }
    
    # =========================================================================
    # Status and Configuration
    # =========================================================================
    
    def status(self) -> dict:
        """Get server status."""
        return self._get("/status")
    
    @property
    def config(self) -> dict:
        """Get model config (cached)."""
        if self._config is None:
            status = self.status()
            if status["model_loaded"]:
                self._config = {
                    "num_layers": None,
                    "num_heads": None,
                }
        return self._config
    
    # =========================================================================
    # Model Loading
    # =========================================================================
    
    def load_model(
        self, 
        model_path: str, 
        device: str = "cuda:0",
        quantization: str = "gptq"
    ) -> dict:
        """Load a model on the server."""
        result = self._post("/load_model", {
            "model_path": model_path,
            "device": device,
            "quantization": quantization
        })
        self._config = result.get("config")
        return result
    
    # =========================================================================
    # Forward Pass and Caching
    # =========================================================================
    
    def forward(
        self, 
        dataset_id: str, 
        sample_id: int, 
        prompt: str,
        labels: Optional[List[int]] = None,
        candidate_ids: Optional[List[int]] = None,
    ) -> dict:
        """Run forward pass and cache activations with task metadata."""
        return self._post("/forward", {
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "prompt": prompt,
            "labels": labels,
            "candidate_ids": candidate_ids,
        })
    
    def batch_forward(
        self, 
        dataset_id: str, 
        prompts: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> dict:
        """Run forward pass for multiple samples.
        
        Args:
            dataset_id: ID for this dataset
            prompts: List of {sample_id, prompt, labels?, candidate_ids?}
            show_progress: Show tqdm progress bar
        """
        self.dataset_id = dataset_id  # Remember for subsequent calls
        
        if show_progress:
            results = []
            for item in tqdm(prompts, desc="Forwarding"):
                result = self.forward(
                    dataset_id, 
                    item["sample_id"], 
                    item["prompt"],
                    item.get("labels"),
                    item.get("candidate_ids"),
                )
                results.append(result)
            return {"status": "success", "results": results}
        else:
            return self._post("/batch_forward", {
                "dataset_id": dataset_id,
                "prompts": prompts
            })
    
    def forward_results(
        self, 
        dataset_id: str, 
        results: list,
        prompt_attr: str = "prompt",
        show_progress: bool = True
    ):
        """Forward a list of Result objects with task metadata.
        
        Args:
            dataset_id: ID for this dataset
            results: List of Result objects with .prompt, .labels, .candidate_ids
            prompt_attr: Attribute name for prompt (default: "prompt")
        """
        prompts = []
        for i, r in enumerate(results):
            item = {
                "sample_id": getattr(r, 'index', i),
                "prompt": getattr(r, prompt_attr),
            }
            # Include task metadata if available
            if hasattr(r, 'labels'):
                item["labels"] = r.labels
            if hasattr(r, 'candidate_ids'):
                item["candidate_ids"] = r.candidate_ids
            prompts.append(item)
        
        return self.batch_forward(dataset_id, prompts, show_progress)
    
    # =========================================================================
    # Get Tensors (sample_id, layer, head=None, pos_ids=None, device='cpu', dataset_id=None)
    # =========================================================================
    
    def _rpc(self, endpoint, **kw):
        """RPC helper: POST to endpoint, return tensor."""
        device = kw.pop('device', self.device)
        kw['dataset_id'] = self._get_dataset_id(kw.get('dataset_id'))
        # Convert tensor pos_ids to list for JSON serialization
        if 'pos_ids' in kw and hasattr(kw['pos_ids'], 'tolist'):
            kw['pos_ids'] = kw['pos_ids'].tolist()
        return self._response_to_tensor(self._post(endpoint, kw), device)
    
    def get_hidden_states(self, **kw): return self._rpc("/hidden_states", **kw)
    def get_attn_weights(self, **kw): return self._rpc("/attn_weights", **kw)
    def get_head_output(self, **kw): return self._rpc("/head_output", **kw)
    def get_attn_output(self, **kw): return self._rpc("/attn_output", **kw)
    
    # =========================================================================
    # Dataset Management
    # =========================================================================
    
    def list_samples(self, dataset_id: Optional[str] = None) -> dict:
        """List sample IDs in a dataset."""
        return self._get(f"/datasets/{self._get_dataset_id(dataset_id)}/samples")
    
    def clear_dataset(self, dataset_id: Optional[str] = None) -> dict:
        """Clear a dataset."""
        return self._delete(f"/datasets/{self._get_dataset_id(dataset_id)}")
    
    def clear_all_datasets(self) -> dict:
        """Clear all datasets."""
        return self._delete("/datasets")
    
    def empty_cuda_cache(self) -> dict:
        """Empty CUDA cache on server to free GPU memory."""
        return self._post("/empty_cache", {})
    
    # =========================================================================
    # Gradient Computation
    # =========================================================================
    
    def compute_node_gradient(
        self,
        sample_id: Optional[int],  # None = average over entire dataset
        node_type: str,  # 'attn_q', 'attn_k', 'attn_v', 'mlp_i', 'mlp_g', 'lm_head'
        layer: int,
        pos_ids: List[int],
        head: Optional[int] = None,
        src_pos_ids: Optional[List[int]] = None,
        output_grad: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        device: str = 'cpu',
        dataset_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute IG gradient for a node.
        
        For lm_head, labels and candidate_ids are retrieved from the cached
        sample (passed during forward_results).
        
        Args:
            sample_id: Sample index, or None to average over entire dataset
            node_type: One of 'attn_q', 'attn_k', 'attn_v', 'mlp_i', 'mlp_g', 'lm_head'
            layer: Layer index
            pos_ids: Query position indices
            head: Attention head index (required for attention nodes)
            src_pos_ids: Source positions for k/v nodes
            output_grad: Gradient from downstream in attribution
            steps: IG steps (None=4, 0=plain backward)
            device: Device for returned tensor
            dataset_id: Dataset ID (uses stored value if None)
        
        Returns:
            Gradient tensor
        """
        payload = {
            "dataset_id": self._get_dataset_id(dataset_id),
            "sample_id": sample_id,
            "node_type": node_type,
            "layer": layer,
            "pos_ids": pos_ids,
            "head": head,
            "src_pos_ids": src_pos_ids,
            "steps": steps,
        }
        
        if output_grad is not None:
            payload["output_grad"] = self._tensor_to_payload(output_grad)
        
        resp = self._post("/compute_node_gradient", payload)
        
        # attn_a returns tuple (aw, grad), others return single tensor
        if node_type == 'attn_a':
            return (
                self._response_to_tensor(resp["first"], device),
                self._response_to_tensor(resp["second"], device),
            )
        return self._response_to_tensor(resp, device)
    
    def compute_node_forward(
        self,
        sample_id: int,
        node_type: str,  # 'attn_q', 'attn_k', 'attn_v', 'mlp_i', 'mlp_g', 'lm_head'
        layer: int,
        pos_ids: List[int],
        head: Optional[int] = None,
        src_pos_ids: Optional[List[int]] = None,
        upstream_sum: Optional[torch.Tensor] = None,
        device: str = 'cpu',
        dataset_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute forward output for a node.
        
        Args:
            sample_id: Sample index
            node_type: One of 'attn_q', 'attn_k', 'attn_v', 'mlp_i', 'mlp_g', 'lm_head'
            layer: Layer index
            pos_ids: Query position indices
            head: Attention head index (required for attention nodes)
            src_pos_ids: Source positions for k/v nodes
            upstream_sum: Tensor to replace cached input (from upstream nodes)
            device: Device for returned tensor
            dataset_id: Dataset ID (uses stored value if None)
        
        Returns:
            Output tensor
        """
        payload = {
            "dataset_id": self._get_dataset_id(dataset_id),
            "sample_id": sample_id,
            "node_type": node_type,
            "layer": layer,
            "pos_ids": pos_ids,
            "head": head,
            "src_pos_ids": src_pos_ids,
        }
        
        if upstream_sum is not None:
            payload["upstream_sum"] = self._tensor_to_payload(upstream_sum)
        
        resp = self._post("/compute_node_forward", payload)
        return self._response_to_tensor(resp, device)
    
    def compute_node_gradient_batch(
        self,
        sample_ids: List[int],
        node_type: str,
        layer: int,
        pos_ids: List[int],
        head: Optional[int] = None,
        src_pos_ids: Optional[List[int]] = None,
        output_grads: Optional[List[torch.Tensor]] = None,
        steps: Optional[int] = None,
        device: str = 'cpu',
        show_progress: bool = False,
        dataset_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute gradients for multiple samples, stacked."""
        dataset_id = self._get_dataset_id(dataset_id)
        iterator = enumerate(sample_ids)
        if show_progress:
            iterator = tqdm(list(iterator), desc=f"Computing grad L{layer}")
        
        grads = []
        for idx, sid in iterator:
            out_grad = output_grads[idx] if output_grads else None
            grad = self.compute_node_gradient(
                sample_id=sid,
                node_type=node_type,
                layer=layer,
                pos_ids=pos_ids,
                head=head,
                src_pos_ids=src_pos_ids,
                output_grad=out_grad,
                steps=steps,
                device=device,
                dataset_id=dataset_id,
            )
            grads.append(grad)
        
        return torch.stack(grads)
    
    def attribute_residual(
        self,
        sample_id: int,
        pos_ids: List[int],
        downstream_grads: List[torch.Tensor],
        downstream_layers: List[float],
        from_layer: int = 0,
        to_layer: Optional[int] = None,
        device: str = 'cpu',
        dataset_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Attribute gradients to upstream residual connections.
        
        Returns tensor of shape [num_layers, num_heads+1, num_downstream_nodes]
        where the last index in dim 1 is MLP attribution.
        """
        payload = {
            "dataset_id": self._get_dataset_id(dataset_id),
            "sample_id": sample_id,
            "pos_ids": pos_ids,
            "downstream_grads": [self._tensor_to_payload(g) for g in downstream_grads],
            "downstream_layers": downstream_layers,
            "from_layer": from_layer,
            "to_layer": to_layer,
        }
        
        resp = self._post("/attribute_residual", payload)
        return self._response_to_tensor(resp, device)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_client = None

def get_client(base_url: str = "http://localhost:8000") -> ModelClient:
    """Get or create default client."""
    global _default_client
    if _default_client is None or _default_client.base_url != base_url:
        _default_client = ModelClient(base_url)
    return _default_client


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    client = ModelClient("http://localhost:8000")
    
    print("Server status:", client.status())
    
    status = client.status()
    if not status["model_loaded"]:
        print("Loading model...")
        client.load_model("/data0/modelscope/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4")
    
    # Forward a sample with task metadata
    print("Running forward pass...")
    client.forward("test", 0, "Hello, world!", labels=[1, 2], candidate_ids=[1, 2, 3])
    
    # Now dataset_id is set, can omit it
    print("Getting attention weights...")
    attn = client.get_attn_weights(0, layer=60, head=10)
    print(f"Attention shape: {attn.shape}")
