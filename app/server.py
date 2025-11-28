# app/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model_loader import run_inference
from .monitor import get_system_stats
from .logger import log_metrics
from .scheduler import choose_model
import time
import asyncio

app = FastAPI(title="Adaptive Scheduler - Hybrid CPU/GPU")

# Track concurrent requests for queue depth
active_requests = 0
request_lock = asyncio.Lock()

class InferenceRequest(BaseModel):
    prompt: str


@app.post("/infer")
async def infer(req: InferenceRequest):
    global active_requests
    
    # Track queue depth
    async with request_lock:
        active_requests += 1
        current_queue_depth = active_requests
    
    try:
        # Get system stats BEFORE inference
        stats_before = get_system_stats()
        
        # **ADAPTIVE SCHEDULING** - Chooses CPU or GPU based on load
        # Set force_mode="gpu" or "cpu" to override, or None for adaptive
        selected_model, decision_reason = choose_model(
            stats_before, 
            req.prompt, 
            force_mode=None  # ✅ CHANGED: None = adaptive, "gpu" = force GPU, "cpu" = force CPU
        )
        
        # Add queue depth to decision reason
        decision_reason += f" | Queue depth: {current_queue_depth}"
        
        # Run inference
        start = time.time()
        output, latency = await asyncio.to_thread(run_inference, req.prompt, selected_model)
        end = time.time()
        
        # Check for errors
        if "[Error" in output:
            raise HTTPException(status_code=500, detail=f"Inference failed: {output}")
        
        # Get system stats AFTER inference
        stats_after = get_system_stats()
        
        # Calculate deltas
        gpu_util_delta = stats_after["gpu_util"] - stats_before["gpu_util"]
        cpu_util_delta = stats_after["cpu_util"] - stats_before["cpu_util"]
        
        # Calculate throughput
        output_tokens = len(output.split())
        throughput = round(output_tokens / latency, 2) if latency > 0 else 0.0
        
        # Determine mode from model name
        mode = "gpu" if "gemma" in selected_model.lower() else "cpu"
        
        # Log all metrics
        record = {
            "timestamp": stats_before["timestamp"],
            "mode": mode,
            "selected_model": selected_model,
            "decision_reason": decision_reason,
            "latency_s": round(latency, 3),
            "prompt_length": len(req.prompt.split()),
            "output_tokens": output_tokens,
            "throughput_tokens_per_s": throughput,
            
            "cpu_util_before": stats_before["cpu_util"],
            "cpu_util_after": stats_after["cpu_util"],
            "cpu_util_delta": round(cpu_util_delta, 2),
            
            "gpu_util_before": stats_before["gpu_util"],
            "gpu_util_after": stats_after["gpu_util"],
            "gpu_util_delta": round(gpu_util_delta, 2),
            
            "cpu_mem_before_gb": stats_before["cpu_mem_used_gb"],
            "cpu_mem_after_gb": stats_after["cpu_mem_used_gb"],
            
            "gpu_mem_before_gb": stats_before["gpu_mem_used_gb"],
            "gpu_mem_after_gb": stats_after["gpu_mem_used_gb"],
            
            "gpu_mem_util_before_pct": stats_before["gpu_mem_util_pct"],
            "gpu_mem_util_after_pct": stats_after["gpu_mem_util_pct"],
        }
        
        log_metrics(record)
        
        # Return response (with truncated output)
        record["output"] = output[:1200]
        return record
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Decrement queue depth
        async with request_lock:
            active_requests -= 1


@app.get("/health")
async def health():
    """Health check endpoint."""
    stats = get_system_stats()
    return {
        "status": "healthy",
        "active_requests": active_requests,
        "cpu_util": stats["cpu_util"],
        "gpu_util": stats["gpu_util"],
    }


@app.get("/stats")
async def stats():
    """Get current system statistics."""
    return get_system_stats()