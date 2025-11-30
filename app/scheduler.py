# app/scheduler.py
def choose_model(stats: dict, prompt: str, force_mode: str = None) -> tuple[str, str]:
    """
    Adaptive rule-based scheduler that chooses between CPU and GPU models.
    
    ✅ UPDATED: Now uses gemma2:2b (lighter GPU model)
    ✅ UPDATED: Stricter thresholds for better routing
    
    Args:
        stats: System statistics (CPU/GPU utilization, memory)
        prompt: Input prompt text
        force_mode: Optional override ('cpu', 'gpu', or None for adaptive)
    
    Returns:
        tuple: (selected_model, decision_reason)
    """
    
    # ⚙️ FORCE MODES FOR TESTING
    # Change these to test different scenarios:
    # force_mode = "gpu"  # Uncomment for GPU-only test
    # force_mode = "cpu"  # Uncomment for CPU-only test
    # force_mode = None   # Use adaptive scheduling (default)
    
    if force_mode == "gpu":
        return "gemma2:2b", "Force GPU mode"  # ✅ CHANGED: gemma3 → gemma2:2b
    if force_mode == "cpu":
        return "phi3", "Force CPU mode"
    
    # ✅ ADAPTIVE SCHEDULING LOGIC
    gpu_util = stats.get("gpu_util") or 0
    cpu_util = stats.get("cpu_util") or 0
    gpu_mem_used = stats.get("gpu_mem_used_gb") or 0
    gpu_mem_total = stats.get("gpu_mem_total_gb") or 1
    gpu_mem_ratio = gpu_mem_used / gpu_mem_total
    prompt_len = len(prompt.split())
    
    # ✅ Rule 1: GPU is critically overloaded → MUST use CPU
    if gpu_util > 90 and gpu_mem_ratio > 0.95:
        return "phi3", f"GPU critically overloaded (util={gpu_util}%, mem={gpu_mem_ratio*100:.1f}%)"
    
    # ✅ Rule 2: GPU is busy (>75%) → Route to CPU for small/medium prompts
    if gpu_util > 75 and prompt_len < 50:
        return "phi3", f"GPU busy ({gpu_util}%), routing small prompt to CPU"
    
    # ✅ Rule 3: Very large prompt (>80 words) → Prefer GPU if available
    if prompt_len > 80 and gpu_util < 85:
        return "gemma2:2b", f"Large prompt (len={prompt_len}), using GPU"  # ✅ CHANGED
    
    # ✅ Rule 4: Tiny prompt (<8 words) → Always use CPU (faster for small tasks)
    # CHANGED: Stricter threshold (was 15, now 8)
    if prompt_len < 8:
        return "phi3", f"Tiny prompt (len={prompt_len}), CPU optimal"
    
    # ✅ Rule 5: CPU is overloaded (>80%) → Offload to GPU
    if cpu_util > 80 and gpu_util < 75:
        return "gemma2:2b", f"CPU overloaded ({cpu_util}%), offloading to GPU"  # ✅ CHANGED
    
    # ✅ Rule 6: Small-medium prompts (8-25 words) → Prefer GPU if available
    # CHANGED: New rule for 8-25 word prompts
    if 8 <= prompt_len <= 25:
        if gpu_util < 60:
            return "gemma2:2b", f"Small prompt, GPU available (util={gpu_util}%)"  # ✅ CHANGED
        else:
            return "phi3", f"Small prompt, GPU busy, using CPU"
    
    # ✅ Rule 7: Medium prompts (25-80 words) → Prefer GPU
    # CHANGED: Renamed from Rule 6, adjusted range
    if 25 < prompt_len <= 80:
        if gpu_util < 70:
            return "gemma2:2b", f"Medium prompt, GPU available (util={gpu_util}%)"  # ✅ CHANGED
        else:
            return "phi3", f"Medium prompt, GPU busy, using CPU"
    
    # ✅ Fallback: Choose less busy resource
    if gpu_util < cpu_util and gpu_util < 70:
        return "gemma2:2b", f"GPU less busy (GPU={gpu_util}% vs CPU={cpu_util}%)"  # ✅ CHANGED
    else:
        return "phi3", f"CPU selected (CPU={cpu_util}%, GPU={gpu_util}%)"