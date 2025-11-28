# app/monitor.py
import psutil
from datetime import datetime

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def get_cpu_util() -> float:
    return psutil.cpu_percent(interval=None)


def get_memory_info() -> dict:
    mem = psutil.virtual_memory()
    return {
        "cpu_mem_total_gb": round(mem.total / (1024**3), 2),
        "cpu_mem_used_gb": round(mem.used / (1024**3), 2),
        "cpu_mem_util_pct": mem.percent,
    }


def get_gpu_stats() -> dict:
    # Always return numeric values
    if not GPU_AVAILABLE:
        return {
            "gpu_util": 0.0,
            "gpu_mem_used_gb": 0.0,
            "gpu_mem_total_gb": 0.0,
            "gpu_mem_util_pct": 0.0,
        }

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util": float(util.gpu),
            "gpu_mem_used_gb": round(mem.used / (1024**3), 2),
            "gpu_mem_total_gb": round(mem.total / (1024**3), 2),
            "gpu_mem_util_pct": round((mem.used / mem.total) * 100, 2),
        }
    except:
        return {
            "gpu_util": 0.0,
            "gpu_mem_used_gb": 0.0,
            "gpu_mem_total_gb": 0.0,
            "gpu_mem_util_pct": 0.0,
        }


def get_system_stats() -> dict:
    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "cpu_util": get_cpu_util(),
    }
    stats.update(get_memory_info())
    stats.update(get_gpu_stats())
    return stats
