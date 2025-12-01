import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# Configuration
# ================================
SEQUENTIAL_LOG = "sequential.csv"     # replace with actual file
CONCURRENT_LOG = "concurrent.csv"     # replace with actual file
GPU_ONLY_LOG = "gpu_only.csv"         # raw GPU-only logs
HYBRID_LOG = "hybrid.csv"             # raw hybrid logs

# Output directory
OUTPUT_DIR = "./plots/"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# Load Logs
# ================================
def load_logs(path, workload_type):
    df = pd.read_csv(path)
    df["workload_type"] = workload_type
    return df

df_seq = load_logs(SEQUENTIAL_LOG, "sequential")
df_con = load_logs(CONCURRENT_LOG, "concurrent")

# Load these later once logs are available
try:
    df_gpu_only = load_logs(GPU_ONLY_LOG, "gpu_only")
except:
    df_gpu_only = pd.DataFrame(columns=df_seq.columns.tolist() + ["workload_type"])

try:
    df_hybrid = load_logs(HYBRID_LOG, "hybrid")
except:
    df_hybrid = pd.DataFrame(columns=df_seq.columns.tolist() + ["workload_type"])

# Merge all available logs
df = pd.concat([df_seq, df_con, df_gpu_only, df_hybrid], ignore_index=True)

# ================================
# Convert timestamp to datetime
# ================================
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="ignore")

# ================================
# Basic Cleaning
# ================================
df["mode"] = df["mode"].astype(str)
df["selected_model"] = df["selected_model"].astype(str)

# ================================
# Utility function
# ================================
def savefig(name):
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=300, bbox_inches="tight")
    plt.clf()


# =============================================================
# 1. Latency vs Prompt Length (CPU vs GPU)
# =============================================================
sns.scatterplot(
    data=df,
    x="prompt_length",
    y="latency_s",
    hue="mode",
    style="workload_type",
    alpha=0.75
)
plt.title("Latency vs Prompt Length")
plt.xlabel("Prompt Length (tokens)")
plt.ylabel("Latency (s)")
savefig("latency_vs_prompt_length")


# =============================================================
# 2. Throughput vs Prompt Length
# =============================================================
sns.scatterplot(
    data=df,
    x="prompt_length",
    y="throughput_tokens_per_s",
    hue="mode",
    style="workload_type"
)
plt.title("Throughput vs Prompt Length")
plt.xlabel("Prompt Length (tokens)")
plt.ylabel("Throughput (tokens/s)")
savefig("throughput_vs_prompt_length")


# =============================================================
# 3. Sequential vs Concurrent Latency Distribution
# =============================================================
sns.boxplot(
    data=df[df["workload_type"].isin(["sequential", "concurrent"])],
    x="workload_type",
    y="latency_s"
)
plt.title("Latency Distribution: Sequential vs Concurrent")
plt.xlabel("Workload Type")
plt.ylabel("Latency (s)")
savefig("latency_seq_vs_concurrent")


# =============================================================
# 4. Sequential vs Concurrent Throughput
# =============================================================
sns.boxplot(
    data=df[df["workload_type"].isin(["sequential", "concurrent"])],
    x="workload_type",
    y="throughput_tokens_per_s"
)
plt.title("Throughput Distribution: Sequential vs Concurrent")
plt.xlabel("Workload Type")
plt.ylabel("Throughput (tokens/s)")
savefig("throughput_seq_vs_concurrent")


# =============================================================
# 5. CPU Utilization Before/After vs Prompt Length
# =============================================================
df_melt_cpu = df.melt(
    id_vars=["prompt_length", "mode", "workload_type"],
    value_vars=["cpu_util_before", "cpu_util_after"],
    var_name="metric",
    value_name="cpu_util"
)
sns.lineplot(
    data=df_melt_cpu,
    x="prompt_length",
    y="cpu_util",
    hue="metric",
    style="workload_type"
)
plt.title("CPU Utilization Before/After vs Prompt Length")
plt.xlabel("Prompt Length")
plt.ylabel("CPU Util (%)")
savefig("cpu_util_vs_prompt")


# =============================================================
# 6. GPU Utilization Before/After vs Prompt Length
# =============================================================
df_melt_gpu = df.melt(
    id_vars=["prompt_length", "mode", "workload_type"],
    value_vars=["gpu_util_before", "gpu_util_after"],
    var_name="metric",
    value_name="gpu_util"
)
sns.lineplot(
    data=df_melt_gpu,
    x="prompt_length",
    y="gpu_util",
    hue="metric",
    style="workload_type"
)
plt.title("GPU Utilization Before/After vs Prompt Length")
plt.xlabel("Prompt Length")
plt.ylabel("GPU Util (%)")
savefig("gpu_util_vs_prompt")


# =============================================================
# 7. Tokens/sec Degradation Under Load
# =============================================================
sns.scatterplot(
    data=df,
    x="latency_s",
    y="throughput_tokens_per_s",
    hue="workload_type",
    style="mode"
)
plt.title("Token Throughput vs Latency (Degradation Under Load)")
plt.xlabel("Latency (s)")
plt.ylabel("Throughput (tokens/s)")
savefig("throughput_vs_latency")


# =============================================================
# 8. Routing Distribution (CPU vs GPU)
# =============================================================
sns.countplot(data=df, x="mode")
plt.title("Routing Distribution: CPU vs GPU")
plt.xlabel("Device")
plt.ylabel("Request Count")
savefig("routing_distribution")


# =============================================================
# 9. Latency vs Output Tokens
# =============================================================
sns.scatterplot(
    data=df,
    x="output_tokens",
    y="latency_s",
    hue="mode",
    style="workload_type"
)
plt.title("Latency vs Output Tokens")
plt.xlabel("Output Tokens")
plt.ylabel("Latency (s)")
savefig("latency_vs_output_tokens")


# =============================================================
# 10. Throughput vs Output Tokens
# =============================================================
sns.scatterplot(
    data=df,
    x="output_tokens",
    y="throughput_tokens_per_s",
    hue="mode",
    style="workload_type"
)
plt.title("Throughput vs Output Tokens")
plt.xlabel("Output Tokens")
plt.ylabel("Throughput (tokens/s)")
savefig("throughput_vs_output_tokens")


print("All plots generated in:", OUTPUT_DIR)