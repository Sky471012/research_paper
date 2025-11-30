import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# ============================================================
# 🔧 CHANGE THIS FILE FOR EACH RUN
# ============================================================
CSV_FILE = "logs_sequential.csv"   # change to:
# logs_concurrent.csv
# logs_gpu_only.csv
# logs_hybrid.csv

df = pd.read_csv(CSV_FILE)

# ============================================================
# ✅ DEVICE CLASSIFICATION FROM decision_reason
# ============================================================
def classify_device(reason):
    r = str(reason).lower()
    if "gpu" in r:
        return "GPU"
    elif "cpu" in r:
        return "CPU"
    else:
        return "UNKNOWN"

df["routed_device"] = df["decision_reason"].apply(classify_device)

# # ============================================================
# # 🔵 PART A — SEQUENTIAL LOGS (BASELINE)
# # Enable ONLY when CSV = logs_sequential.csv
# # ============================================================

# # ---- Latency vs Prompt Length
# plt.figure()
# sns.scatterplot(data=df, x="prompt_length", y="latency_s")
# plt.title("Sequential: Latency vs Prompt Length")
# plt.savefig("A_latency_vs_prompt.png")
# plt.close()

# # ---- Throughput vs Prompt Length
# plt.figure()
# sns.scatterplot(data=df, x="prompt_length", y="throughput_tokens_per_s")
# plt.title("Sequential: Throughput vs Prompt Length")
# plt.savefig("A_throughput_vs_prompt.png")
# plt.close()

# # ---- CPU Util Before vs After
# plt.figure()
# sns.scatterplot(data=df, x="cpu_util_before", y="cpu_util_after")
# plt.title("Sequential: CPU Util Before vs After")
# plt.savefig("A_cpu_util_before_after.png")
# plt.close()

# # ---- GPU Util Before vs After
# plt.figure()
# sns.scatterplot(data=df, x="gpu_util_before", y="gpu_util_after")
# plt.title("Sequential: GPU Util Before vs After")
# plt.savefig("A_gpu_util_before_after.png")
# plt.close()

# # ============================================================
# # 🔴 PART B — CONCURRENT LOGS (CONTENDED SYSTEM)
# # Enable ONLY when CSV = logs_concurrent.csv
# # ============================================================

# # ---- Latency Distribution
# plt.figure()
# sns.histplot(df["latency_s"], kde=True)
# plt.title("Concurrent: Latency Distribution")
# plt.savefig("B_latency_distribution.png")
# plt.close()

# # ---- Throughput Distribution
# plt.figure()
# sns.histplot(df["throughput_tokens_per_s"], kde=True)
# plt.title("Concurrent: Throughput Distribution")
# plt.savefig("B_throughput_distribution.png")
# plt.close()

# # ---- Routing Distribution CPU vs GPU
# plt.figure()
# sns.countplot(data=df, x="routed_device")
# plt.title("Concurrent: Routing Distribution (CPU vs GPU)")
# plt.savefig("B_routing_distribution.png")
# plt.close()

# # ---- CPU Util Per Request
# plt.figure()
# sns.histplot(df["cpu_util_after"], kde=True)
# plt.title("Concurrent: CPU Utilization After Each Request")
# plt.savefig("B_cpu_util_per_request.png")
# plt.close()

# # ---- GPU Util Per Request
# plt.figure()
# sns.histplot(df["gpu_util_after"], kde=True)
# plt.title("Concurrent: GPU Utilization After Each Request")
# plt.savefig("B_gpu_util_per_request.png")
# plt.close()

# =======================
# PART C — PHASE 1 (SUMMARY ONLY)
# Run this ONCE per dataset
# =======================

mean_latency = df["latency_s"].mean()
p95_latency = df["latency_s"].quantile(0.95)
mean_throughput = df["throughput_tokens_per_s"].mean()

summary_df = pd.DataFrame([{
    "mean_latency_s": mean_latency,
    "p95_latency_s": p95_latency,
    "mean_throughput_tok_s": mean_throughput
}])

summary_df.to_csv("C_summary_metrics.csv", index=False)

print("✅ C_summary_metrics.csv generated for this mode ONLY")


# # ============================================================
# # 🟢 PART C — TRUE GPU-ONLY vs HYBRID COMPARISON (PAPER FIGURES)
# # ============================================================

# gpu_df = pd.read_csv("C_summary_gpu_only.csv").assign(mode="GPU-Only")
# hybrid_df = pd.read_csv("C_summary_hybrid.csv").assign(mode="Hybrid")

# comp_df = pd.concat([gpu_df, hybrid_df])

# # ---- Mean Latency Comparison
# plt.figure()
# sns.barplot(data=comp_df, x="mode", y="mean_latency_s")
# plt.title("Mean Latency: GPU-Only vs Hybrid")
# plt.savefig("C_mean_latency_gpu_vs_hybrid.png")
# plt.close()

# # ---- P95 Latency Comparison
# plt.figure()
# sns.barplot(data=comp_df, x="mode", y="p95_latency_s")
# plt.title("P95 Latency: GPU-Only vs Hybrid")
# plt.savefig("C_p95_latency_gpu_vs_hybrid.png")
# plt.close()

# # ---- Mean Throughput Comparison
# plt.figure()
# sns.barplot(data=comp_df, x="mode", y="mean_throughput_tok_s")
# plt.title("Mean Throughput: GPU-Only vs Hybrid")
# plt.savefig("C_mean_throughput_gpu_vs_hybrid.png")
# plt.close()

# print("✅ GPU-Only vs Hybrid comparison plots generated")


# # ---- Routing Distribution (Hybrid Only)

# def classify_device(reason):
#     reason = str(reason).lower()
#     if "gpu" in reason:
#         return "GPU"
#     elif "cpu" in reason:
#         return "CPU"
#     else:
#         return "UNKNOWN"

# df["routed_device"] = df["decision_reason"].apply(classify_device)

# plt.figure()
# sns.countplot(data=df, x="routed_device")
# plt.title("Routing Distribution (Hybrid Mode)")
# plt.savefig("C_routing_distribution_hybrid.png")
# plt.close()

# gpu_raw = pd.read_csv("logs_gpu_only.csv").assign(mode="GPU-Only")
# hybrid_raw = pd.read_csv("logs_hybrid.csv").assign(mode="Hybrid")

# lat_df = pd.concat([gpu_raw, hybrid_raw])

# plt.figure()
# sns.histplot(data=lat_df, x="latency_s", hue="mode", kde=True)
# plt.title("Latency Distribution: GPU-Only vs Hybrid")
# plt.savefig("C_latency_distribution_gpu_vs_hybrid.png")
# plt.close()


# # ============================================================
# # 🟣 PART D — FINAL COMBINED SUMMARY (PAPER FIGURES)
# # ============================================================

# import os

# required_files = [
#     "C_summary_sequential.csv",
#     "C_summary_concurrent.csv",
#     "C_summary_hybrid.csv"
# ]

# missing = [f for f in required_files if not os.path.exists(f)]
# if missing:
#     print("❌ Cannot run Part-D. Missing files:")
#     for f in missing:
#         print("   -", f)
#     exit()

# # ---- Load all summaries
# seq_df = pd.read_csv("C_summary_sequential.csv").assign(mode="Sequential")
# con_df = pd.read_csv("C_summary_concurrent.csv").assign(mode="Concurrent")
# hyb_df = pd.read_csv("C_summary_hybrid.csv").assign(mode="Hybrid")

# summary_df = pd.concat([seq_df, con_df, hyb_df])

# # ============================================================
# # ✅ FIGURE 2 — Overall Mean Latency Comparison
# # ============================================================

# plt.figure()
# sns.barplot(data=summary_df, x="mode", y="mean_latency_s")
# plt.title("Overall Mean Latency Comparison")
# plt.xlabel("Execution Mode")
# plt.ylabel("Mean Latency (s)")
# plt.savefig("D_overall_mean_latency_comparison.png")
# plt.close()

# # ============================================================
# # ✅ FIGURE 3 — Overall Mean Throughput Comparison
# # ============================================================

# plt.figure()
# sns.barplot(data=summary_df, x="mode", y="mean_throughput_tok_s")
# plt.title("Overall Mean Throughput Comparison")
# plt.xlabel("Execution Mode")
# plt.ylabel("Mean Throughput (tokens/sec)")
# plt.savefig("D_overall_mean_throughput_comparison.png")
# plt.close()

# print("✅ Part-D Final Summary Figures Generated Successfully")
