import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load all CSV files
concurrent = pd.read_csv('concurrent.csv')
gpu_only = pd.read_csv('gpu_only.csv')
hybrid = pd.read_csv('hybrid.csv')
sequential = pd.read_csv('sequential.csv')

# Parse timestamps
for df in [concurrent, gpu_only, hybrid, sequential]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# ============================================================================
# PLOT 1: Latency vs Prompt Size (CPU vs GPU)
# ============================================================================
def plot_latency_vs_prompt_size():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter data for CPU and GPU
    cpu_data = concurrent[concurrent['mode'] == 'cpu']
    gpu_data = concurrent[concurrent['mode'] == 'gpu']
    
    # Scatter plots
    ax.scatter(cpu_data['prompt_length'], cpu_data['latency_s'], 
               alpha=0.6, s=100, label='CPU (Phi-3)', marker='o')
    ax.scatter(gpu_data['prompt_length'], gpu_data['latency_s'], 
               alpha=0.6, s=100, label='GPU (Gemma-2)', marker='s')
    
    # Add trend lines
    if len(cpu_data) > 1:
        z_cpu = np.polyfit(cpu_data['prompt_length'], cpu_data['latency_s'], 2)
        p_cpu = np.poly1d(z_cpu)
        x_cpu = np.linspace(cpu_data['prompt_length'].min(), 
                           cpu_data['prompt_length'].max(), 100)
        ax.plot(x_cpu, p_cpu(x_cpu), '--', alpha=0.5, linewidth=2)
    
    if len(gpu_data) > 1:
        z_gpu = np.polyfit(gpu_data['prompt_length'], gpu_data['latency_s'], 2)
        p_gpu = np.poly1d(z_gpu)
        x_gpu = np.linspace(gpu_data['prompt_length'].min(), 
                           gpu_data['prompt_length'].max(), 100)
        ax.plot(x_gpu, p_gpu(x_gpu), '--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Prompt Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Scaling: CPU vs GPU by Prompt Size', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot1_latency_vs_prompt_size.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 2: Throughput vs Prompt Size (CPU vs GPU)
# ============================================================================
def plot_throughput_vs_prompt_size():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cpu_data = concurrent[concurrent['mode'] == 'cpu']
    gpu_data = concurrent[concurrent['mode'] == 'gpu']
    
    # Group by prompt length and calculate mean throughput
    cpu_grouped = cpu_data.groupby('prompt_length')['throughput_tokens_per_s'].mean()
    gpu_grouped = gpu_data.groupby('prompt_length')['throughput_tokens_per_s'].mean()
    
    ax.plot(cpu_grouped.index, cpu_grouped.values, 
            marker='o', linewidth=2.5, markersize=8, label='CPU (Phi-3)')
    ax.plot(gpu_grouped.index, gpu_grouped.values, 
            marker='s', linewidth=2.5, markersize=8, label='GPU (Gemma-2)')
    
    ax.set_xlabel('Prompt Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Comparison: CPU vs GPU Across Prompt Sizes', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot2_throughput_vs_prompt_size.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 3: Latency Distribution Histograms (GPU-only vs Hybrid)
# ============================================================================
def plot_latency_distribution():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.hist(gpu_only['latency_s'], bins=30, alpha=0.6, 
            label='GPU-only Mode', edgecolor='black')
    ax.hist(hybrid['latency_s'], bins=30, alpha=0.6, 
            label='Hybrid Mode', edgecolor='black')
    
    ax.set_xlabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Latency Distribution: GPU-only vs Hybrid Mode', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plot3_latency_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 4: ECDF with Percentiles (GPU-only vs Hybrid)
# ============================================================================
def plot_ecdf_percentiles():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate ECDF
    gpu_sorted = np.sort(gpu_only['latency_s'])
    hybrid_sorted = np.sort(hybrid['latency_s'])
    
    gpu_ecdf = np.arange(1, len(gpu_sorted) + 1) / len(gpu_sorted)
    hybrid_ecdf = np.arange(1, len(hybrid_sorted) + 1) / len(hybrid_sorted)
    
    ax.plot(gpu_sorted, gpu_ecdf, linewidth=2.5, label='GPU-only Mode')
    ax.plot(hybrid_sorted, hybrid_ecdf, linewidth=2.5, label='Hybrid Mode')
    
    # Add percentile lines
    percentiles = [50, 95, 99]
    for p in percentiles:
        gpu_p = np.percentile(gpu_only['latency_s'], p)
        hybrid_p = np.percentile(hybrid['latency_s'], p)
        ax.axhline(y=p/100, color='gray', linestyle='--', alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.95, p/100, f'P{p}', 
                verticalalignment='bottom', fontsize=10)
    
    ax.set_xlabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('ECDF: Latency Percentiles (P50, P95, P99)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot4_ecdf_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 5: Mean and Median Throughput Comparison (Bar Chart)
# ============================================================================
def plot_throughput_comparison():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    metrics = {
        'GPU-only': {
            'mean': gpu_only['throughput_tokens_per_s'].mean(),
            'median': gpu_only['throughput_tokens_per_s'].median()
        },
        'Hybrid': {
            'mean': hybrid['throughput_tokens_per_s'].mean(),
            'median': hybrid['throughput_tokens_per_s'].median()
        }
    }
    
    x = np.arange(len(metrics))
    width = 0.35
    
    means = [metrics[m]['mean'] for m in metrics]
    medians = [metrics[m]['median'] for m in metrics]
    
    ax.bar(x - width/2, means, width, label='Mean', alpha=0.8)
    ax.bar(x + width/2, medians, width, label='Median', alpha=0.8)
    
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Comparison: GPU-only vs Hybrid Mode', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (m, med) in enumerate(zip(means, medians)):
        ax.text(i - width/2, m, f'{m:.1f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, med, f'{med:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plot5_throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 6: Request Distribution Pie Charts (CPU vs GPU in Hybrid)
# ============================================================================
def plot_request_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hybrid mode distribution
    hybrid_counts = hybrid['mode'].value_counts()
    colors = sns.color_palette('husl', len(hybrid_counts))
    
    ax1.pie(hybrid_counts.values, labels=hybrid_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 11})
    ax1.set_title('Hybrid Mode: Request Distribution', 
                  fontsize=13, fontweight='bold')
    
    # Overall concurrent distribution
    concurrent_counts = concurrent['mode'].value_counts()
    ax2.pie(concurrent_counts.values, labels=concurrent_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 11})
    ax2.set_title('Concurrent Mode: Request Distribution', 
                  fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plot6_request_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 7: CPU Utilization Changes (Scatter Plot)
# ============================================================================
def plot_cpu_utilization():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.scatter(hybrid.index, hybrid['cpu_util_delta'], 
               alpha=0.6, s=100, c=hybrid['mode'].map({'cpu': 'blue', 'gpu': 'red'}),
               label='Hybrid Mode')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Request Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('CPU Utilization Delta (%)', fontsize=12, fontweight='bold')
    ax.set_title('CPU Load Changes in Hybrid Mode', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='CPU'),
                      Patch(facecolor='red', label='GPU')]
    ax.legend(handles=legend_elements, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('plot7_cpu_utilization.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 8: GPU Utilization Changes (Scatter Plot)
# ============================================================================
def plot_gpu_utilization():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Combine relevant datasets
    all_data = pd.concat([gpu_only, hybrid])
    
    ax.scatter(range(len(all_data)), all_data['gpu_util_delta'], 
               alpha=0.6, s=100, 
               c=['red' if 'gpu_only' in str(i) else 'blue' 
                  for i in range(len(all_data))])
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.set_xlabel('Request Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('GPU Utilization Delta (%)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Load Changes: GPU-only vs Hybrid Mode', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='GPU-only'),
                      Patch(facecolor='blue', label='Hybrid')]
    ax.legend(handles=legend_elements, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('plot8_gpu_utilization.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 9: Memory Footprint Over Time
# ============================================================================
def plot_memory_footprint():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # CPU Memory
    time_offset = (hybrid['timestamp'] - hybrid['timestamp'].min()).dt.total_seconds()
    ax1.plot(time_offset, hybrid['cpu_mem_before_gb'], 
             label='Before Request', linewidth=2, marker='o', markersize=4)
    ax1.plot(time_offset, hybrid['cpu_mem_after_gb'], 
             label='After Request', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CPU Memory (GB)', fontsize=12, fontweight='bold')
    ax1.set_title('CPU Memory Footprint Over Time (Hybrid Mode)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # GPU Memory
    ax2.plot(time_offset, hybrid['gpu_mem_before_gb'], 
             label='Before Request', linewidth=2, marker='o', markersize=4)
    ax2.plot(time_offset, hybrid['gpu_mem_after_gb'], 
             label='After Request', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
    ax2.set_title('GPU Memory Footprint Over Time (Hybrid Mode)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot9_memory_footprint.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 10: Latency vs Queue Depth
# ============================================================================
def plot_latency_vs_queue_depth():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract queue depth from decision_reason
    def extract_queue_depth(reason):
        import re
        match = re.search(r'Queue depth: (\d+)', str(reason))
        return int(match.group(1)) if match else None
    
    concurrent['queue_depth'] = concurrent['decision_reason'].apply(extract_queue_depth)
    gpu_only['queue_depth'] = gpu_only['decision_reason'].apply(extract_queue_depth)
    hybrid['queue_depth'] = hybrid['decision_reason'].apply(extract_queue_depth)
    
    # Plot for different modes
    for df, label in [(gpu_only, 'GPU-only'), (hybrid, 'Hybrid')]:
        data = df.dropna(subset=['queue_depth'])
        grouped = data.groupby('queue_depth')['latency_s'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', 
                linewidth=2.5, markersize=8, label=label)
    
    ax.set_xlabel('Queue Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Latency Degradation by Queue Depth', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot10_latency_vs_queue_depth.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 11: Throughput vs Queue Depth (Concurrency)
# ============================================================================
def plot_throughput_vs_queue_depth():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for df, label in [(gpu_only, 'GPU-only'), (hybrid, 'Hybrid'), 
                      (concurrent, 'Concurrent')]:
        data = df.dropna(subset=['queue_depth'])
        grouped = data.groupby('queue_depth')['throughput_tokens_per_s'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', 
                linewidth=2.5, markersize=8, label=label)
    
    ax.set_xlabel('Queue Depth (Concurrency)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Throughput (tokens/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Degradation with Increasing Concurrency', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot11_throughput_vs_queue_depth.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 12: Queue Depth vs Prompt Size Heatmap (Latency)
# ============================================================================
def plot_heatmap_queue_prompt():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use concurrent data for comprehensive view
    data = concurrent.dropna(subset=['queue_depth'])
    
    # Create pivot table
    pivot = data.pivot_table(values='latency_s', 
                             index='queue_depth', 
                             columns='prompt_length', 
                             aggfunc='mean')
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Latency (seconds)'})
    
    ax.set_xlabel('Prompt Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Queue Depth', fontsize=12, fontweight='bold')
    ax.set_title('Latency Heatmap: Queue Depth × Prompt Size Interaction', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot12_heatmap_queue_prompt.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 13: Output Tokens vs Latency (Generation Cost)
# ============================================================================
def plot_output_tokens_vs_latency():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot for different modes
    for df, label, marker in [(gpu_only, 'GPU-only', 'o'), 
                               (hybrid, 'Hybrid', 's'),
                               (sequential, 'Sequential', '^')]:
        ax.scatter(df['output_tokens'], df['latency_s'], 
                   alpha=0.6, s=100, label=label, marker=marker)
    
    # Add correlation line for all data
    all_data = pd.concat([gpu_only, hybrid, sequential])
    z = np.polyfit(all_data['output_tokens'], all_data['latency_s'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_data['output_tokens'].min(), 
                        all_data['output_tokens'].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='gray', 
            alpha=0.7, linewidth=2, label='Trend')
    
    ax.set_xlabel('Output Tokens Generated', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Generation Length vs Latency: Isolating Generation Cost', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot13_output_tokens_vs_latency.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Generate all plots
# ============================================================================
if __name__ == "__main__":
    print("Generating Plot 1: Latency vs Prompt Size...")
    plot_latency_vs_prompt_size()
    
    print("Generating Plot 2: Throughput vs Prompt Size...")
    plot_throughput_vs_prompt_size()
    
    print("Generating Plot 3: Latency Distribution...")
    plot_latency_distribution()
    
    print("Generating Plot 4: ECDF with Percentiles...")
    plot_ecdf_percentiles()
    
    print("Generating Plot 5: Throughput Comparison...")
    plot_throughput_comparison()
    
    print("Generating Plot 6: Request Distribution...")
    plot_request_distribution()
    
    print("Generating Plot 7: CPU Utilization...")
    plot_cpu_utilization()
    
    print("Generating Plot 8: GPU Utilization...")
    plot_gpu_utilization()
    
    print("Generating Plot 9: Memory Footprint...")
    plot_memory_footprint()
    
    print("Generating Plot 10: Latency vs Queue Depth...")
    plot_latency_vs_queue_depth()
    
    print("Generating Plot 11: Throughput vs Queue Depth...")
    plot_throughput_vs_queue_depth()
    
    print("Generating Plot 12: Queue Depth × Prompt Size Heatmap...")
    plot_heatmap_queue_prompt()
    
    print("Generating Plot 13: Output Tokens vs Latency...")
    plot_output_tokens_vs_latency()
    
    print("\nAll plots generated successfully!")
    print("Plots saved as PNG files with 300 DPI resolution.")