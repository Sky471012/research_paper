import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Load datasets
sequential = pd.read_csv('sequential.csv')
concurrent = pd.read_csv('concurrent.csv')
gpu_only = pd.read_csv('gpu_only.csv')
hybrid = pd.read_csv('hybrid.csv')

# ==============================================================================
# 1. Sequential Latency vs Prompt Length
# ==============================================================================
def plot_sequential_latency_vs_prompt():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate by mode
    cpu_data = sequential[sequential['mode'] == 'cpu']
    gpu_data = sequential[sequential['mode'] == 'gpu']
    
    sns.scatterplot(data=cpu_data, x='prompt_length', y='latency_s', 
                    label='CPU (phi3)', s=100, alpha=0.7, ax=ax)
    sns.scatterplot(data=gpu_data, x='prompt_length', y='latency_s', 
                    label='GPU (gemma2:2b)', s=100, alpha=0.7, ax=ax)
    
    # Add trend lines
    if len(cpu_data) > 1:
        z_cpu = np.polyfit(cpu_data['prompt_length'], cpu_data['latency_s'], 1)
        p_cpu = np.poly1d(z_cpu)
        x_cpu = np.linspace(cpu_data['prompt_length'].min(), cpu_data['prompt_length'].max(), 100)
        ax.plot(x_cpu, p_cpu(x_cpu), '--', alpha=0.5, linewidth=2)
    
    if len(gpu_data) > 1:
        z_gpu = np.polyfit(gpu_data['prompt_length'], gpu_data['latency_s'], 1)
        p_gpu = np.poly1d(z_gpu)
        x_gpu = np.linspace(gpu_data['prompt_length'].min(), gpu_data['prompt_length'].max(), 100)
        ax.plot(x_gpu, p_gpu(x_gpu), '--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Prompt Length (tokens)')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Sequential Processing: Latency vs Prompt Length')
    ax.legend()
    plt.tight_layout()
    return fig

# ==============================================================================
# 2. Sequential Throughput vs Prompt Length
# ==============================================================================
def plot_sequential_throughput_vs_prompt():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cpu_data = sequential[sequential['mode'] == 'cpu']
    gpu_data = sequential[sequential['mode'] == 'gpu']
    
    sns.scatterplot(data=cpu_data, x='prompt_length', y='throughput_tokens_per_s',
                    label='CPU (phi3)', s=100, alpha=0.7, ax=ax)
    sns.scatterplot(data=gpu_data, x='prompt_length', y='throughput_tokens_per_s',
                    label='GPU (gemma2:2b)', s=100, alpha=0.7, ax=ax)
    
    ax.set_xlabel('Prompt Length (tokens)')
    ax.set_ylabel('Throughput (tokens/second)')
    ax.set_title('Sequential Processing: Throughput vs Prompt Length')
    ax.legend()
    plt.tight_layout()
    return fig

# ==============================================================================
# 3. Latency Under Concurrent Load (GPU Saturation Curve)
# ==============================================================================
def plot_gpu_saturation_curve():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract queue depth from decision_reason
    concurrent['queue_depth'] = concurrent['decision_reason'].str.extract(r'Queue depth: (\d+)').astype(float)
    
    # Focus on GPU requests
    gpu_concurrent = concurrent[concurrent['mode'] == 'gpu'].copy()
    
    # Group by queue depth and calculate stats
    queue_stats = gpu_concurrent.groupby('queue_depth').agg({
        'latency_s': ['mean', 'std', 'count']
    }).reset_index()
    queue_stats.columns = ['queue_depth', 'mean_latency', 'std_latency', 'count']
    
    # Plot with error bars
    ax.errorbar(queue_stats['queue_depth'], queue_stats['mean_latency'],
                yerr=queue_stats['std_latency'], marker='o', markersize=8,
                capsize=5, capthick=2, linewidth=2, label='GPU (gemma2:2b)')
    
    # Add individual points
    sns.scatterplot(data=gpu_concurrent, x='queue_depth', y='latency_s',
                    alpha=0.3, s=50, ax=ax, legend=False)
    
    ax.set_xlabel('Queue Depth (Concurrent Requests)')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('GPU Saturation Curve: Latency vs Concurrent Load')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ==============================================================================
# 4. Throughput Collapse at High Load
# ==============================================================================
def plot_throughput_collapse():
    fig, ax = plt.subplots(figsize=(12, 7))
    
    concurrent['queue_depth'] = concurrent['decision_reason'].str.extract(r'Queue depth: (\d+)').astype(float)
    
    # Separate CPU and GPU
    cpu_concurrent = concurrent[concurrent['mode'] == 'cpu'].copy()
    gpu_concurrent = concurrent[concurrent['mode'] == 'gpu'].copy()
    
    # Calculate stats
    cpu_stats = cpu_concurrent.groupby('queue_depth')['throughput_tokens_per_s'].agg(['mean', 'std']).reset_index()
    gpu_stats = gpu_concurrent.groupby('queue_depth')['throughput_tokens_per_s'].agg(['mean', 'std']).reset_index()
    
    # Plot
    ax.plot(cpu_stats['queue_depth'], cpu_stats['mean'], marker='o', linewidth=2,
            markersize=8, label='CPU (phi3)')
    ax.fill_between(cpu_stats['queue_depth'], 
                     cpu_stats['mean'] - cpu_stats['std'],
                     cpu_stats['mean'] + cpu_stats['std'], alpha=0.2)
    
    ax.plot(gpu_stats['queue_depth'], gpu_stats['mean'], marker='s', linewidth=2,
            markersize=8, label='GPU (gemma2:2b)')
    ax.fill_between(gpu_stats['queue_depth'],
                     gpu_stats['mean'] - gpu_stats['std'],
                     gpu_stats['mean'] + gpu_stats['std'], alpha=0.2)
    
    ax.set_xlabel('Queue Depth (Concurrent Requests)')
    ax.set_ylabel('Throughput (tokens/second)')
    ax.set_title('Throughput Collapse Under High Concurrent Load')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ==============================================================================
# 5. Latency Distribution (GPU-only vs Hybrid)
# ==============================================================================
def plot_latency_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # GPU-only distribution
    sns.histplot(data=gpu_only, x='latency_s', kde=True, bins=20, ax=axes[0])
    axes[0].axvline(gpu_only['latency_s'].median(), color='red', linestyle='--',
                    linewidth=2, label=f'Median: {gpu_only["latency_s"].median():.2f}s')
    axes[0].set_xlabel('Latency (seconds)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('GPU-Only Mode: Latency Distribution')
    axes[0].legend()
    
    # Hybrid distribution
    hybrid_cpu = hybrid[hybrid['mode'] == 'cpu']
    hybrid_gpu = hybrid[hybrid['mode'] == 'gpu']
    
    sns.histplot(data=hybrid_cpu, x='latency_s', kde=True, bins=15,
                 alpha=0.5, label='CPU', ax=axes[1])
    sns.histplot(data=hybrid_gpu, x='latency_s', kde=True, bins=15,
                 alpha=0.5, label='GPU', ax=axes[1])
    axes[1].set_xlabel('Latency (seconds)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Hybrid Mode: Latency Distribution by Backend')
    axes[1].legend()
    
    plt.tight_layout()
    return fig

# ==============================================================================
# 6. ECDF of Latency
# ==============================================================================
def plot_latency_ecdf():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create ECDF for each mode
    datasets = [
        (sequential, 'Sequential'),
        (gpu_only, 'GPU-Only'),
        (hybrid, 'Hybrid')
    ]
    
    for data, label in datasets:
        latencies = np.sort(data['latency_s'])
        ecdf = np.arange(1, len(latencies) + 1) / len(latencies)
        ax.plot(latencies, ecdf, marker='.', linestyle='none',
                markersize=4, alpha=0.7, label=label)
    
    # Add percentile lines
    percentiles = [50, 95, 99]
    colors = ['gray', 'orange', 'red']
    for p, c in zip(percentiles, colors):
        ax.axhline(p/100, color=c, linestyle='--', alpha=0.5,
                   linewidth=1, label=f'P{p}')
    
    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical Cumulative Distribution Function (ECDF) of Latency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ==============================================================================
# 7. Throughput Comparison
# ==============================================================================
def plot_throughput_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    comparison_data = pd.DataFrame({
        'Mode': ['Sequential', 'GPU-Only', 'Hybrid'],
        'Mean': [
            sequential['throughput_tokens_per_s'].mean(),
            gpu_only['throughput_tokens_per_s'].mean(),
            hybrid['throughput_tokens_per_s'].mean()
        ],
        'Median': [
            sequential['throughput_tokens_per_s'].median(),
            gpu_only['throughput_tokens_per_s'].median(),
            hybrid['throughput_tokens_per_s'].median()
        ],
        'Std': [
            sequential['throughput_tokens_per_s'].std(),
            gpu_only['throughput_tokens_per_s'].std(),
            hybrid['throughput_tokens_per_s'].std()
        ]
    })
    
    x = np.arange(len(comparison_data))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison_data['Mean'], width,
                   label='Mean', yerr=comparison_data['Std'], capsize=5)
    bars2 = ax.bar(x + width/2, comparison_data['Median'], width,
                   label='Median')
    
    ax.set_xlabel('Execution Mode')
    ax.set_ylabel('Throughput (tokens/second)')
    ax.set_title('Throughput Comparison Across Execution Modes')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_data['Mode'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

# ==============================================================================
# 8. Request Routing Breakdown (CPU vs GPU)
# ==============================================================================
def plot_routing_breakdown():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [
        (sequential, 'Sequential', axes[0]),
        (gpu_only, 'GPU-Only', axes[1]),
        (hybrid, 'Hybrid', axes[2])
    ]
    
    for data, title, ax in datasets:
        mode_counts = data['mode'].value_counts()
        colors = ['#ff9999', '#66b3ff']
        ax.pie(mode_counts.values, labels=mode_counts.index,
               autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title(f'{title}\n({len(data)} total requests)')
    
    plt.suptitle('Request Routing Breakdown: CPU vs GPU', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

# ==============================================================================
# 9. CPU Utilization Before/After Requests
# ==============================================================================
def plot_cpu_utilization():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    datasets = [
        (sequential, 'Sequential', axes[0, 0]),
        (concurrent, 'Concurrent', axes[0, 1]),
        (gpu_only, 'GPU-Only', axes[1, 0]),
        (hybrid, 'Hybrid', axes[1, 1])
    ]
    
    for data, title, ax in datasets:
        ax.scatter(data['cpu_util_before'], data['cpu_util_after'],
                  alpha=0.6, s=50)
        
        # Add diagonal line
        max_val = max(data['cpu_util_before'].max(), data['cpu_util_after'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('CPU Utilization Before (%)')
        ax.set_ylabel('CPU Utilization After (%)')
        ax.set_title(f'{title} Mode')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_delta = data['cpu_util_delta'].mean()
        ax.text(0.05, 0.95, f'Mean Δ: {mean_delta:.1f}%',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('CPU Utilization: Before vs After Request Processing', fontsize=14)
    plt.tight_layout()
    return fig

# ==============================================================================
# 10. GPU Utilization Before/After Requests
# ==============================================================================
def plot_gpu_utilization():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    datasets = [
        (sequential, 'Sequential', axes[0, 0]),
        (concurrent, 'Concurrent', axes[0, 1]),
        (gpu_only, 'GPU-Only', axes[1, 0]),
        (hybrid, 'Hybrid', axes[1, 1])
    ]
    
    for data, title, ax in datasets:
        scatter = ax.scatter(data['gpu_util_before'], data['gpu_util_after'],
                            c=data['latency_s'], cmap='viridis',
                            alpha=0.6, s=50)
        
        # Add diagonal line
        max_val = max(data['gpu_util_before'].max(), data['gpu_util_after'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('GPU Utilization Before (%)')
        ax.set_ylabel('GPU Utilization After (%)')
        ax.set_title(f'{title} Mode')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Latency (s)')
        
        # Add statistics
        mean_delta = data['gpu_util_delta'].mean()
        ax.text(0.05, 0.95, f'Mean Δ: {mean_delta:.1f}%',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('GPU Utilization: Before vs After Request Processing', fontsize=14)
    plt.tight_layout()
    return fig

# ==============================================================================
# BONUS PLOTS
# ==============================================================================

# 11. Memory Utilization Trends
def plot_memory_utilization():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # CPU Memory
    for i, (data, title) in enumerate([(gpu_only, 'GPU-Only'), (hybrid, 'Hybrid')]):
        ax = axes[0, i]
        ax.plot(data.index, data['cpu_mem_before_gb'], label='Before', alpha=0.7)
        ax.plot(data.index, data['cpu_mem_after_gb'], label='After', alpha=0.7)
        ax.fill_between(data.index, data['cpu_mem_before_gb'],
                        data['cpu_mem_after_gb'], alpha=0.2)
        ax.set_xlabel('Request Index')
        ax.set_ylabel('CPU Memory (GB)')
        ax.set_title(f'{title}: CPU Memory Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # GPU Memory
    for i, (data, title) in enumerate([(gpu_only, 'GPU-Only'), (hybrid, 'Hybrid')]):
        ax = axes[1, i]
        ax.plot(data.index, data['gpu_mem_util_before_pct'], label='Before', alpha=0.7)
        ax.plot(data.index, data['gpu_mem_util_after_pct'], label='After', alpha=0.7)
        ax.fill_between(data.index, data['gpu_mem_util_before_pct'],
                        data['gpu_mem_util_after_pct'], alpha=0.2)
        ax.set_xlabel('Request Index')
        ax.set_ylabel('GPU Memory Utilization (%)')
        ax.set_title(f'{title}: GPU Memory Utilization')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Memory Utilization Over Time', fontsize=14)
    plt.tight_layout()
    return fig

# 12. Latency Heatmap by Queue Depth and Prompt Length
def plot_latency_heatmap():
    concurrent['queue_depth'] = concurrent['decision_reason'].str.extract(r'Queue depth: (\d+)').astype(float)
    
    # Create bins for prompt length
    concurrent['prompt_bin'] = pd.cut(concurrent['prompt_length'],
                                      bins=[0, 5, 15, 50, 150],
                                      labels=['Tiny (0-5)', 'Small (5-15)', 'Medium (15-50)', 'Large (50+)'])
    
    pivot_data = concurrent.pivot_table(values='latency_s',
                                        index='queue_depth',
                                        columns='prompt_bin',
                                        aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Mean Latency (s)'}, ax=ax)
    ax.set_xlabel('Prompt Size Category')
    ax.set_ylabel('Queue Depth')
    ax.set_title('Latency Heatmap: Queue Depth vs Prompt Size')
    plt.tight_layout()
    return fig

# 13. Output Tokens vs Latency Scatter
def plot_output_tokens_vs_latency():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [
        (sequential, 'Sequential', axes[0]),
        (gpu_only, 'GPU-Only', axes[1]),
        (hybrid, 'Hybrid', axes[2])
    ]
    
    for data, title, ax in datasets:
        scatter = ax.scatter(data['output_tokens'], data['latency_s'],
                            c=data['prompt_length'], cmap='viridis',
                            alpha=0.6, s=50)
        ax.set_xlabel('Output Tokens')
        ax.set_ylabel('Latency (seconds)')
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label='Prompt Length')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Output Tokens vs Latency (colored by Prompt Length)', fontsize=14)
    plt.tight_layout()
    return fig

# ==============================================================================
# GENERATE ALL PLOTS
# ==============================================================================
if __name__ == "__main__":
    print("Generating all plots...")
    
    plots = [
        ("1_sequential_latency_vs_prompt.png", plot_sequential_latency_vs_prompt),
        ("2_sequential_throughput_vs_prompt.png", plot_sequential_throughput_vs_prompt),
        ("3_gpu_saturation_curve.png", plot_gpu_saturation_curve),
        ("4_throughput_collapse.png", plot_throughput_collapse),
        ("5_latency_distribution.png", plot_latency_distribution),
        ("6_latency_ecdf.png", plot_latency_ecdf),
        ("7_throughput_comparison.png", plot_throughput_comparison),
        ("8_routing_breakdown.png", plot_routing_breakdown),
        ("9_cpu_utilization.png", plot_cpu_utilization),
        ("10_gpu_utilization.png", plot_gpu_utilization),
        ("11_memory_utilization.png", plot_memory_utilization),
        ("12_latency_heatmap.png", plot_latency_heatmap),
        ("13_output_tokens_vs_latency.png", plot_output_tokens_vs_latency),
    ]
    
    for filename, plot_func in plots:
        try:
            print(f"Creating {filename}...")
            fig = plot_func()
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved {filename}")
        except Exception as e:
            print(f"✗ Error creating {filename}: {str(e)}")
    
    print("\nAll plots generated successfully!")