#!/usr/bin/env python3
"""
Analyze logs.csv to compare GPU-only vs hybrid scheduler performance.
"""
import pandas as pd
import statistics

def analyze_logs(csv_file="logs.csv"):
    """Analyze concurrency test results."""
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Error: {csv_file} not found. Run tests first!")
        return
    
    print("\n" + "="*70)
    print("📊 CONCURRENCY TEST ANALYSIS")
    print("="*70)
    
    # Overall stats
    print(f"\n📈 Overall Statistics:")
    print(f"   Total requests: {len(df)}")
    print(f"   GPU requests: {len(df[df['mode'] == 'gpu'])}")
    print(f"   CPU requests: {len(df[df['mode'] == 'cpu'])}")
    
    # Latency analysis
    print(f"\n⏱️  Latency Analysis:")
    latencies = df['latency_s'].dropna()
    if len(latencies) > 0:
        print(f"   Min: {latencies.min():.2f}s")
        print(f"   Max: {latencies.max():.2f}s")
        print(f"   Mean: {latencies.mean():.2f}s")
        print(f"   Median: {latencies.median():.2f}s")
        print(f"   P95: {latencies.quantile(0.95):.2f}s")
        print(f"   P99: {latencies.quantile(0.99):.2f}s")
    
    # Compare GPU vs CPU performance
    gpu_df = df[df['mode'] == 'gpu']
    cpu_df = df[df['mode'] == 'cpu']
    
    if len(gpu_df) > 0 and len(cpu_df) > 0:
        print(f"\n🆚 GPU vs CPU Comparison:")
        print(f"\n   GPU (Gemma3):")
        print(f"   • Requests: {len(gpu_df)}")
        print(f"   • Mean latency: {gpu_df['latency_s'].mean():.2f}s")
        print(f"   • Mean throughput: {gpu_df['throughput_tokens_per_s'].mean():.2f} tokens/s")
        
        print(f"\n   CPU (Phi3):")
        print(f"   • Requests: {len(cpu_df)}")
        print(f"   • Mean latency: {cpu_df['latency_s'].mean():.2f}s")
        print(f"   • Mean throughput: {cpu_df['throughput_tokens_per_s'].mean():.2f} tokens/s")
    
    # Analyze by prompt length
    print(f"\n📏 Performance by Prompt Length:")
    df['prompt_category'] = pd.cut(
        df['prompt_length'], 
        bins=[0, 10, 30, 60, 1000],
        labels=['Tiny (<10)', 'Small (10-30)', 'Medium (30-60)', 'Large (>60)']
    )
    
    for category in ['Tiny (<10)', 'Small (10-30)', 'Medium (30-60)', 'Large (>60)']:
        subset = df[df['prompt_category'] == category]
        if len(subset) > 0:
            print(f"\n   {category} words:")
            print(f"   • Count: {len(subset)}")
            print(f"   • GPU: {len(subset[subset['mode']=='gpu'])}, CPU: {len(subset[subset['mode']=='cpu'])}")
            print(f"   • Mean latency: {subset['latency_s'].mean():.2f}s")
    
    # Resource utilization
    print(f"\n💻 Resource Utilization:")
    print(f"   CPU util - Before: {df['cpu_util_before'].mean():.1f}%")
    print(f"   CPU util - After: {df['cpu_util_after'].mean():.1f}%")
    print(f"   GPU util - Before: {df['gpu_util_before'].mean():.1f}%")
    print(f"   GPU util - After: {df['gpu_util_after'].mean():.1f}%")
    
    # Decision reasons
    print(f"\n🧠 Scheduling Decisions:")
    decisions = df['decision_reason'].value_counts()
    for reason, count in decisions.head(5).items():
        print(f"   • {reason[:60]}: {count}")
    
    # Check for errors/timeouts
    errors = df[df['latency_s'] >= 120]
    if len(errors) > 0:
        print(f"\n⚠️  Timeouts/Errors: {len(errors)} requests")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    analyze_logs()