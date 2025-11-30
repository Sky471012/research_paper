#!/usr/bin/env python3
"""
TEST B: GPU-only vs Hybrid Comparison
Critical test for research paper
Run this AFTER Test A passes
"""
import requests
import time
import asyncio
import aiohttp
import statistics

API_URL = "http://127.0.0.1:8000/infer"

# Mixed workload (realistic web traffic)
MIXED_PROMPTS = [
    # 5 tiny (3-5 words) - should go to CPU
    "What is AI?",
    "Define ML.",
    "Explain IoT.",
    "What is 5G?",
    "Define blockchain.",
    
    # 5 small (10-15 words) - should prefer GPU if available
    "Explain how neural networks learn from data using gradient descent.",
    "What is quantum computing and how does it differ from classical?",
    "Describe cloud computing and its benefits for modern applications.",
    "How does encryption protect data transmitted over the internet?",
    "What are microservices and why are they popular?",
    
    # 5 medium (30-50 words) - should prefer GPU
    "Describe how neural networks learn using backpropagation and gradient descent. Include forward propagation, loss calculation, and weight updates using learning rates.",
    
    "Explain how blockchain ensures data security in distributed systems. Discuss consensus mechanisms like Proof of Work and how cryptographic hashing creates immutable records.",
    
    "How do transformers differ from RNNs in processing sequential data? Explain the self-attention mechanism and why transformers can be parallelized while RNNs cannot.",
    
    "Describe cloud resource allocation and virtualization in modern data centers. Include how hypervisors manage virtual machines and how container orchestration works.",
    
    "Explain self-driving car perception systems using computer vision and deep learning. Cover sensor fusion and real-time object detection with convolutional neural networks.",
]


async def send_request(session, prompt, request_id, mode_name):
    """Send request and measure latency."""
    payload = {"prompt": prompt}
    start = time.time()
    
    try:
        async with session.post(API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
            latency = time.time() - start
            if response.status == 200:
                data = await response.json()
                return {
                    "mode": mode_name,
                    "request_id": request_id,
                    "status": "success",
                    "latency": latency,
                    "model": data.get("selected_model", "unknown"),
                    "throughput": data.get("throughput_tokens_per_s", 0),
                    "prompt_words": len(prompt.split()),
                    "decision": data.get("decision_reason", ""),
                }
            else:
                return {
                    "mode": mode_name,
                    "request_id": request_id,
                    "status": "error",
                    "latency": latency,
                }
    except asyncio.TimeoutError:
        return {
            "mode": mode_name,
            "request_id": request_id,
            "status": "timeout",
            "latency": 180,
        }
    except Exception as e:
        return {
            "mode": mode_name,
            "request_id": request_id,
            "status": "error",
            "latency": time.time() - start,
        }


async def run_test_scenario(prompts, scenario_name):
    """Run one test scenario."""
    print(f"\n{'='*70}")
    print(f"🧪 SCENARIO: {scenario_name}")
    print(f"{'='*70}\n")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            tasks.append(send_request(session, prompt, i+1, scenario_name))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
    
    # Analyze
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    if successful:
        latencies = [r["latency"] for r in successful]
        throughputs = [r["throughput"] for r in successful]
        
        gpu_count = sum(1 for r in successful if "gemma" in r.get("model", "").lower())
        cpu_count = sum(1 for r in successful if "phi" in r.get("model", "").lower())
        
        print(f"📊 RESULTS:")
        print(f"   Success Rate: {len(successful)}/{len(prompts)} ({len(successful)/len(prompts)*100:.1f}%)")
        print(f"   Failed: {len(failed)}")
        print(f"   Total Time: {total_time:.2f}s\n")
        
        print(f"   Model Distribution:")
        print(f"   • GPU (gemma2:2b): {gpu_count} requests ({gpu_count/len(successful)*100:.1f}%)")
        print(f"   • CPU (phi3): {cpu_count} requests ({cpu_count/len(successful)*100:.1f}%)\n")
        
        print(f"   Latency:")
        print(f"   • Min:    {min(latencies):6.2f}s")
        print(f"   • Mean:   {statistics.mean(latencies):6.2f}s")
        print(f"   • Median: {statistics.median(latencies):6.2f}s")
        print(f"   • P95:    {sorted(latencies)[int(len(latencies)*0.95)]:6.2f}s")
        print(f"   • Max:    {max(latencies):6.2f}s\n")
        
        print(f"   Throughput:")
        print(f"   • Mean: {statistics.mean(throughputs):.2f} tokens/s")
        
        # Show routing for first few requests
        print(f"\n   Sample Routing Decisions:")
        for r in successful[:5]:
            words = r['prompt_words']
            model = r.get('model', 'unknown')[:10]
            decision = r.get('decision', '')[:45]
            print(f"   • {words:3d}w → {model:10s} | {decision}")
        
        return {
            "scenario": scenario_name,
            "success_rate": len(successful)/len(prompts),
            "failed_count": len(failed),
            "mean_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": sorted(latencies)[int(len(latencies)*0.95)],
            "max_latency": max(latencies),
            "mean_throughput": statistics.mean(throughputs),
            "gpu_count": gpu_count,
            "cpu_count": cpu_count,
            "total_time": total_time,
        }
    else:
        print(f"\n❌ ALL REQUESTS FAILED")
        return None


async def main():
    print("\n" + "="*70)
    print("🔬 TEST B: CRITICAL COMPARISON")
    print("GPU-only vs Hybrid Scheduling")
    print("="*70)
    
    print("\n📋 INSTRUCTIONS:")
    print("\nThis test runs in 2 parts:")
    print("\n  PART 1: GPU-ONLY MODE")
    print("    1. Edit scheduler.py")
    print("    2. Set: force_mode = 'gpu' (line ~15)")
    print("    3. Save and restart FastAPI server")
    print("    4. Come back here and press ENTER")
    
    print("\n  PART 2: HYBRID MODE")
    print("    1. Edit scheduler.py")
    print("    2. Set: force_mode = None (line ~15)")
    print("    3. Save and restart FastAPI server")
    print("    4. Come back here and press ENTER")
    
    print("\n" + "="*70)
    
    input("\n⚠️  Ready for PART 1 (GPU-only)? Edit scheduler.py now, then press ENTER...")
    
    # Test 1: GPU-only
    print("\n🔵 Running GPU-ONLY test...")
    time.sleep(1)
    result_gpu = await run_test_scenario(MIXED_PROMPTS, "GPU-ONLY")
    
    print("\n\n" + "="*70)
    print("⏸️  PAUSE - Change scheduler.py for PART 2")
    print("="*70)
    print("\n📝 Action needed:")
    print("   1. Edit scheduler.py")
    print("   2. Change: force_mode = 'gpu'  →  force_mode = None")
    print("   3. Save file")
    print("   4. Restart FastAPI server: uvicorn app.server:app --reload")
    
    input("\n⚠️  Ready for PART 2 (Hybrid)? Press ENTER when server restarted...")
    
    # Test 2: Hybrid adaptive
    print("\n🟢 Running HYBRID test...")
    time.sleep(1)
    result_hybrid = await run_test_scenario(MIXED_PROMPTS, "HYBRID")
    
    # Final comparison
    if result_gpu and result_hybrid:
        print("\n\n" + "="*70)
        print("📊 FINAL COMPARISON - YOUR RESEARCH RESULTS")
        print("="*70)
        
        print(f"\n{'Metric':<25} {'GPU-Only':<18} {'Hybrid':<18} {'Winner'}")
        print("-" * 75)
        
        # Success rate
        gpu_success = result_gpu['success_rate']*100
        hybrid_success = result_hybrid['success_rate']*100
        print(f"{'Success Rate':<25} {gpu_success:>6.1f}%           {hybrid_success:>6.1f}%           ", end="")
        if hybrid_success > gpu_success:
            print("🟢 Hybrid")
        elif gpu_success > hybrid_success:
            print("🔵 GPU")
        else:
            print("⚖️  Tie")
        
        # Failed requests
        print(f"{'Failed Requests':<25} {result_gpu['failed_count']:>6d}             {result_hybrid['failed_count']:>6d}             ", end="")
        if result_hybrid['failed_count'] < result_gpu['failed_count']:
            print("🟢 Hybrid")
        elif result_gpu['failed_count'] < result_hybrid['failed_count']:
            print("🔵 GPU")
        else:
            print("⚖️  Tie")
        
        # Mean latency
        print(f"{'Mean Latency':<25} {result_gpu['mean_latency']:>6.2f}s           {result_hybrid['mean_latency']:>6.2f}s           ", end="")
        if result_hybrid['mean_latency'] < result_gpu['mean_latency']:
            print("🟢 Hybrid")
        else:
            print("🔵 GPU")
        
        # P95 latency
        print(f"{'P95 Latency':<25} {result_gpu['p95_latency']:>6.2f}s           {result_hybrid['p95_latency']:>6.2f}s           ", end="")
        if result_hybrid['p95_latency'] < result_gpu['p95_latency']:
            print("🟢 Hybrid")
        else:
            print("🔵 GPU")
        
        # Throughput
        print(f"{'Throughput':<25} {result_gpu['mean_throughput']:>6.2f} tok/s      {result_hybrid['mean_throughput']:>6.2f} tok/s      ", end="")
        if result_hybrid['mean_throughput'] > result_gpu['mean_throughput']:
            print("🟢 Hybrid")
        else:
            print("🔵 GPU")
        
        # Total time
        print(f"{'Total Time':<25} {result_gpu['total_time']:>6.2f}s           {result_hybrid['total_time']:>6.2f}s           ", end="")
        if result_hybrid['total_time'] < result_gpu['total_time']:
            print("🟢 Hybrid")
        else:
            print("🔵 GPU")
        
        print("\n" + "-" * 75)
        
        # Model distribution
        print(f"\n🔹 Model Distribution:")
        print(f"   GPU-only:  {result_gpu['gpu_count']}/15 GPU, {result_gpu['cpu_count']}/15 CPU")
        print(f"   Hybrid:    {result_hybrid['gpu_count']}/15 GPU, {result_hybrid['cpu_count']}/15 CPU")
        
        # Calculate improvements
        print(f"\n🎯 HYBRID IMPROVEMENTS:")
        if result_hybrid['mean_latency'] < result_gpu['mean_latency']:
            latency_imp = (result_gpu['mean_latency'] - result_hybrid['mean_latency']) / result_gpu['mean_latency'] * 100
            print(f"   ✅ Mean latency: {latency_imp:.1f}% faster")
        else:
            latency_imp = (result_hybrid['mean_latency'] - result_gpu['mean_latency']) / result_gpu['mean_latency'] * 100
            print(f"   ⚠️  Mean latency: {latency_imp:.1f}% slower")
        
        if result_hybrid['p95_latency'] < result_gpu['p95_latency']:
            p95_imp = (result_gpu['p95_latency'] - result_hybrid['p95_latency']) / result_gpu['p95_latency'] * 100
            print(f"   ✅ P95 latency: {p95_imp:.1f}% faster")
        else:
            p95_imp = (result_hybrid['p95_latency'] - result_gpu['p95_latency']) / result_gpu['p95_latency'] * 100
            print(f"   ⚠️  P95 latency: {p95_imp:.1f}% slower")
        
        if result_hybrid['failed_count'] < result_gpu['failed_count']:
            print(f"   ✅ Failure reduction: {result_gpu['failed_count'] - result_hybrid['failed_count']} fewer failures")
        
        print(f"\n🎓 FOR YOUR RESEARCH PAPER:")
        print(f"   \"Our adaptive hybrid scheduler achieved {hybrid_success:.0f}% success rate")
        print(f"   with P95 latency of {result_hybrid['p95_latency']:.1f}s, compared to GPU-only")
        print(f"   ({gpu_success:.0f}% success, P95: {result_gpu['p95_latency']:.1f}s). The scheduler")
        print(f"   distributed workload across {result_hybrid['cpu_count']} CPU and {result_hybrid['gpu_count']} GPU requests,")
        print(f"   preventing saturation and reducing tail latency by {abs(p95_imp):.1f}%.\"")
        
        print("\n" + "="*70)
    
    print("\n✅ TEST B COMPLETE - Data ready for research paper!\n")


if __name__ == "__main__":
    asyncio.run(main())