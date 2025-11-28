import requests
import time
import asyncio
import aiohttp
from datetime import datetime
import statistics

API_URL = "http://127.0.0.1:8000/infer"

# ✅ REALISTIC TEST PROMPTS (optimized for CPU vs GPU comparison)
TEST_PROMPTS = {
    "tiny": [
        "What is AI?",
        "Define ML.",
        "Explain IoT.",
        "What is 5G?",
        "Define blockchain.",
    ],
    "small": [
        "Explain neural networks briefly.",
        "What is quantum computing?",
        "Describe cloud computing basics.",
        "How does encryption work?",
        "What are microservices?",
    ],
    "medium": [
        "Describe how neural networks learn using backpropagation.",
        "Explain how blockchain ensures data security in distributed systems.",
        "How do transformers differ from RNNs in processing sequential data?",
        "Describe the process of cloud resource allocation and virtualization.",
        "Explain how self-driving cars use computer vision for object detection.",
    ],
    "large": [
        "Explain in detail how large language models like GPT are trained on massive text corpora. Discuss the Transformer architecture, attention mechanisms, dataset preprocessing, tokenization, and fine-tuning strategies like RLHF. Include how distributed training and model parallelism scale across GPUs.",
        
        "Write a comprehensive explanation of genetic algorithms for optimization. Include examples from scheduling and AI, explain mutation, crossover, and selection, and discuss the balance between exploration and exploitation.",
        
        "Discuss the ethical challenges of AI in healthcare, including bias, privacy, transparency, and accountability. Provide real-world examples of AI diagnostic tools and controversies regarding trust in medical decision-making.",
    ]
}


async def send_request(session, prompt, request_id):
    """Send a single async request and measure latency."""
    payload = {"prompt": prompt}
    start = time.time()
    
    try:
        async with session.post(API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
            latency = time.time() - start
            if response.status == 200:
                data = await response.json()
                return {
                    "request_id": request_id,
                    "status": "success",
                    "latency": latency,
                    "server_latency": data.get("latency_s", 0),
                    "model": data.get("selected_model", "unknown"),
                    "throughput": data.get("throughput_tokens_per_s", 0),
                    "prompt_words": len(prompt.split()),
                    "output_tokens": data.get("output_tokens", 0),
                }
            else:
                return {
                    "request_id": request_id,
                    "status": "error",
                    "latency": latency,
                    "error": f"HTTP {response.status}"
                }
    except asyncio.TimeoutError:
        return {
            "request_id": request_id,
            "status": "timeout",
            "latency": time.time() - start,
            "error": "Request timeout (180s)"
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "status": "error",
            "latency": time.time() - start,
            "error": str(e)
        }


async def burst_test(prompts, burst_size, test_name):
    """
    Simulate burst traffic: send multiple requests concurrently.
    
    Args:
        prompts: List of prompts to use
        burst_size: Number of concurrent requests
        test_name: Name for this test scenario
    """
    print(f"\n{'='*60}")
    print(f"🚀 TEST: {test_name}")
    print(f"   Burst size: {burst_size} concurrent requests")
    print(f"{'='*60}\n")
    
    # Create burst of requests
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(burst_size):
            prompt = prompts[i % len(prompts)]  # Cycle through prompts
            tasks.append(send_request(session, prompt, i+1))
        
        # Send all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    if successful:
        latencies = [r["latency"] for r in successful]
        server_latencies = [r["server_latency"] for r in successful]
        throughputs = [r["throughput"] for r in successful]
        
        # Count models used
        gpu_count = sum(1 for r in successful if "gemma" in r["model"].lower())
        cpu_count = sum(1 for r in successful if "phi" in r["model"].lower())
        
        print(f"\n📊 RESULTS:")
        print(f"   ✅ Successful: {len(successful)}/{burst_size}")
        print(f"   ❌ Failed: {len(failed)}")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"\n   Model Distribution:")
        print(f"   🔵 GPU (Gemma3): {gpu_count} requests")
        print(f"   🟢 CPU (Phi3): {cpu_count} requests")
        print(f"\n   Latency Stats (end-to-end):")
        print(f"   • Min: {min(latencies):.2f}s")
        print(f"   • Max: {max(latencies):.2f}s")
        print(f"   • Mean: {statistics.mean(latencies):.2f}s")
        print(f"   • Median: {statistics.median(latencies):.2f}s")
        print(f"   • P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
        print(f"\n   Server Processing Time:")
        print(f"   • Mean: {statistics.mean(server_latencies):.2f}s")
        print(f"   • P95: {sorted(server_latencies)[int(len(server_latencies)*0.95)]:.2f}s")
        print(f"\n   Throughput:")
        print(f"   • Mean: {statistics.mean(throughputs):.2f} tokens/s")
    
    if failed:
        print(f"\n   ⚠️  Failed Requests:")
        for r in failed[:5]:  # Show first 5 failures
            print(f"   • Request {r['request_id']}: {r['error']}")
    
    return results


async def run_all_tests():
    """Run comprehensive concurrency tests."""
    
    print("\n" + "="*70)
    print("🧪 ADAPTIVE SCHEDULER CONCURRENCY TEST SUITE")
    print("="*70)
    
    # Test 1: Small burst with tiny prompts (CPU should handle)
    await burst_test(
        TEST_PROMPTS["tiny"],
        burst_size=5,
        test_name="Warm-up: 5 tiny prompts (CPU expected)"
    )
    await asyncio.sleep(3)
    
    # Test 2: Medium burst with small prompts
    await burst_test(
        TEST_PROMPTS["small"],
        burst_size=10,
        test_name="Test 2: 10 small prompts (mixed CPU/GPU)"
    )
    await asyncio.sleep(3)
    
    # Test 3: Large burst with medium prompts (GPU saturation test)
    await burst_test(
        TEST_PROMPTS["medium"],
        burst_size=15,
        test_name="Test 3: 15 medium prompts (GPU saturation)"
    )
    await asyncio.sleep(3)
    
    # Test 4: Extreme burst with mixed sizes
    mixed_prompts = (
        TEST_PROMPTS["tiny"] * 3 +
        TEST_PROMPTS["small"] * 2 +
        TEST_PROMPTS["medium"]
    )
    await burst_test(
        mixed_prompts,
        burst_size=20,
        test_name="Test 4: 20 mixed prompts (stress test)"
    )
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED - Check logs.csv for detailed metrics")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_all_tests())