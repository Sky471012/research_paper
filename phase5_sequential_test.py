#!/usr/bin/env python3
"""
Sequential test for baseline comparison.
Use this to establish baseline GPU-only performance before testing concurrency.
"""
import requests
import time

API_URL = "http://127.0.0.1:8000/infer"

# ✅ Balanced test prompts (realistic distribution)
TEST_PROMPTS = [
    # Tiny prompts (should go to CPU)
    "What is AI?",
    "Define ML.",
    "Explain IoT.",
    
    # Small prompts
    "Explain neural networks briefly.",
    "What is quantum computing?",
    "Describe cloud computing basics.",
    
    # Medium prompts
    "Describe how neural networks learn using backpropagation.",
    "Explain how blockchain ensures data security in distributed systems.",
    "How do transformers differ from RNNs in processing sequential data?",
    
    # Large prompt
    "Explain in detail how large language models like GPT are trained. Discuss the Transformer architecture, attention mechanisms, dataset preprocessing, tokenization, and fine-tuning strategies like RLHF.",
]

print("\n🚀 Starting SEQUENTIAL baseline test...\n")
print("This test sends requests one-by-one to measure baseline performance.\n")

for i, prompt in enumerate(TEST_PROMPTS, 1):
    payload = {"prompt": prompt}
    
    print(f"[{i}/{len(TEST_PROMPTS)}] Sending request... ", end="", flush=True)
    
    start = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=180)
        latency = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(
                f"✅ {data['selected_model']:7s} | "
                f"{data['latency_s']:5.2f}s | "
                f"{data['throughput_tokens_per_s']:5.1f} tok/s | "
                f"{len(prompt.split()):3d} words"
            )
        else:
            print(f"❌ Error {response.status_code}")
    
    except Exception as e:
        print(f"❌ {str(e)}")
    
    time.sleep(1)  # Small delay between requests

print("\n✅ Sequential test completed. Check logs.csv\n")