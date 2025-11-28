import requests
import time

API_URL = "http://127.0.0.1:8000/infer"

# ✅ Define test prompts (short, medium, long)
TEST_PROMPTS = [
    # 🔹 SHORT PROMPTS (CPU likely)
    "What is AI?",
    "Define quantum computing in simple terms.",
    "Explain photosynthesis briefly.",
    "Who invented the Internet?",
    "What is the capital of France?",

    # 🔸 MEDIUM PROMPTS (mixed CPU/GPU)
    "Describe how a neural network learns using backpropagation and gradient descent.",
    "Explain how blockchain ensures data security and integrity in distributed systems.",
    "How do self-driving cars use computer vision and deep learning to detect objects and make driving decisions?",
    "Describe how transformers differ from RNNs and CNNs in processing sequential data.",
    "Explain the process of cloud computing resource allocation and virtualization in modern data centers.",

    # 🔺 LONG / COMPLEX PROMPTS (GPU heavy)
    "Explain in detail how large language models such as GPT or Gemini are trained on massive text corpora. Discuss the architecture of the Transformer, focusing on how attention mechanisms enable context retention across long sequences. Include the process of dataset preprocessing, tokenization, pretraining objectives (e.g., next-token prediction), and fine-tuning strategies such as reinforcement learning from human feedback (RLHF). Finally, discuss how distributed training, quantization, and model parallelism are used to scale training across thousands of GPUs.",
    
    "Write a comprehensive explanation of how genetic algorithms simulate evolution to solve optimization problems. Include examples from scheduling, design, and artificial intelligence. Explain how mutation, crossover, and selection work together to evolve better solutions over generations, and discuss how the balance between exploration and exploitation affects algorithm performance.",
    
    "Discuss in depth the ethical challenges and societal impacts of AI in healthcare, including algorithmic bias, patient privacy, transparency, and accountability. Provide real-world examples of AI diagnostic tools, their successes, and the controversies they have sparked regarding trust and responsibility in medical decision-making.",
    
    "Describe the architecture, training process, and inference pipeline of a multimodal AI system that combines text, images, and speech data. Explain how cross-attention layers work in multimodal transformers and discuss the challenges of aligning different modalities in a shared latent space for coherent reasoning.",
    
    "Provide an extended overview of edge computing and its relationship to cloud AI deployment. Explain how models are optimized for latency, power efficiency, and limited memory. Include how adaptive scheduling strategies between CPU, GPU, and specialized NPUs can improve throughput in real-time edge inference scenarios."
]

print("\n🚀 Starting Phi-3 GPU test run...\n")

for prompt in TEST_PROMPTS:
    payload = {"prompt": prompt}

    start = time.time()
    response = requests.post(API_URL, json=payload)
    latency = time.time() - start

    if response.status_code == 200:
        data = response.json()
        print(
            f"✅ Prompt {len(prompt.split())} words | "
            f"Latency: {round(data['latency_s'], 2)}s | "
            f"Model: {data['selected_model']}"
        )
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

    time.sleep(2)

print("\n🎯 Phi-3 GPU test completed. Check logs.csv\n")