# app/model_loader.py
import requests
import time

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "gemma3"

def run_inference(prompt: str, model: str = DEFAULT_MODEL, max_retries: int = 1):
    """
    Run inference via Ollama API.
    
    ✅ CHANGES:
    - Increased timeout to 180s (was 120s)
    - Added retry logic
    - Better error handling
    
    Args:
        prompt: Input text
        model: Model name (gemma3 or phi3)
        max_retries: Number of retry attempts
    
    Returns:
        tuple: (output_text, latency_seconds)
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    for attempt in range(max_retries):
        start = time.time()
        try:
            response = requests.post(
                OLLAMA_URL, 
                json=payload, 
                timeout=180  # ✅ CHANGED: 180s timeout (was 120s)
            )
            latency = time.time() - start
            response.raise_for_status()
            
            data = response.json()
            text = data.get("response", "").strip()
            
            # Check if response is valid
            if not text or len(text) < 5:
                if attempt < max_retries - 1:
                    print(f"⚠️ Empty response, retrying... (attempt {attempt+1}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    return "[Error: Empty response from model]", latency
            
            return text, latency
            
        except requests.exceptions.Timeout:
            latency = time.time() - start
            if attempt < max_retries - 1:
                print(f"⚠️ Timeout, retrying... (attempt {attempt+1}/{max_retries})")
                time.sleep(2)
                continue
            else:
                return f"[Error: Request timeout after {latency:.1f}s - model may be hung]", latency
                
        except requests.exceptions.ConnectionError:
            latency = time.time() - start
            return "[Error: Cannot connect to Ollama - is it running?]", latency
            
        except requests.exceptions.HTTPError as e:
            latency = time.time() - start
            return f"[Error: HTTP {e.response.status_code} from Ollama]", latency
            
        except Exception as e:
            latency = time.time() - start
            return f"[Error: {type(e).__name__}: {str(e)}]", latency
    
    # Should never reach here
    return "[Error: Max retries exceeded]", time.time() - start