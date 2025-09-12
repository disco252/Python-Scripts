import os
import discord
import asyncio
import torch
import time
import requests
import gc
import psutil
from pathlib import Path

# ─── Performance Optimizations ────────────────────────────────────────────────
# Disable verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(psutil.cpu_count(logical=False))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging
)
logging.set_verbosity_error()

print("🤖 DeepSeek-R1 High-Performance Discord Bot")
print("=" * 50)

# ─── Dependencies ──────────────────────────────────────────────────────────────
try:
    from duckduckgo_search import DDGS
    print("✅ DuckDuckGo search available")
except ImportError:
    print("⚠️ DuckDuckGo search not available")
    DDGS = None

# ─── System Info ───────────────────────────────────────────────────────────────
cpu_cores = psutil.cpu_count(logical=False)
available_ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"💻 CPU: {cpu_cores} cores | RAM: {available_ram_gb:.1f} GB")

# ─── Discord Setup ──────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)
CHANNEL_ID = YOUR DISCORD CHANNEL ID

# ─── Perplexity Config ─────────────────────────────────────────────────────────
PERPLEXITY_API_KEY = "YOUR API KEY"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

async def perplexity_check(query):
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "sonar-pro", "messages": [{"role": "user", "content": query}], "max_tokens": 1000, "temperature": 0.2}
    
    loop = asyncio.get_event_loop()
    resp = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=30)),
        timeout=35.0
    )
    return resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else f"API error {resp.status_code}"

# ─── GPU Memory Management ──────────────────────────────────────────────────────
def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_gpu_status():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        return total, allocated, total - allocated
    return 0, 0, 0

# ─── Optimized Model Loading ────────────────────────────────────────────────────
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
print("🔄 Loading optimized DeepSeek-R1...")

gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
if gpu_memory_gb > 0:
    print(f"💾 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory_gb:.1f} GB)")

# Optimized quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.uint8,
)

cleanup_gpu()

# Load tokenizer
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)

# Load and optimize model
print("🔄 Loading and compiling model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map={"": 0},  # Pin to GPU 0
    trust_remote_code=True,
    torch_dtype=torch.float16,
    max_memory={0: "11GB", "cpu": "20GB"}
)
model.eval()

# Compile model for performance (PyTorch 2.0+)
try:
    compiled_model = torch.compile(model, mode="reduce-overhead")
    print("✅ Model compiled with torch.compile")
except Exception as e:
    compiled_model = model
    print("⚠️ torch.compile not available, using standard model")

# Pre-warm model
print("🔥 Pre-warming model...")
dummy_input = tokenizer("Hello", return_tensors="pt", padding=True).to(compiled_model.device)
with torch.no_grad():
    compiled_model.generate(**dummy_input, max_new_tokens=5, do_sample=False)
del dummy_input

total_gpu, allocated_gpu, free_gpu = get_gpu_status()
print(f"✅ Model ready! GPU: {allocated_gpu:.1f}GB/{total_gpu:.1f}GB")

cleanup_gpu()

# ─── High-Performance Inference Wrapper ────────────────────────────────────────
class OptimizedDeepSeekR1:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.max_thinking = 4608
        self.max_fast = 2280
        
        # Pre-compile chat template for reuse
        self.system_msg = "You are DeepSeek R1, an advanced AI assistant with reasoning capabilities. Answer clearly, accurately, and comprehensively."
        
        print(f"🔧 Optimized LLM ready | Thinking: {self.max_thinking:,} | Fast: {self.max_fast:,}")
    
    def generate(self, messages, thinking=True):
        start_time = time.time()
        
        # Build prompt (minimize string operations)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize with non-blocking transfer
        inputs = self.tokenizer(text, return_tensors="pt", max_length=32768, truncation=True)
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        
        # Optimized generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_thinking if thinking else self.max_fast,
            "do_sample": True,
            "temperature": 0.7 if thinking else 0.9,
            "top_p": 0.95 if thinking else 0.9,
            "top_k": 20 if thinking else 40,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
            "use_cache": True,
        }
        
        # Generate with minimal overhead
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode efficiently
        new_tokens = outputs[0][len(inputs['input_ids'][0]):]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Performance tracking (minimal)
        elapsed = time.time() - start_time
        tok_per_sec = len(new_tokens) / elapsed if elapsed > 0 else 0
        
        # Fast cleanup
        del outputs, new_tokens, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response, tok_per_sec

# Initialize optimized LLM
llm = OptimizedDeepSeekR1(compiled_model, tokenizer)

# ─── Cached Search (Optimized) ──────────────────────────────────────────────────
web_cache = {}
CACHE_TTL = 86400

async def fast_ddg_search(query, max_results=5):
    if not DDGS:
        return ""
    
    # Check cache first
    cache_key = query.lower().strip()
    now = time.time()
    if cache_key in web_cache and now - web_cache[cache_key][0] < CACHE_TTL:
        return web_cache[cache_key][1]
    
    # Search and cache
    try:
        ddg = DDGS()
        results = [f"{i+1}. {r['title']}\n{r['body']}" for i, r in enumerate(ddg.text(query, max_results=max_results)) if r.get('title') and r.get('body')]
        result_text = "\n\n".join(results)
        web_cache[cache_key] = (now, result_text)
        
        # Trim cache if too large
        if len(web_cache) > 8192:
            oldest = min(web_cache.keys(), key=lambda k: web_cache[k][0])
            del web_cache[oldest]
        
        return result_text
    except:
        return ""

async def fast_serp_search(query, max_results=3):
    try:
        key = os.getenv("SERPAPI_KEY")
        if not key:
            return ""
        params = {"engine": "google", "q": query, "api_key": key, "num": max_results}
        loop = asyncio.get_event_loop()
        resp = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: requests.get("https://serpapi.com/search", params=params, timeout=10)),
            timeout=15.0
        )
        data = resp.json()
        results = [f"{i+1}. {r['title']}\n{r['snippet']}" for i, r in enumerate(data.get("organic_results", [])[:max_results]) if r.get("title")]
        if data.get("answer_box", {}).get("answer"):
            results.insert(0, f"Quick Answer: {data['answer_box']['answer']}")
        return "\n\n".join(results)
    except:
        return ""

# ─── Optimized Discord Helpers ──────────────────────────────────────────────────
async def fast_send(ch, txt):
    if not txt:
        await ch.send("No response.")
        return
    
    # Optimized chunking
    chunks = []
    while txt:
        chunk = txt[:1900]
        if len(txt) > 1900:
            last_period = chunk.rfind('. ')
            if last_period > 950:  # Only break on period if it's reasonably far
                chunk = chunk[:last_period + 1]
        chunks.append(chunk)
        txt = txt[len(chunk):].strip()
    
    # Send chunks with minimal delay
    for i, chunk in enumerate(chunks):
        await ch.send(chunk)
        if i < len(chunks) - 1:  # Don't sleep after last chunk
            await asyncio.sleep(0.2)

def quick_needs_search(q):
    return any(w in q.lower() for w in ["current", "latest", "news", "who", "what", "when", "where", "202"])

# ─── Optimized Bot Events ───────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print("✅ High-performance bot online:", bot.user)
    channel = bot.get_channel(CHANNEL_ID)
    total_gpu, allocated_gpu, free_gpu = get_gpu_status()
    
    await channel.send(
        f"🚀 **DeepSeek-R1 High-Performance Bot Ready!**\n\n"
        f"• GPU: {allocated_gpu:.1f}GB/{total_gpu:.1f}GB | Compiled: ✅\n"
        f"• Features: DuckDuckGo, Perplexity, Optimized Inference\n\n"
        f"**Commands:** `!ask` `!fast` `!search` `!check`"
    )

@bot.event
async def on_message(message):
    if message.author == bot.user or message.channel.id != CHANNEL_ID:
        return
    
    content = message.content
    if content.startswith("!ask "):
        cmd, query = "ask", content[5:].strip()
    elif content.startswith("!fast "):
        cmd, query = "fast", content[6:].strip()
    elif content.startswith("!search "):
        cmd, query = "search", content[8:].strip()
    elif content.startswith("!check "):
        cmd, query = "check", content[7:].strip()
    else:
        return
    
    if not query:
        await message.channel.send("Please provide a question.")
        return
    
    async with message.channel.typing():
        try:
            # Perplexity bypass
            if cmd == "check":
                result = await perplexity_check(query)
                await fast_send(message.channel, f"🔍 **Perplexity:** {result}")
                return
            
            # Build context efficiently
            context = llm.system_msg
            
            # Smart search integration
            if cmd == "search" or (cmd == "ask" and quick_needs_search(query)):
                web_results = await fast_ddg_search(query, 5)
                if not web_results:
                    web_results = await fast_serp_search(query, 3)
                if web_results:
                    context += f"\n\nWeb results:\n{web_results}"
            
            # Generate response with optimized model
            messages = [{"role": "system", "content": context}, {"role": "user", "content": query}]
            thinking_mode = cmd == "ask"
            response, tok_per_sec = llm.generate(messages, thinking=thinking_mode)
            
            # Send with minimal formatting
            tag = {"ask": "🧠", "fast": "⚡", "search": "🌐"}[cmd]
            await fast_send(message.channel, f"{tag} {response}")
            
        except Exception as e:
            await message.channel.send(f"Error: {str(e)}")

# ─── Launch Optimized Bot ──────────────────────────────────────────────────────
if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("❌ Set DISCORD_TOKEN environment variable")
        exit(1)
    
    print("🚀 Launching high-performance DeepSeek-R1 bot...")
    try:
        bot.run(token, log_handler=None)  # Disable discord.py logging
    except KeyboardInterrupt:
        print("👋 Shutting down...")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        cleanup_gpu()
        print("🧹 Cleanup completed")
