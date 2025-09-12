import os
import json
import discord
import asyncio
import torch
import time
import requests
import gc
import psutil
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

print("🤖 Qwen3-14B Discord Bot - Enhanced with Perplexity Integration")
print("=" * 65)

# ─── Additional Dependencies ──────────────────────────────────────────────────
try:
    from duckduckgo_search import DDGS
    print("✅ DuckDuckGo search available")
except ImportError:
    print("⚠️ DuckDuckGo search not available. Install with: pip install duckduckgo-search")
    DDGS = None

# ─── System Configuration ─────────────────────────────────────────────────────
cpu_cores = psutil.cpu_count(logical=False)
available_ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"💻 CPU: {cpu_cores} cores")
print(f"💾 System RAM: {available_ram_gb:.1f} GB")

# ─── Discord Setup ────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)
CHANNEL_ID = DISCORDCHANNELID

# ─── Perplexity API Configuration ─────────────────────────────────────────────
PERPLEXITY_API_KEY = "API KEY"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

async def perplexity_check(query):
    """Query Perplexity API for fact-checking and additional context"""
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate, up-to-date information with citations when possible."
                },
                {
                    "role": "user", 
                    "content": query
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9
        }
        
        print(f"🔍 Checking with Perplexity: {query[:50]}...")
        
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: requests.post(
                PERPLEXITY_URL, headers=headers, json=payload, timeout=30
            )),
            timeout=35.0
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                print("✅ Perplexity check completed")
                return content
            else:
                print("⚠️ No response from Perplexity")
                return "No response received from Perplexity."
        else:
            print(f"⚠️ Perplexity API error: {response.status_code}")
            return f"Perplexity API error: {response.status_code}"
            
    except Exception as e:
        print(f"⚠️ Perplexity check error: {e}")
        return f"Error checking with Perplexity: {str(e)}"

# ─── Search Functions ─────────────────────────────────────────────────────────
web_cache = {}
CACHE_TTL = 86400  # 1 day in seconds

async def cached_ddg_search(query, max_results=5):
    """DuckDuckGo search with caching"""
    if not DDGS:
        return ""
        
    # Check cache
    now = time.time()
    cache_key = query.lower().strip()
    if cache_key in web_cache:
        timestamp, results = web_cache[cache_key]
        if now - timestamp < CACHE_TTL:
            print(f"🔄 Using cached results for: {query[:30]}...")
            return results
    
    try:
        print(f"🔍 Searching DuckDuckGo: {query[:30]}...")
        ddg = DDGS()
        results = []
        
        # Get search results
        search_results = ddg.text(query, region='wt-wt', safesearch='moderate', max_results=max_results)
        
        for i, result in enumerate(search_results):
            title = result.get('title', '')
            body = result.get('body', '')
            if title and body:
                results.append(f"{i+1}. {title}\n{body}")
        
        formatted_results = "\n\n".join(results) if results else ""
        
        # Cache results
        web_cache[cache_key] = (now, formatted_results)
        
        # Limit cache size
        if len(web_cache) > 8192:
            oldest_key = min(web_cache.keys(), key=lambda k: web_cache[k][0])
            del web_cache[oldest_key]
        
        print(f"✅ Found {len(results)} DuckDuckGo results")
        return formatted_results
        
    except Exception as e:
        print(f"⚠️ DuckDuckGo search error: {e}")
        return ""

async def web_search(query, max_results=3):
    """Legacy SerpAPI search - kept for fallback"""
    try:
        SERPAPI_KEY = "1547c1a05bcbf634da52f5cb3cf45d81542a9fbe1cdf8a17285fff3c59a60c89"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": max_results,
            "safe": "active"
        }
        
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: requests.get(
                "https://serpapi.com/search", params=params, timeout=10
            )),
            timeout=15.0
        )
        
        data = response.json()
        results = []
        
        # Get organic results
        for i, result in enumerate(data.get("organic_results", [])[:max_results]):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            if title and snippet:
                results.append(f"{i+1}. {title}\n{snippet}")
        
        # Add quick answer if available
        if "answer_box" in data:
            answer = data["answer_box"].get("answer", "")
            if answer:
                results.insert(0, f"Quick Answer: {answer}")
        
        return "\n\n".join(results) if results else ""
        
    except Exception as e:
        print(f"⚠️ SerpAPI search error: {e}")
        return ""

# ─── GPU Memory Management ────────────────────────────────────────────────────
def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0

def get_gpu_memory_status():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = total - reserved
        return total, allocated, reserved, free
    return 0, 0, 0, 0

# ─── Model Loading Strategy ───────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-14B"

# Check GPU
gpu_memory_gb = get_gpu_memory_gb()
if gpu_memory_gb > 0:
    print(f"💾 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory_gb:.1f} GB)")
else:
    print("⚠️ No CUDA GPU detected - will use CPU")

# Proper quantization hierarchy
loading_strategies = [
    {
        "name": "4-bit NF4 (Good Quality)",
        "use_gpu": True,
        "config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        "max_memory": {0: "12GB", "cpu": "16GB"}
    },
    {
        "name": "4-bit FP4 (Maximum Compression)",
        "use_gpu": True,
        "config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="fp4",
        ),
        "max_memory": {0: "12GB", "cpu": "12GB"}
    }
]

# Load tokenizer
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    use_fast=True,
)
print("✅ Tokenizer loaded")

# Try loading model with proper strategy hierarchy
model = None
strategy_used = None
for i, strategy in enumerate(loading_strategies):
    print(f"\n🔄 Strategy {i+1}/{len(loading_strategies)}: {strategy['name']}")
    
    try:
        cleanup_gpu_memory()
        time.sleep(2)  # Let memory settle
        
        total_gpu, allocated_gpu, reserved_gpu, free_gpu = get_gpu_memory_status()
        print(f"   💾 GPU Status: {free_gpu:.1f}GB free of {total_gpu:.1f}GB total")
        print(f"   🎯 Allocation: {strategy['max_memory']}")
        
        # Base config
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "max_memory": strategy["max_memory"],
            "quantization_config": strategy["config"],
            "device_map": "auto",
            "dtype": torch.float16,
        }
        
        print("   ⏳ Loading model...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
        model.eval()
        
        strategy_used = strategy
        print(f"✅ SUCCESS with {strategy['name']}!")
        
        # Show final memory usage
        total_gpu, allocated_gpu, reserved_gpu, free_gpu = get_gpu_memory_status()
        print(f"   📊 Final GPU: {allocated_gpu:.1f}GB allocated, {free_gpu:.1f}GB free")
        
        break
        
    except Exception as e:
        error_msg = str(e)[:150]
        print(f"   ❌ Failed: {error_msg}...")
        cleanup_gpu_memory()
        continue

if model is None:
    print("\n❌ ALL QUANTIZATION STRATEGIES FAILED!")
    print("\n🔧 DIAGNOSIS:")
    print("- Qwen3-14B is too large for your 12GB RTX 3080 Ti")
    print("- Even with aggressive quantization, model + activations exceed memory")
    print("- This is a fundamental hardware limitation")
    print("\n💡 SOLUTIONS:")
    print("1. Use Qwen2.5-14B-Instruct (more memory efficient)")
    print("2. Use Qwen2.5-7B-Instruct (guaranteed to work)")
    print("3. Upgrade to 16GB+ GPU (RTX 4080/4090)")
    print("4. Use cloud GPU service")
    raise RuntimeError("Hardware insufficient for Qwen3-14B")

cleanup_gpu_memory()

# ─── Enhanced LLM Wrapper ─────────────────────────────────────────────────────
class OptimizedQwen3:
    def __init__(self, model, tokenizer, strategy):
        self.model = model
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.is_gpu = strategy.get("use_gpu", False)
        
        # Set token limits based on quantization type
        strategy_name = strategy["name"].lower()
        if "8-bit" in strategy_name:
            self.max_thinking = 3072
            self.max_fast = 1280
        elif "4-bit nf4" in strategy_name:
            self.max_thinking = 4096
            self.max_fast = 2280
        elif "4-bit fp4" in strategy_name:
            self.max_thinking = 4096
            self.max_fast = 2024
        else:
            self.max_thinking = 1536
            self.max_fast = 768
            
        print(f"🔧 LLM ready with {strategy['name']}")
        print(f"   🧠 Max thinking tokens: {self.max_thinking}")
        print(f"   ⚡ Max fast tokens: {self.max_fast}")
    
    def generate(self, messages, thinking=True, max_tokens=None):
        try:
            start_time = time.time()
            
            # Apply chat template with thinking support
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking
            )
            
            # Tokenize with appropriate context length
            max_context = 16384 if "8-bit" in self.strategy["name"] else 12288
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_context,
                truncation=True,
                return_attention_mask=True
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Generation settings
            target_tokens = max_tokens or (self.max_thinking if thinking else self.max_fast)
            
            gen_kwargs = {
                "max_new_tokens": target_tokens,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            if thinking:
                gen_kwargs.update({
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "repetition_penalty": 1.05,
                })
            else:
                gen_kwargs.update({
                    "temperature": 0.75,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repetition_penalty": 1.02,
                })
            
            # Generate with proper autocast
            with torch.no_grad():
                if self.is_gpu and torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        outputs = self.model.generate(**inputs, **gen_kwargs)
                else:
                    outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode response
            new_tokens = outputs[0][len(inputs.input_ids[0]):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Performance metrics
            end_time = time.time()
            tokens_generated = len(new_tokens)
            tokens_per_second = tokens_generated / (end_time - start_time) if end_time > start_time else 0
            
            print(f"⚡ Generated {tokens_generated} tokens in {end_time - start_time:.2f}s ({tokens_per_second:.1f} tok/s)")
            
            # Cleanup
            del outputs, inputs, new_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            print("❌ GPU OOM during generation")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "GPU out of memory during generation. Try !fast mode or shorter questions."
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"Generation failed: {str(e)}"

# Initialize LLM
llm = OptimizedQwen3(model, tokenizer, strategy_used)

# ─── Discord Helpers ──────────────────────────────────────────────────────────
async def send_message(channel, text):
    """Send message with chunking"""
    if not text:
        await channel.send("No response generated.")
        return
    
    # Simple chunking
    max_len = 1900
    while text:
        chunk = text[:max_len]
        if len(text) > max_len:
            # Try to break at sentence
            last_period = chunk.rfind('. ')
            if last_period > max_len // 2:
                chunk = chunk[:last_period + 1]
        
        await channel.send(chunk)
        text = text[len(chunk):].strip()
        
        if text:  # More chunks coming
            await asyncio.sleep(0.3)

def needs_search(query):
    indicators = ["current", "latest", "today", "now", "2024", "2025", "news", "recent", "what is", "who is", "when", "where"]
    return any(word in query.lower() for word in indicators)

# ─── Bot Events ───────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print(f"✅ Bot online: {bot.user}")
    
    # Get memory status for startup message
    total_gpu, allocated_gpu, reserved_gpu, free_gpu = get_gpu_memory_status()
    
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        features = []
        if DDGS:
            features.append("DuckDuckGo Search")
        features.append("Perplexity Fact-Check")
        
        startup_msg = (
            f"🤖 **Qwen3-14B Bot Ready with Perplexity!**\n\n"
            f"**Configuration:**\n"
            f"• Strategy: `{strategy_used['name']}`\n"
            f"• GPU Memory: `{allocated_gpu:.1f}GB/{total_gpu:.1f}GB used`\n"
            f"• System RAM: `{available_ram_gb:.1f}GB available`\n"
            f"• Features: `{', '.join(features)}`\n\n"
            f"**Commands:**\n"
            f"• `!ask <question>` - Full thinking mode + auto-search\n"
            f"• `!fast <question>` - Quick responses\n"
            f"• `!search <question>` - Force web search\n"
            f"• `!check <question>` - Perplexity fact-check & context\n\n"
            f"*Enhanced with Perplexity API for up-to-date information*"
        )
        await channel.send(startup_msg)

@bot.event
async def on_message(message):
    if message.author == bot.user or message.channel.id != CHANNEL_ID:
        return
    
    # Parse commands
    if message.content.startswith("!ask "):
        command = "ask"
        query = message.content[5:].strip()
    elif message.content.startswith("!fast "):
        command = "fast" 
        query = message.content[6:].strip()
    elif message.content.startswith("!search "):
        command = "search"
        query = message.content[8:].strip()
    elif message.content.startswith("!check "):
        command = "check"
        query = message.content[7:].strip()
    else:
        return
    
    if not query:
        await message.channel.send("Please provide a question.")
        return
    
    print(f"📨 {command.upper()}: {query[:50]}...")
    
    async with message.channel.typing():
        try:
            # Handle Perplexity check command
            if command == "check":
                print("🔍 Checking with Perplexity...")
                perplexity_result = await perplexity_check(query)
                
                response = f"🔍 **Perplexity Fact-Check** ({strategy_used['name']})\n\n{perplexity_result}"
                await send_message(message.channel, response)
                print("✅ Perplexity check sent")
                return
            
            # Build context for other commands
            context = "You are Qwen3, a helpful AI assistant with advanced reasoning capabilities. Answer clearly, accurately, and comprehensively."
            
            # Add web search for search command or auto-detected queries
            if command == "search" or (command == "ask" and needs_search(query)):
                print("🔍 Searching web...")
                
                # Try DuckDuckGo first, fallback to SerpAPI
                search_results = await cached_ddg_search(query)
                if not search_results:
                    print("🔄 Falling back to SerpAPI...")
                    search_results = await web_search(query)
                
                if search_results:
                    context += f"\n\nWeb search results:\n{search_results}"
                    print("✅ Added search results")
            
            # Generate response
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ]
            
            thinking_mode = command in ["ask", "search"]
            print(f"🤖 Generating (thinking: {thinking_mode})...")
            
            response = llm.generate(messages, thinking=thinking_mode)
            
            # Format response with thinking content
            if thinking_mode and "<think>" in response:
                parts = response.split("</think>")
                if len(parts) > 1:
                    thinking = parts[0].replace("<think>", "").strip()
                    final = parts[1].strip()
                    response = f"🧠 **Thinking:**\n``````\n\n💬 **Response:**\n{final}"
            
            # Send with mode indicator
            mode_indicators = {
                "ask": f"🧠 **Thinking Mode** ({strategy_used['name']})\n\n",
                "fast": f"⚡ **Fast Mode** ({strategy_used['name']})\n\n", 
                "search": f"🌐 **Search Mode** ({strategy_used['name']})\n\n"
            }
            
            final_response = mode_indicators.get(command, "") + response
            await send_message(message.channel, final_response)
            print("✅ Response sent")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await message.channel.send(f"Error: {str(e)}")

# ─── Main Execution ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("❌ Set DISCORD_TOKEN environment variable")
        exit(1)
    
    print(f"\n🚀 Starting bot with {strategy_used['name']} and Perplexity integration...")
    
    try:
        bot.run(token)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        cleanup_gpu_memory()
        print("🧹 Cleanup done")