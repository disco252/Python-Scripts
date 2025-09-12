import os
import json
import torch
import time
import gc
import sys
import psutil
import signal
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-14B"
print("🤖 Qwen3-14B Large Prompt Chat Interface - 4-bit NF4 Optimized (Local Only)")
print("=" * 75)

# ─── System Configuration ─────────────────────────────────────────────────────
cpu_cores = psutil.cpu_count(logical=False)
available_ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"💻 CPU: {cpu_cores} cores")
print(f"💾 System RAM: {available_ram_gb:.1f} GB")

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

# ─── Multi-Line Input Handler ─────────────────────────────────────────────────
class InputCancelled(Exception):
    pass

def get_multiline_input(prompt="💭 Enter your prompt (type 'END' on new line to finish):\n"):
    """
    Read multi-line input from user until 'END' is entered on a new line.
    Properly handles Ctrl+C interruption.
    """
    print(prompt)
    lines = []
    
    def sigint_handler(signum, frame):
        raise InputCancelled()
    
    # Set up signal handler for Ctrl+C
    old_handler = signal.signal(signal.SIGINT, sigint_handler)
    
    try:
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        
        # Join all lines and clean up
        full_input = '\n'.join(lines).strip()
        return full_input
        
    except InputCancelled:
        print("\n⚠️ Input cancelled by user (Ctrl+C)")
        return ""
    except EOFError:
        # Handle case where input ends unexpectedly
        return '\n'.join(lines).strip()
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, old_handler)

# ─── Model Loading Strategy with 12GB Focus ──────────────────────────────────
print(f"🔄 Loading {MODEL_NAME} with 4-bit NF4 quantization...")
gpu_memory_gb = get_gpu_memory_gb()
if gpu_memory_gb > 0:
    print(f"💾 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory_gb:.1f} GB)")
else:
    print("⚠️ No CUDA GPU detected - will use CPU")

# Enhanced 4-bit quantization strategies optimized for 12GB
loading_strategies = [
    {
        "name": "4-bit NF4 Maximum Performance (12GB)",
        "use_gpu": True,
        "config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8,
        ),
        "max_memory": {0: "12GB", "cpu": "26GB"},
        "context_length": 18432,  # Large context for big prompts
        "description": "Maximum 12GB GPU utilization - optimal for large prompts"
    },
    {
        "name": "4-bit NF4 Balanced (11GB)",
        "use_gpu": True,
        "config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8,
        ),
        "max_memory": {0: "11GB", "cpu": "16GB"},
        "context_length": 18432,
        "description": "Balanced allocation - reliable fallback"
    },
    {
        "name": "4-bit NF4 Conservative (10GB)",
        "use_gpu": True,
        "config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8,
        ),
        "max_memory": {0: "12GB", "cpu": "26GB"},
        "context_length": 18432,
        "description": "Conservative allocation - stability focused"
    }
]

# Load tokenizer
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    use_fast=True,
    padding_side="left"
)
print("✅ Tokenizer loaded")

# Try loading model with enhanced strategy hierarchy
model = None
strategy_used = None
for i, strategy in enumerate(loading_strategies):
    print(f"\n🔄 Strategy {i+1}/{len(loading_strategies)}: {strategy['name']}")
    print(f"   ℹ️ {strategy['description']}")
    
    try:
        cleanup_gpu_memory()
        time.sleep(2)
        
        total_gpu, allocated_gpu, reserved_gpu, free_gpu = get_gpu_memory_status()
        print(f"   💾 GPU Status: {free_gpu:.1f}GB free of {total_gpu:.1f}GB total")
        print(f"   🎯 Allocation: {strategy['max_memory']}")
        print(f"   📏 Context Length: {strategy['context_length']:,} tokens")
        
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
        
        # Optional compilation for maximum performance
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                print("   🚀 Compiling model for enhanced performance...")
                model = torch.compile(model, mode="reduce-overhead")
                print("   ✅ Model compiled successfully!")
            except Exception as e:
                print(f"   ⚠️ Compilation failed (continuing anyway): {e}")
        
        strategy_used = strategy
        print(f"✅ SUCCESS with {strategy['name']}!")
        
        # Show detailed memory usage
        total_gpu, allocated_gpu, reserved_gpu, free_gpu = get_gpu_memory_status()
        print(f"   📊 Final GPU: {allocated_gpu:.1f}GB allocated, {free_gpu:.1f}GB free")
        
        # Performance prediction based on strategy
        if "12GB" in strategy['name']:
            print(f"   🚀 Expected performance: 30-45 tokens/second")
        elif "11GB" in strategy['name']:
            print(f"   ✅ Expected performance: 25-35 tokens/second")
        else:
            print(f"   📈 Expected performance: 20-30 tokens/second")
        
        break
        
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"   ❌ Failed: {error_msg}...")
        
        if "out of memory" in error_msg.lower():
            print(f"   📊 GPU memory issue - trying next strategy")
        
        cleanup_gpu_memory()
        continue

if model is None:
    print("\n❌ ALL QUANTIZATION STRATEGIES FAILED!")
    print("\n🔧 TROUBLESHOOTING:")
    print("- Close other GPU-using applications")
    print("- Restart Python to clear GPU memory")
    print("- Try Qwen2.5-14B-Instruct (more memory efficient)")
    print("- Consider Qwen2.5-7B-Instruct (guaranteed to work)")
    sys.exit(1)

cleanup_gpu_memory()

# ─── Load Repository Chunks with Enhanced Local Search ───────────────────────
HERE = Path(__file__).resolve().parent
repo_paths = [
    HERE / "repo_chunks.json",
    Path("E:/repos/repo_chunks.json"),
    Path("C:/Users/billo/repo_chunks.json")
]

repo_chunks = []
for repo_path in repo_paths:
    if repo_path.exists():
        try:
            with repo_path.open("r", encoding="utf-8") as f:
                all_chunks = json.load(f)
                # Optimize chunk count for 12GB performance
                repo_chunks = all_chunks[:2000] if len(all_chunks) > 2000 else all_chunks
            print(f"📚 Loaded {len(repo_chunks)} repository chunks from {repo_path}")
            break
        except Exception as e:
            print(f"⚠️ Error loading {repo_path}: {e}")
            continue

if not repo_chunks:
    print("⚠️ No repo_chunks.json found - proceeding without repository context")

def find_relevant_repo_chunks(query, top_k=8):
    """Enhanced local repository search with better scoring"""
    if not repo_chunks:
        return []
    
    query_lower = query.lower()
    query_words = query_lower.split()
    scored_chunks = []
    
    for chunk in repo_chunks[:1750]:  # Search more chunks for better results
        text = chunk.get("text", "").lower()
        filename = chunk.get("filename", "").lower()
        
        # Enhanced scoring algorithm
        text_score = 0
        filename_score = 0
        
        # Word-based scoring
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                text_score += text.count(word)
                filename_score += filename.count(word) * 3  # Higher weight for filename matches
        
        # Phrase-based scoring
        if len(query_lower) > 3:
            text_score += text.count(query_lower) * 2
            filename_score += filename.count(query_lower) * 5
        
        total_score = text_score + filename_score
        
        if total_score > 0:
            scored_chunks.append((total_score, chunk))
    
    # Sort by relevance and return top chunks
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    
    results = []
    for score, chunk in scored_chunks[:top_k]:
        repo = chunk.get("repo", "Unknown")
        filename = chunk.get("filename", "Unknown")
        text = chunk.get("text", "")[:800]  # Increased context for 12GB config
        results.append({
            "repo": repo,
            "filename": filename,
            "text": text,
            "score": score
        })
    
    return results

# ─── Enhanced LLM Wrapper for 12GB Performance ───────────────────────────────
class Optimized12GBQwen3Chat:
    def __init__(self, model, tokenizer, strategy):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.strategy = strategy
        self.is_gpu = strategy.get("use_gpu", False)
        self.context_length = strategy.get("context_length", 18432)
        
        # Fixed token limits calculation - convert to string first
        gpu_allocation_raw = strategy["max_memory"].get(0, "10GB")
        gpu_allocation = str(gpu_allocation_raw)  # Ensure it's a string
        gpu_gb = int(gpu_allocation.replace("GB", ""))
        
        # Optimized token limits for 12GB configuration with large prompt support
        if gpu_gb >= 12:
            self.max_thinking = 4608   # Fixed syntax error from original
            self.max_fast = 3072      # Reduced for stability
            self.performance_tier = "Maximum"
        elif gpu_gb >= 11:
            self.max_thinking = 3560
            self.max_fast = 2048
            self.performance_tier = "High"
        elif gpu_gb >= 10:
            self.max_thinking = 3072
            self.max_fast = 1536
            self.performance_tier = "Good"
        else:
            self.max_thinking = 3560
            self.max_fast = 2280
            self.performance_tier = "Basic"
            
        print(f"🔧 Large-prompt-optimized LLM ready with {strategy['name']}")
        print(f"   🧠 Max thinking tokens: {self.max_thinking:,}")
        print(f"   ⚡ Max fast tokens: {self.max_fast:,}")
        print(f"   📏 Context length: {self.context_length:,}")
        print(f"   🎯 Performance tier: {self.performance_tier}")
        
    def generate_response(self, messages, enable_thinking=True, max_new_tokens=None):
        try:
            start_time = time.time()
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            
            # Show token count for large prompts
            input_tokens = len(self.tokenizer.encode(text))
            print(f"📊 Input tokens: {input_tokens:,} (max context: {self.context_length:,})")
            
            # Enhanced tokenization with large prompt support
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.context_length,
                return_attention_mask=True
            ).to(self.device)
            
            target_tokens = max_new_tokens or (
                self.max_thinking if enable_thinking else self.max_fast
            )
            
            # Optimized generation parameters for 12GB setup with large prompts
            if enable_thinking:
                gen_params = {
                    "max_new_tokens": target_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 30,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": 1.05,
                    "use_cache": True,
                }
            else:
                gen_params = {
                    "max_new_tokens": target_tokens,
                    "temperature": 0.8,
                    "top_p": 0.85,
                    "top_k": 40,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": 1.02,
                    "use_cache": True,
                }
            
            # High-performance generation
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        outputs = self.model.generate(**inputs, **gen_params)
                else:
                    outputs = self.model.generate(**inputs, **gen_params)
            
            # Decode response
            new_tokens = outputs[0][len(inputs.input_ids[0]):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Performance metrics
            end_time = time.time()
            tokens_generated = len(new_tokens)
            elapsed = end_time - start_time
            tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0
            
            performance_emoji = "🚀" if tokens_per_second > 25 else "✅" if tokens_per_second > 15 else "📊"
            print(f"\n{performance_emoji} Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_per_second:.1f} tok/s)")
            
            # Cleanup
            del outputs, new_tokens, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            print("❌ GPU out of memory - reducing token limits")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Adaptive reduction - fixed syntax error
            if enable_thinking:
                self.max_thinking = max(2048, int(self.max_thinking * 0.7))
                print(f"   🔄 Reduced thinking tokens to {self.max_thinking:,}")
            else:
                self.max_fast = max(512, int(self.max_fast * 0.7))
                print(f"   🔄 Reduced fast tokens to {self.max_fast:,}")
            
            return "GPU memory exhausted. Token limits reduced. Please try your question again."
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"Generation error: {str(e)}"

# Initialize enhanced chat
chat = Optimized12GBQwen3Chat(model, tokenizer, strategy_used)
print("🚀 Large-prompt-optimized local chat initialized!")

# ─── Helper Functions ─────────────────────────────────────────────────────────
def format_thinking_response(response):
    """Format thinking responses with clear separation"""
    if "<think>" in response:
        parts = response.split("</think>")
        if len(parts) > 1:
            thinking_part = parts[0].replace("<think>", "").strip()
            final_part = parts[1].strip()
            if thinking_part:
                return f"\n🧠 REASONING:\n{'─' * 60}\n{thinking_part}\n{'─' * 60}\n\n💬 RESPONSE:\n{final_part}\n"
    return f"\n💬 RESPONSE:\n{response}\n"

def format_response(response, mode, has_context=False):
    """Format response with mode indicators"""
    mode_icons = {
        "thinking": "🧠",
        "fast": "⚡",
        "repo": "📚"
    }
    
    icon = mode_icons.get(mode, "💬")
    strategy_name = strategy_used['name']
    performance_tier = chat.performance_tier
    
    context_indicator = " + Repository Context" if has_context else ""
    
    if mode == "thinking":
        prefix = f"\n{icon} THINKING MODE{context_indicator} ({strategy_name} - {performance_tier})"
    elif mode == "fast":
        prefix = f"\n{icon} FAST MODE{context_indicator} ({strategy_name} - {performance_tier})"
    elif mode == "repo":
        prefix = f"\n{icon} REPOSITORY MODE ({strategy_name} - {performance_tier})"
    else:
        prefix = f"\n💬 RESPONSE ({strategy_name} - {performance_tier})"
    
    print("=" * 75)
    print(prefix)
    print("=" * 75)
    
    if mode == "thinking":
        return format_thinking_response(response)
    else:
        return f"\n💬 RESPONSE:\n{response}\n"

# ─── Main Chat Loop with Large Prompt Support ────────────────────────────────
def main():
    print("\n" + "=" * 75)
    print("🤖 QWEN3-14B LARGE PROMPT CHAT INTERFACE - 12GB OPTIMIZED")
    print("=" * 75)
    print("🚀 Configuration:")
    print(f"   • Strategy: {strategy_used['name']}")
    print(f"   • Performance Tier: {chat.performance_tier}")
    print(f"   • GPU Memory: {strategy_used['max_memory']}")
    print(f"   • Max Thinking Tokens: {chat.max_thinking:,}")
    print(f"   • Max Fast Tokens: {chat.max_fast:,}")
    print(f"   • Repository Chunks: {len(repo_chunks):,}")
    print(f"   • Context Length: {chat.context_length:,} tokens")
    print("=" * 75)
    print("⌨️  Input Instructions:")
    print("   • Type or paste your prompt (supports multi-line)")
    print("   • Type 'END' on a new line to submit")
    print("   • Use Ctrl+C to exit the script")
    print("   • Commands: !ask, !fast, !repo, help, quit")
    print("=" * 75)
    print("Commands:")
    print("  !ask <prompt>     - Ask with reasoning (shows thinking process)")
    print("  !fast <prompt>    - Quick response without thinking")
    print("  !repo <prompt>    - Search repository context only")
    print("  help             - Show this help")
    print("  quit or exit     - Exit the program")
    print("=" * 75)
    
    while True:
        try:
            # Get multi-line input - requires typing 'END'
            user_input = get_multiline_input(f"\n💭 Enter your prompt [{chat.performance_tier}] (type 'END' to finish):")
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower().strip() in ["quit", "exit"]:
                print("\n👋 Goodbye!")
                break
                
            elif user_input.lower().strip() == "help":
                print("\n📖 Available Commands:")
                print("  !ask <prompt>     - Full reasoning mode (shows thought process)")
                print("  !fast <prompt>    - Quick response mode (faster)")
                print("  !repo <prompt>    - Repository context search only")
                print("  help             - Show this help")
                print("  quit or exit     - Exit the program")
                print(f"\n🔧 Current Configuration:")
                print(f"  • Strategy: {strategy_used['name']}")
                print(f"  • Performance: {chat.performance_tier} tier")
                print(f"  • Context: {chat.context_length:,} tokens")
                print(f"  • Repository: {len(repo_chunks):,} chunks available")
                print(f"\n⌨️  Input Tips:")
                print(f"  • Type multi-line prompts, then 'END' to submit")
                print(f"  • Paste large content, then type 'END'")
                print(f"  • Use Ctrl+C to exit anytime")
                print(f"  • Commands work across multiple lines too")
                continue
            
            # Parse command from multi-line input
            enable_thinking = True
            mode = "thinking"
            user_query = user_input
            
            # Check for commands at the start of the input
            first_line = user_input.split('\n')[0].strip()
            
            if first_line.startswith("!ask "):
                user_query = user_input[5:].strip()  # Remove !ask from entire input
                enable_thinking = True
                mode = "thinking"
                
            elif first_line.startswith("!fast "):
                user_query = user_input[6:].strip()  # Remove !fast from entire input
                enable_thinking = False
                mode = "fast"
                
            elif first_line.startswith("!repo "):
                user_query = user_input[6:].strip()  # Remove !repo from entire input
                mode = "repo"
            
            if not user_query.strip():
                print("⚠️ Please provide a prompt after the command.")
                continue
            
            # Show prompt statistics
            word_count = len(user_query.split())
            char_count = len(user_query)
            
            if char_count > 100:  # Show stats for larger prompts
                estimated_tokens = len(tokenizer.encode(user_query[:4000]))  # Estimate on first 4k chars
                print(f"\n🔄 Processing prompt...")
                print(f"   📊 Characters: {char_count:,}")
                print(f"   📊 Words: {word_count:,}")
                print(f"   📊 Estimated tokens: {estimated_tokens:,}")
                print(f"   🎯 Mode: {mode}")
            else:
                print(f"\n🔄 Processing: {user_query[:60]}{'...' if len(user_query) > 60 else ''}")
            
            # Handle repo-only mode
            if mode == "repo":
                relevant_chunks = find_relevant_repo_chunks(user_query, top_k=8)
                if relevant_chunks:
                    response = f"**Repository Search Results:**\n\n"
                    for i, chunk in enumerate(relevant_chunks, 1):
                        response += f"**{i}. [{chunk['repo']}/{chunk['filename']}]** (Score: {chunk['score']})\n"
                        response += f"{chunk['text'][:1000]}...\n\n"
                    formatted_response = format_response(response, "repo")
                    print(formatted_response)
                else:
                    print("📚 No relevant repository context found for your prompt.")
                continue
            
            # Build system prompt with repository context
            system_prompt = (
                "You are Qwen3, an advanced AI assistant with comprehensive reasoning capabilities. "
                "Provide clear, accurate, and detailed responses to prompts of any size. "
                "When thinking mode is enabled, show your reasoning process step by step. "
                "Handle complex multi-part questions and long documents effectively."
            )
            
            # Add repository context
            relevant_chunks = find_relevant_repo_chunks(user_query, top_k=8)
            has_context = len(relevant_chunks) > 0
            
            if relevant_chunks:
                context_text = "\n\nRepository Context:\n"
                for chunk in relevant_chunks:
                    context_text += f"\n[{chunk['repo']}/{chunk['filename']}]:\n{chunk['text']}\n"
                system_prompt += context_text
                print(f"   📚 Added context from {len(relevant_chunks)} repository chunks")
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
            
            # Generate response
            print(f"🤖 Generating response ({mode} mode, tier: {chat.performance_tier})...")
            response = chat.generate_response(
                messages,
                enable_thinking=enable_thinking,
                max_new_tokens=chat.max_thinking if enable_thinking else chat.max_fast
            )
            
            # Display response
            formatted_response = format_response(response, mode, has_context)
            print(formatted_response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Memory cleanup completed")