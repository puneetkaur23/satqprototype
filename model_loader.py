# model_loader.py — Bulletproof model loader with auto-fallback
import torch
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Change this if you need to
# ═══════════════════════════════════════════════════════════════

# Set to True if you want to skip the AI model entirely and use mocked responses.
# Useful for hackathon demos on low-RAM machines.
MOCK_MODE = False

# Primary model (big, needs 8GB+ VRAM or 16GB+ RAM)
PRIMARY_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

# Fallback model (smaller, works on 4GB+ RAM, slower but functional)
FALLBACK_MODEL = "llava-hf/llava-1.5-7b-hf"

# ═══════════════════════════════════════════════════════════════
# MOCK MODE (Zero RAM, instant load)
# ═══════════════════════════════════════════════════════════════

class MockModel:
    """Fake model that returns plausible answers without any AI loading."""
    def __init__(self):
        self.device = "cpu"
    
    def generate(self, **kwargs):
        # Return a dummy tensor that decodes to a generic answer
        return None  # We handle this in app.py

class MockProcessor:
    """Fake processor that bypasses tokenization."""
    pass


# ═══════════════════════════════════════════════════════════════
# REAL MODEL LOADER
# ═══════════════════════════════════════════════════════════════

def load_model():
    """
    Loads the vision-language model with automatic fallback strategy:
    1. Try 4-bit GPU (fastest, needs 8GB+ VRAM)
    2. Try fp16 CPU (needs 16GB+ RAM, very slow)
    3. Try smaller LLaVA-1.5 (needs 8GB+ RAM, slow but works)
    4. Fallback to MOCK_MODE (zero RAM, instant)
    """
    
    if MOCK_MODE:
        print("🎭 MOCK MODE ENABLED — No AI model loaded.")
        print("   Responses will be generic placeholders.")
        return MockProcessor(), MockModel()
    
    print("🔄 SatQuery AI: Loading vision model...")
    
    # Detect hardware
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        gpu_mem_gb = 0
    
    print(f"🖥️  Hardware Detected: CUDA={has_cuda}, GPU_VRAM={gpu_mem_gb:.1f}GB")
    
    # ─── STRATEGY 1: 4-bit GPU (FASTEST) ───
    if has_cuda and gpu_mem_gb >= 8:
        print("✅ Strategy 1: 4-bit GPU quantization")
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = LlavaNextProcessor.from_pretrained(PRIMARY_MODEL)
            model = LlavaNextForConditionalGeneration.from_pretrained(
                PRIMARY_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("✅ Model loaded (GPU 4-bit)")
            return processor, model
        except Exception as e:
            print(f"   ⚠️  4-bit failed: {e}")
    
    # ─── STRATEGY 2: fp16 on GPU with CPU offload ───
    if has_cuda and gpu_mem_gb >= 4:
        print("⚠️  Strategy 2: fp16 with GPU+CPU offload")
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = LlavaNextProcessor.from_pretrained(PRIMARY_MODEL)
            model = LlavaNextForConditionalGeneration.from_pretrained(
                PRIMARY_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            print("✅ Model loaded (GPU+CPU mixed)")
            return processor, model
        except Exception as e:
            print(f"   ⚠️  Mixed offload failed: {e}")
    
    # ─── STRATEGY 3: CPU-only with primary model ───
    if not has_cuda:
        print("🔄 Strategy 3: CPU-only with primary model")
        print("   ⚠️  WARNING: This needs ~14GB free RAM and takes 2–5 min to load!")
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = LlavaNextProcessor.from_pretrained(PRIMARY_MODEL)
            model = LlavaNextForConditionalGeneration.from_pretrained(
                PRIMARY_MODEL,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            model = model.to("cpu")
            print("✅ Model loaded (CPU-only, primary)")
            return processor, model
        except Exception as e:
            print(f"   ⚠️  CPU primary failed: {e}")
    
    # ─── STRATEGY 4: Smaller fallback model (LLaVA-1.5) ───
    print("🔄 Strategy 4: Loading smaller fallback model (LLaVA-1.5)...")
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(FALLBACK_MODEL)
        model = LlavaForConditionalGeneration.from_pretrained(
            FALLBACK_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        if not has_cuda:
            model = model.to("cpu")
        print("✅ Model loaded (Fallback LLaVA-1.5)")
        return processor, model
    except Exception as e:
        print(f"   ❌ Fallback also failed: {e}")
    
    # ─── STRATEGY 5: MOCK MODE (last resort) ───
    print("🎭 ALL STRATEGIES FAILED — Enabling MOCK_MODE for demo...")
    print("   Set MOCK_MODE = True at the top of model_loader.py to skip this next time.")
    return MockProcessor(), MockModel()


# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    proc, mod = load_model()
    print(f"\n📍 Final device: {mod.device if hasattr(mod, 'device') else 'mock'}")
    print("🚀 Ready!")