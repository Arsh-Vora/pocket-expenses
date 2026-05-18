import os
from config.settings import Settings

# 1. Force heavy cache generations onto designated storage arrays
os.environ["HF_HOME"] = "D:\\HF_Cache"
os.environ["TEMP"] = "D:\\temp"
os.environ["TMP"] = "D:\\temp"

os.makedirs("D:\\temp", exist_ok=True)
os.makedirs("D:\\HF_Cache", exist_ok=True)

# 2. Import Unsloth securely after variables are bound
from unsloth import FastLanguageModel

print("Loading local checkpoint weights for GGUF compilation...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = Settings.OUTPUT_DIR,            # Fixed configuration reference
    max_seq_length = Settings.MAX_SEQ_LENGTH,    # Fixed the 10x mismatch
    load_in_4bit = True
)

print("Starting isolated GGUF export entirely on storage array...")
model.save_pretrained_gguf(
    f"{Settings.OUTPUT_DIR}_gguf", 
    tokenizer, 
    quantization_method = "q4_k_m"
)
print("Export complete!")