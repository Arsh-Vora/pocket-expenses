import os

class Settings:
    # API Configurations
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_KEY_HERE")
    
    # Model Architecture
    BASE_MODEL = "unsloth/qwen2.5-1.5b-instruct-bnb-4bit"
    MAX_SEQ_LENGTH = 512
    LORA_R = 32
    LORA_ALPHA = 64
    
    # Training Hyperparameters
    BATCH_SIZE = 1
    GRAD_ACCUM = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-4
    
    # Paths
    DATA_PATH = "data/data.jsonl"
    OUTPUT_DIR = "outputs/financial_slm_final"