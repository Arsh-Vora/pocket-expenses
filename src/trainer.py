import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from config.settings import Settings

class SLMTrainer:
    # Handles low-resource fine-tuning via Unsloth.
    
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def initialize_model(self):
        # Loads base quantized model weights.
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=Settings.BASE_MODEL,
            max_seq_length=Settings.MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )

        # Apply LoRA configurations with 4GB VRAM optimizations
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=Settings.LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=Settings.LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    def train(self):
        # Runs the supervised fine-tuning loop.
        print("Preparing dataset for training...")
        raw_dataset = load_dataset("json", data_files=Settings.DATA_PATH, split="train")
        
        # Format dataset using ChatML mapping
        def format_prompt(example):
            return {"text": self.tokenizer.apply_chat_template(example["messages"], tokenize=False)}
        
        formatted_dataset = raw_dataset.map(format_prompt)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_dataset,
            dataset_text_field="text",
            max_seq_length=Settings.MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=Settings.BATCH_SIZE,
                gradient_accumulation_steps=Settings.GRAD_ACCUM,
                warmup_steps=5,
                num_train_epochs=Settings.EPOCHS,
                learning_rate=Settings.LEARNING_RATE,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                output_dir="outputs",
                save_strategy="no",
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
            ),
        )

        print("Executing model training session...")
        trainer.train()

if __name__ == "__main__":
    trainer = SLMTrainer()
    trainer.initialize_model()
    trainer.train()