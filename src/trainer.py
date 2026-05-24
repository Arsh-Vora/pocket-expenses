import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from config.settings import Settings

class SLMTrainer:
    
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def initialize_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=Settings.BASE_MODEL,
            max_seq_length=Settings.MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )

        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=Settings.LORA_R,
            target_modules=modules,
            lora_alpha=Settings.LORA_ALPHA,
            use_gradient_checkpointing="unsloth",
            random_state=42, 
        )

    def train(self):
        raw_dataset = load_dataset("json", data_files=Settings.DATA_PATH, split="train")
        
        formatted_dataset = raw_dataset.map(
            lambda x: {"text": self.tokenizer.apply_chat_template(x["messages"], tokenize=False)}
        )

        training_args = TrainingArguments(
            per_device_train_batch_size=Settings.BATCH_SIZE,
            gradient_accumulation_steps=Settings.GRAD_ACCUM,
            warmup_steps=5,
            num_train_epochs=Settings.EPOCHS,
            learning_rate=Settings.LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,             
            save_strategy="steps",       
            save_steps=100,
            output_dir="./checkpoints",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",    
            seed=42,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_dataset,
            dataset_text_field="text",
            max_seq_length=Settings.MAX_SEQ_LENGTH,
            dataset_num_proc=4,           
            packing=False,
            args=training_args,
        )

        print("Beginning SFT run...")
        trainer.train()

if __name__ == "__main__":
    trainer = SLMTrainer()
    trainer.initialize_model()
    trainer.train()
