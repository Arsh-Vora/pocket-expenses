import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.settings import Settings


MOCK_TEST_CASES = [
    {"input": "Whatsapp Group: I spent $3000 on a vacation guys!", "expected": "<|IGNORE|>"},
    {"input": "HDFC Bank Alert: Rs. 450.00 spent on Zomato.", "expected": '{"amount": 450.0, "merchant": "Zomato"}'},
    {"input": "Whatsapp from Sis: Did you get the Netflix subscription?", "expected": "<|IGNORE|>"},
    {"input": "Your payment of $12.00 to AMZN Digital was processed successfully.", "expected": '{"amount": 12.0, "merchant": "AMAZON"}'},
    {"input": "CitiBank Card: Transaction of USD 85.50 at Uber Trips.", "expected": '{"amount": 85.5, "merchant": "Uber"}'},
    {"input": "Calendar Reminder: Dinner with Dave at 8 PM.", "expected": "<|IGNORE|>"},
    {"input": "Venmo: You paid $25.00 to Starbucks for coffee.", "expected": '{"amount": 25.0, "merchant": "Starbucks"}'},
    {"input": "Telegram - Mom: Don't forget to buy 2 gallons of milk.", "expected": "<|IGNORE|>"},
    {"input": "Chase Alert: $120.45 debited at Target Store 2431.", "expected": '{"amount": 120.45, "merchant": "Target"}'},
    {"input": "Swiggy Promo: Get 50% off up to Rs. 100 on your next order!", "expected": "<|IGNORE|>"}
]

class SLMEvaluator:
    """Evaluates fine-tuned model performance on extraction edge-cases."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self, limit: int = 20) -> list:
        """Parses local verification datasets with a clean procedural pipeline."""
        if not os.path.exists(Settings.DATA_PATH):
            print("Dataset target path missing. Loading fallback fixtures.")
            return MOCK_TEST_CASES

        cases = []
        with open(Settings.DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if len(cases) >= limit:
                    break
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    
                    user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
                    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
                    
                    if user_msg and assistant_msg:
                        cases.append({"input": user_msg, "expected": assistant_msg})
                except json.JSONDecodeError:
                    continue
                    
        return cases if cases else MOCK_TEST_CASES

    def _init_inference_engine(self):
        """Dispatches model loading context dynamically based on environment features."""
        print(f"Initializing inference backend on target hardware: {self.device}")
        
        if self.device == "cuda":
            try:
                from unsloth import FastLanguageModel
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=Settings.OUTPUT_DIR,
                    max_seq_length=Settings.MAX_SEQ_LENGTH,
                    load_in_4bit=True
                )
                FastLanguageModel.for_inference(self.model)
                return
            except (ImportError, NotImplementedError):
                print("Unsloth acceleration unavailable. Reverting to base transformers...")

        self.tokenizer = AutoTokenizer.from_pretrained(Settings.OUTPUT_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            Settings.OUTPUT_DIR,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

    def run_suite(self):
        """Executes verification sequences and profiles extraction metrics."""
        self._init_inference_engine()

        y_true, y_pred = [], []
        exact_match_hits = 0

        for case in self.test_cases:
            messages = [
                {"role": "system", "content": "You are a financial assistant. Extract merchant and amount or return <|IGNORE|>."},
                {"role": "user", "content": case["input"]}
            ]
            
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=64, 
                    temperature=0.0, 
                    do_sample=False
                )
            
            raw_response = self.tokenizer.batch_decode(outputs)[0]
            

            response = raw_response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            
            if response == case["expected"]:
                exact_match_hits += 1

            y_true.append(0 if case["expected"] == "<|IGNORE|>" else 1)
            y_pred.append(0 if response == "<|IGNORE|>" else 1)

            print(f"\nInput:  {case['input']}\nGot:    {response}\nMatch:  {response == case['expected']}")

        self._print_metrics_report(exact_match_hits, y_true, y_pred)

    def _print_metrics_report(self, exact_matches: int, y_true: list, y_pred: list):
        """Computes statistical metrics across transaction classifications."""
        total = len(self.test_cases)
        accuracy = (exact_matches / total) * 100

  
        tp = sum(1 for gt, pr in zip(y_true, y_pred) if gt == 1 and pr == 1)
        fp = sum(1 for gt, pr in zip(y_true, y_pred) if gt == 0 and pr == 1)
        fn = sum(1 for gt, pr in zip(y_true, y_pred) if gt == 1 and pr == 0)

        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"\n" + "="*40)
        print(f"Evaluation Complete | Total Records: {total}")
        print(f"Exact Match Accuracy: {accuracy:.2f}%")
        print(f"-"*40)
        print(f"Transaction Detection Profiling:")
        print(f"  Precision : {precision:.2f}%")
        print(f"  Recall    : {recall:.2f}%")
        print(f"  F1-Score  : {f1_score:.2f}%")
        print(f"="*40)

if __name__ == "__main__":
    SLMEvaluator().run_suite()
