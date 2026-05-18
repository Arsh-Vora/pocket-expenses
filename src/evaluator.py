import torch
import json
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except (ImportError, NotImplementedError):
    HAS_UNSLOTH = False

from transformers import AutoTokenizer
from config.settings import Settings

class SLMEvaluator:
    # Evaluates fine-tuned model edge-case performance.
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self, limit=20):
        # Dynamically load test cases from the dataset
        cases = []
        try:
            with open(Settings.DATA_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        messages = data.get("messages", [])
                        user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
                        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
                        if user_msg and assistant_msg:
                            cases.append({"input": user_msg, "expected": assistant_msg})
                        if len(cases) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
            
        if not cases:
            print("Dataset not available or empty. Using fallback test cases.")
            cases = [
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
        return cases

    def run_suite(self):
        # Loads local model and evaluates validation records.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading local weights for evaluation on {device}...")
        
        if HAS_UNSLOTH and device == "cuda":
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=Settings.OUTPUT_DIR,
                max_seq_length=Settings.MAX_SEQ_LENGTH,
                load_in_4bit=True
            )
            FastLanguageModel.for_inference(self.model)
        else:
            print("Unsloth/CUDA not detected. Falling back to transformers on CPU...")
            from transformers import AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(Settings.OUTPUT_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                Settings.OUTPUT_DIR,
                device_map=device
            )

        correct = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for case in self.test_cases:
            messages = [
                {"role": "system", "content": "You are a financial assistant. Extract merchant and amount or return <|IGNORE|>."},
                {"role": "user", "content": case["input"]}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([prompt], return_tensors="pt").to(device)

            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=64, 
                temperature=0.0, # Greedy inference tracking
                do_sample=False
            )
            
            # Extract assistant response only
            raw_response = self.tokenizer.batch_decode(outputs)[0]
            response = raw_response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            
            is_correct = (response == case["expected"])
            if is_correct:
                correct += 1
                
            expected_ignore = (case["expected"] == "<|IGNORE|>")
            got_ignore = (response == "<|IGNORE|>")
            
            if expected_ignore and got_ignore:
                tn += 1
            elif expected_ignore and not got_ignore:
                fp += 1
            elif not expected_ignore and got_ignore:
                fn += 1
            elif not expected_ignore and not got_ignore:
                tp += 1
            
            print(f"\nInput: {case['input']}\nExpected: {case['expected']}\nGot: {response}\nPass: {is_correct}")

        accuracy = (correct / len(self.test_cases)) * 100
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"\n======================")
        print(f"Final Exact Match Accuracy: {accuracy:.2f}%")
        print(f"Transaction Detection Metrics:")
        print(f"  - Precision: {precision:.2f}%")
        print(f"  - Recall:    {recall:.2f}%")
        print(f"  - F1-Score:  {f1_score:.2f}%")
        print(f"======================")

if __name__ == "__main__":
    evaluator = SLMEvaluator()
    evaluator.run_suite()