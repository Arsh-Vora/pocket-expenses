import json
import os
import sys
import time
from google import genai
from google.genai import types
from config.settings import Settings

class DatasetGenerator:
    """Generates balanced synthetic financial data."""
    
    def __init__(self):
        self.client = genai.Client(api_key=Settings.GEMINI_API_KEY)
        self.target_count = 1000
        self.output_path = Settings.DATA_PATH

    def _build_prompt(self, batch_size: int) -> str:
        return f"""
        Generate a JSON array containing exactly {batch_size} realistic smartphone notifications or email snippets.
        Every entry must strictly follow this ChatML structure:
        {{
            "messages": [
                {{"role": "system", "content": "You are a financial assistant. Extract merchant and amount or return <|IGNORE|>."}},
                {{"role": "user", "content": "RAW_TEXT_HERE"}},
                {{"role": "assistant", "content": "JSON_STRING_OR_IGNORE_HERE"}}
            ]
        }}

        Data Distribution Rules:
        1. 50% must be IGNORE cases. Include app names like Whatsapp, Instagram, Telegram, Netflix, Prime, Linkedin.
           Example raw text: "[Whatsapp] Mom: Did you pay the milk bill?" -> Assistant output: "<|IGNORE|>"
           Example raw text: "Netflix: Your subscription auto-renewed" -> Assistant output: "<|IGNORE|>"
        2. 50% must be VALID transactions. Use trigger words like sent, paid, debited, deducted, payment, loaded.
           Vary the sources (Bank SMS, Gmail notifications, Stripe, Google Pay).
           Example raw text: "Alert: USD 15.50 debited at Starbucks." -> Assistant output: "{{\\"amount\\": 15.50, \\"merchant\\": \\"Starbucks\\"}}"

        Return ONLY raw valid JSON array. No markdown formatting.
        """

    def generate_all(self):
        # Executes generation loop until threshold met.
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        current_count = 0
        if os.path.exists(self.output_path):
            with open(self.output_path, "r", encoding="utf-8") as f:
                current_count = sum(1 for _ in f)

        batch_size = 50

        print(f"Starting dataset generation. Current: {current_count}, Target: {self.target_count}...")

        with open(self.output_path, "a", encoding="utf-8") as f:
            while current_count < self.target_count:
                try:
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=self._build_prompt(batch_size),
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                    batch_data = json.loads(response.text)
                    for item in batch_data:
                        if current_count < self.target_count:
                            f.write(json.dumps(item) + "\n")
                            current_count += 1
                    f.flush()
                    print(f"Progress: {current_count}/{self.target_count} records collected.")
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        print("Rate limit hit (429). Waiting for 35 seconds before retrying...")
                        time.sleep(35)
                        continue
                    else:
                        print(f"Batch generation failed: {e}. Stopping to preserve collected data.")
                        sys.exit(1)

        print("Dataset generation process finished.")

if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate_all()