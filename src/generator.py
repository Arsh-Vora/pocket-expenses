import json
import os
import sys
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError  
from config.settings import Settings

class DatasetGenerator:
    
    
    def __init__(self):
        self.client = genai.Client(api_key=Settings.GEMINI_API_KEY)
        self.target_count = 1000
        self.output_path = Settings.DATA_PATH
        self.batch_size = 50

    def _get_current_count(self) -> int:
        """Efficient byte-chunk line counting for resume states."""
        if not os.path.exists(self.output_path):
            return 0
        with open(self.output_path, "rb") as f:
            return sum(bl.count(b"\n") for bl in iter(lambda: f.read(8192), b""))

    def generate_all(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        current_count = self._get_current_count()

        print(f"Starting dataset generation. Current: {current_count}/{self.target_count}")

        
        prompt_template = (
            f"Generate exactly {self.batch_size} ChatML format objects for financial data fine-tuning.\n"
            "Include an even 50/50 mix of valid transactions (extract amount/merchant) and noise "
            "(app notifications from social media/streaming to trigger <|IGNORE|>).\n"
            "Return raw JSON array only."
        )

        with open(self.output_path, "a", encoding="utf-8") as f:
            while current_count < self.target_count:
                try:
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_template,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                    
                    batch_data = json.loads(response.text)
                    for item in batch_data:
                        if current_count >= self.target_count:
                            break
                        f.write(json.dumps(item) + "\n")
                        current_count += 1
                        
                    f.flush()
                    print(f"Progress: {current_count}/{self.target_count}")

                except APIError as e:
                    
                    if e.code == 429:
                        print("Rate limited. Backing off for 45s...")
                        time.sleep(45)
                        continue
                    raise e
                except Exception as e:
                    print(f"Fatal processing error: {e}")
                    sys.exit(1)

if __name__ == "__main__":
    DatasetGenerator().generate_all()
