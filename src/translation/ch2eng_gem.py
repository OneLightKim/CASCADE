import json
import time
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)

class TranslationConfig:
    def __init__(self, 
                 api_key: str,
                 model_name: str = 'gemini-1.5-flash',
                 save_interval: int = 5,
                 temperature: float = 0.3,
                 input_file: str = "data/main/PsyQA_full.json",
                 output_file: str = "data/main/psyqa_translated_gemini.json",
                 backup_file: str = "data/main/psyqa_translated_gemini_backup.json"):
        self.api_key = api_key
        self.model_name = model_name
        self.save_interval = save_interval
        self.temperature = temperature
        self.input_file = input_file
        self.output_file = output_file
        self.backup_file = backup_file

class TranslationManager:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.setup_genai()
        
    def setup_genai(self):
        """Initialize Google AI client"""
        genai.configure(api_key=self.config.api_key)
    
    def translate_text(self, text: str) -> str:
        """Translate single text with retry mechanism"""
        if not text:
            return ""
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                model = genai.GenerativeModel(self.config.model_name)
                prompt = "You are a professional translator. Translate the following Chinese text to English. Maintain the original meaning and tone while ensuring natural English expression.\n\nText to translate: " + text
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                retry_count += 1
                logging.error(f"Translation error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(20)  # Wait before retry
                else:
                    logging.error(f"Failed to translate after {max_retries} attempts")
                    return f"[TRANSLATION ERROR] {text}"
    
    def translate_keywords(self, keywords: List[str]) -> List[str]:
        """Translate list of keywords"""
        if not keywords:
            return []
        
        keywords_str = ", ".join(keywords)
        translated = self.translate_text(keywords_str)
        return [k.strip() for k in translated.split(",")]

    def load_progress(self) -> Tuple[List[Dict], int]:
        """Load previous translation progress"""
        if os.path.exists(self.config.output_file):
            try:
                with open(self.config.output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data, len(data)
            except json.JSONDecodeError:
                logging.warning(f"Error reading {self.config.output_file}, trying backup file")
                return self._try_load_backup()
        return [], 0
    
    def _try_load_backup(self) -> Tuple[List[Dict], int]:
        """Try to load from backup file if main file is corrupted"""
        if os.path.exists(self.config.backup_file):
            try:
                with open(self.config.backup_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data, len(data)
            except json.JSONDecodeError:
                logging.error("Both main and backup files are corrupted")
        return [], 0

    def save_progress(self, translated_data: List[Dict], is_backup: bool = False):
        """Save translation progress"""
        target_file = self.config.backup_file if is_backup else self.config.output_file
        try:
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving to {target_file}: {e}")

    def translate_entry(self, entry: Dict) -> Dict:
        """Translate a single PsyQA entry"""
        translated_entry = {
            "id": entry.get("id", ""),
            "question": self.translate_text(entry.get("question", "")),
            "description": self.translate_text(entry.get("description", "")),
            "keywords": self.translate_keywords(entry.get("keywords", [])),
            "answers": []
        }
        
        for answer in entry.get("answers", []):
            translated_answer = {
                "answer_id": answer.get("answer_id", ""),
                "answer_text": self.translate_text(answer.get("answer_text", "")),
                "upvotes": answer.get("upvotes", 0),
                "is_accepted": answer.get("is_accepted", False)
            }
            translated_entry["answers"].append(translated_answer)
            
        return translated_entry

    def run_translation(self):
        """Main translation process"""
        logging.info("Loading PsyQA dataset...")
        with open(self.config.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        translated_data, start_idx = self.load_progress()
        logging.info(f"Resuming from entry {start_idx}/{len(data)}")
        
        for idx, entry in enumerate(tqdm(data[start_idx:], 
                                       desc="Translating entries", 
                                       initial=start_idx, 
                                       total=len(data))):
            translated_entry = self.translate_entry(entry)
            translated_data.append(translated_entry)
            
            # Save progress periodically
            if (len(translated_data) % self.config.save_interval) == 0:
                self.save_progress(translated_data)
                self.save_progress(translated_data, is_backup=True)
        
        # Save final results
        self.save_progress(translated_data)
        logging.info(f"Translation completed! Results saved to {self.config.output_file}")

def main():
    # Configuration
    config = TranslationConfig(
        api_key="",  # Replace with your API key
    )
    
    # Initialize and run translation
    translator = TranslationManager(config)
    translator.run_translation()

if __name__ == "__main__":
    main() 