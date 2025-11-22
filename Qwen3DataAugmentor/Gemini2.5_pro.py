"""
Gemini 2.5 Pro 数据生成器
使用Pro模型生成高质量训练数据
"""
import json
import time
from typing import List, Dict, Optional
from tqdm import tqdm
from google import genai
from google.genai import types
from datetime import datetime, timedelta
from collections import deque
import threading


def load_api_keys(key_file: str) -> List[str]:
    """从文件加载API keys"""
    with open(key_file, 'r', encoding='utf-8') as f:
        keys = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return keys


class APIKeyPool:
    """API Key池管理器"""

    def __init__(self, api_keys: List[str], cooldown_seconds: int = 120):
        self.keys = api_keys
        self.cooldown = cooldown_seconds
        self.available = deque(range(len(api_keys)))
        self.cooling = {}
        self.stats = {
            i: {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'last_used': None,
                'daily_count': 0,
                'last_reset_date': datetime.now().date()
            }
            for i in range(len(api_keys))
        }
        self.lock = threading.Lock()

    def get_available_key(self) -> Optional[tuple]:
        with self.lock:
            current_time = datetime.now()
            recovered = []

            for key_index, recover_time in list(self.cooling.items()):
                if current_time >= recover_time:
                    self.available.append(key_index)
                    recovered.append(key_index)

            for key_index in recovered:
                del self.cooling[key_index]

            if self.available:
                key_index = self.available.popleft()
                stat = self.stats[key_index]
                today = datetime.now().date()

                if stat['last_reset_date'] != today:
                    stat['daily_count'] = 0
                    stat['last_reset_date'] = today

                if stat['daily_count'] >= 45:
                    return self.get_available_key()

                return key_index, self.keys[key_index]

            return None

    def mark_key_used(self, key_index: int, success: bool = True):
        with self.lock:
            stat = self.stats[key_index]
            stat['total_requests'] += 1
            stat['last_used'] = datetime.now()

            if success:
                stat['successful_requests'] += 1
                stat['daily_count'] += 1
            else:
                stat['failed_requests'] += 1

    def return_key(self, key_index: int, need_cooldown: bool = False):
        with self.lock:
            if need_cooldown:
                recover_time = datetime.now() + timedelta(seconds=self.cooldown)
                self.cooling[key_index] = recover_time
            else:
                self.available.append(key_index)

    def get_stats(self) -> Dict:
        with self.lock:
            return {
                'available': len(self.available),
                'cooling': len(self.cooling),
                'total_keys': len(self.keys),
                'details': self.stats
            }

    def has_available_keys(self) -> bool:
        with self.lock:
            return bool(self.available or self.cooling)

    def wait_for_available_key(self, timeout: int = 600) -> Optional[tuple]:
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout:
            key_info = self.get_available_key()
            if key_info:
                return key_info

            with self.lock:
                if self.cooling:
                    min_recover_time = min(self.cooling.values())
                    wait_seconds = max(1, (min_recover_time - datetime.now()).total_seconds())
                    wait_seconds = min(wait_seconds, 10)
                    time.sleep(wait_seconds)
                else:
                    return None

        return None


class GeminiProGenerator:
    """Gemini 2.5 Pro 数据生成器"""

    def __init__(self, api_keys: List[str], key_cooldown: int = 120, request_delay: int = 30):
        self.model_name = "gemini-2.5-pro"
        self.request_delay = request_delay
        self.key_pool = APIKeyPool(api_keys, cooldown_seconds=key_cooldown)

    def _create_prompt(self) -> str:
        prompt = """Generate a high-quality narrative similarity example for SemEval-2026 Task 4.

TASK: Create 3 Wikipedia-style story summaries

NARRATIVE SIMILARITY DIMENSIONS:
1. Abstract Theme: Core problems, central ideas, motifs
2. Course of Action: Event sequences, turning points, ORDER
3. Outcomes: Final results, character fates, moral lessons

REQUIREMENTS:
- anchor_text: Story with specific names and details (2-3 sentences, 25-60 words)
- text_a: Similar story matching anchor in 2-3 dimensions (2-3 sentences)
- text_b: Different story differing in 2+ dimensions (2-3 sentences)

GUIDELINES:
- Use diverse genres
- Make similarities meaningful
- Each story 25-60 words
- Focus on narrative elements

Output valid JSON only:
{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "...",
  "text_a_is_closer": true
}"""
        return prompt

    def generate_one_sample(self, max_retries: int = 3) -> Optional[Dict]:
        for retry in range(max_retries):
            key_info = self.key_pool.wait_for_available_key(timeout=600)

            if not key_info:
                return None

            key_index, api_key = key_info

            try:
                time.sleep(self.request_delay)

                client = genai.Client(api_key=api_key)

                config = types.GenerateContentConfig(
                    temperature=1.0,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=2048,
                    thinking_config=types.ThinkingConfig(thinking_budget=1024)
                )

                prompt = self._create_prompt()

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )

                if not response or not hasattr(response, 'text') or not response.text:
                    raise ValueError("Empty response")

                text = response.text.strip()

                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.replace('```', '').strip()

                try:
                    data = json.loads(text)
                except json.JSONDecodeError as je:
                    raise ValueError(f"Invalid JSON")

                required = ['anchor_text', 'text_a', 'text_b', 'text_a_is_closer']
                missing = [f for f in required if f not in data]
                if missing:
                    raise ValueError(f"Missing fields")

                for field in ['anchor_text', 'text_a', 'text_b']:
                    words = len(data[field].split())
                    if words < 15:
                        raise ValueError(f"{field} too short")

                self.key_pool.mark_key_used(key_index, success=True)
                self.key_pool.return_key(key_index, need_cooldown=False)

                return data

            except Exception as e:
                error_msg = str(e).lower()
                self.key_pool.mark_key_used(key_index, success=False)

                if any(kw in error_msg for kw in ['rate', '429', 'quota', 'resource', 'limit']):
                    self.key_pool.return_key(key_index, need_cooldown=True)
                    continue
                elif 'thinking' in error_msg or 'budget' in error_msg:
                    self.key_pool.return_key(key_index, need_cooldown=False)
                    if retry < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        return None
                else:
                    self.key_pool.return_key(key_index, need_cooldown=False)
                    if retry < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        return None

        return None

    def generate_dataset(self, num_samples: int, output_file: str, save_interval: int = 5):
        """生成数据集"""

        generated_samples = []
        successful = 0
        failed = 0

        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                generated_samples = [json.loads(line) for line in f]
                successful = len(generated_samples)
        except:
            pass

        pbar = tqdm(total=num_samples, initial=successful, desc="Generating", ncols=80)

        consecutive_failures = 0
        start_time = time.time()

        while successful < num_samples:
            if not self.key_pool.has_available_keys():
                break

            if consecutive_failures >= 5:
                break

            sample = self.generate_one_sample()

            if sample:
                generated_samples.append(sample)
                successful += 1
                consecutive_failures = 0
                pbar.update(1)

                stats = self.key_pool.get_stats()
                elapsed = time.time() - start_time
                rate = successful / elapsed * 60 if elapsed > 0 else 0

                pbar.set_postfix({
                    'rate': f'{rate:.1f}/min',
                    'avail': stats['available'],
                    'cool': stats['cooling']
                })

                if successful % save_interval == 0:
                    self._save_samples(generated_samples, output_file)
            else:
                failed += 1
                consecutive_failures += 1

        pbar.close()
        self._save_samples(generated_samples, output_file)

        elapsed_total = time.time() - start_time
        print(f"\n✅ 完成: {successful}/{num_samples} 样本")
        print(f"   耗时: {elapsed_total / 60:.1f} 分钟")
        print(f"   速率: {successful / elapsed_total * 60:.1f} 样本/分钟")
        print(f"   文件: {output_file}")

    def _save_samples(self, samples: List[Dict], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')


def main():
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    KEY_FILE = f"{PROJECT_ROOT}/config/gemini_api_keys.txt"
    OUTPUT_FILE = f"{PROJECT_ROOT}/GeminiData/gemini_pro_data.jsonl"

    API_KEYS = load_api_keys(KEY_FILE)
    print(f"✅ 加载 {len(API_KEYS)} 个API keys")

    generator = GeminiProGenerator(
        api_keys=API_KEYS,
        key_cooldown=120,
        request_delay=30
    )

    generator.generate_dataset(
        num_samples=500,
        output_file=OUTPUT_FILE,
        save_interval=5
    )


if __name__ == "__main__":
    main()