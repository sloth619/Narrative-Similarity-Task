"""
Gemini 2.5 Pro API - Track A
"""
import json
import time
import os
from typing import List, Dict, Optional
from collections import deque
from datetime import datetime, timedelta
from tqdm import tqdm
from google import genai
from google.genai import types
from datasets import load_dataset
import threading


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
        """获取一个可用的key"""
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

    def wait_for_available_key(self, timeout: int = 600) -> Optional[tuple]:
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            key_info = self.get_available_key()
            if key_info:
                return key_info
            with self.lock:
                if self.cooling:
                    wait_seconds = max(1, (min(self.cooling.values()) - datetime.now()).total_seconds())
                    time.sleep(min(wait_seconds, 10))
                else:
                    time.sleep(1)
        return None

    def get_stats(self) -> Dict:
        with self.lock:
            return {
                'available': len(self.available),
                'cooling': len(self.cooling),
                'details': self.stats
            }


class GeminiProSubmissionGenerator:
    """使用Gemini 2.5 Pro生成提交文件"""

    def __init__(self, api_keys: List[str], request_delay: int = 30, key_cooldown: int = 120):
        self.key_pool = APIKeyPool(api_keys, cooldown_seconds=key_cooldown)
        self.model = "gemini-2.5-pro"
        self.request_delay = request_delay

    def _create_track_a_prompt(self, anchor: str, text_a: str, text_b: str) -> str:
        prompt = f"""You are an expert in narrative analysis and story comparison. Analyze three story summaries to determine narrative similarity.

NARRATIVE SIMILARITY FRAMEWORK:
Narrative similarity is evaluated across three independent dimensions:
1. **Abstract Theme** (30% weight): Core philosophical ideas, central conflicts, motifs.
2. **Course of Action** (40% weight): Sequence of events, plot structure, turning points.
3. **Outcomes** (30% weight): Final resolution, character fates, moral implications.

TASK:
Compare the following three stories and determine which candidate (A or B) is MORE narratively similar to the Anchor story.

═══════════════════════════════════════════════════════════
ANCHOR STORY:
{anchor}

CANDIDATE A:
{text_a}

CANDIDATE B:
{text_b}
═══════════════════════════════════════════════════════════

ANALYSIS REQUIREMENTS:
1. Evaluate each dimension independently.
2. Ignore surface features (names, locations).
3. Focus on structural and thematic parallels.

Respond with ONLY the letter "A" or "B".

Your response: """
        return prompt

    def predict_track_a_one(self, anchor: str, text_a: str, text_b: str, max_retries: int = 3) -> Optional[bool]:
        """预测单个样本"""
        prompt = self._create_track_a_prompt(anchor, text_a, text_b)

        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
        ]

        for retry in range(max_retries):
            key_info = self.key_pool.wait_for_available_key(timeout=600)
            if not key_info:
                return None
            key_index, api_key = key_info

            try:
                time.sleep(2)
                client = genai.Client(api_key=api_key)

                config = types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                    top_k=40,
                    top_p=0.95,
                    safety_settings=safety_settings,
                    thinking_config=types.ThinkingConfig(thinking_budget=1024)
                )

                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )

                final_text = None
                if response.text:
                    final_text = response.text
                elif response.candidates and response.candidates[0].content.parts:
                    final_text = " ".join([p.text for p in response.candidates[0].content.parts if p.text])

                if not final_text:
                    if response.candidates:
                        reason = response.candidates[0].finish_reason
                        raise ValueError(f"Empty response. Finish reason: {reason}")
                    raise ValueError("Response content is completely empty")

                answer = final_text.strip().upper()

                if 'A' in answer and 'B' not in answer:
                    result = True
                elif 'B' in answer and 'A' not in answer:
                    result = False
                elif answer == 'A':
                    result = True
                elif answer == 'B':
                    result = False
                else:
                    if "OUTPUT: A" in answer or answer.endswith("A") or "ANSWER: A" in answer:
                        result = True
                    elif "OUTPUT: B" in answer or answer.endswith("B") or "ANSWER: B" in answer:
                        result = False
                    else:
                        lines = [l.strip() for l in answer.split('\n') if l.strip()]
                        if lines and lines[-1] == 'A':
                            result = True
                        elif lines and lines[-1] == 'B':
                            result = False
                        else:
                            raise ValueError(f"Ambiguous answer")

                self.key_pool.mark_key_used(key_index, success=True)
                self.key_pool.return_key(key_index, need_cooldown=False)
                return result

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    self.key_pool.return_key(key_index, need_cooldown=True)
                else:
                    self.key_pool.return_key(key_index, need_cooldown=False)
                    time.sleep(2)

        return None

    def generate_track_a_submission(self, test_file: str, output_file: str = 'track_a.jsonl', save_interval: int = 5):
        """生成Track A提交文件"""
        dataset = load_dataset('json', data_files=test_file, split='train')

        predictions = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    predictions = [json.loads(line) for line in f if line.strip()]
            except:
                predictions = []

        start_idx = len(predictions)
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="Predicting", ncols=80)
        consecutive_failures = 0

        for idx in range(start_idx, len(dataset)):
            item = dataset[idx]
            anchor = item.get('anchor_text') or item.get('anchor_story')
            text_a = item.get('text_a') or item.get('similar_story')
            text_b = item.get('text_b') or item.get('dissimilar_story')

            if not all([anchor, text_a, text_b]):
                predictions.append({'text_a_is_closer': True})
                pbar.update(1)
                continue

            if consecutive_failures >= 10:
                break

            result = self.predict_track_a_one(anchor, text_a, text_b)

            if result is None:
                result = True
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            predictions.append({'text_a_is_closer': result})

            if (idx + 1) % save_interval == 0:
                self._save_jsonl(predictions, output_file)

            # 简洁的进度显示
            stats = self.key_pool.get_stats()
            pbar.set_postfix({
                'avail': stats['available'],
                'cool': stats['cooling']
            })
            pbar.update(1)

        pbar.close()
        self._save_jsonl(predictions, output_file)

        # 简洁统计
        a_count = sum(1 for p in predictions if p['text_a_is_closer'])
        print(f"\n✅ 完成: {len(predictions)} 样本")
        print(f"   A={a_count}, B={len(predictions)-a_count}")
        print(f"   文件: {output_file}")

    def _save_jsonl(self, data: List[Dict], filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_api_keys(key_file: str) -> List[str]:
    """从文件加载API keys"""
    with open(key_file, 'r', encoding='utf-8') as f:
        keys = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return keys


def main():
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    # 从外部文件加载keys
    KEY_FILE = f"{PROJECT_ROOT}/config/gemini_api_keys.txt"
    API_KEYS = load_api_keys(KEY_FILE)

    print(f"✅ 加载 {len(API_KEYS)} 个API keys")

    # 测试集路径
    TRACK_A_TEST = f"{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl"
    # TRACK_A_TEST = f"{PROJECT_ROOT}/test/track_a.jsonl"

    OUTPUT_DIR = f"{PROJECT_ROOT}/submissions/gemini_pro_multikey"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generator = GeminiProSubmissionGenerator(
        api_keys=API_KEYS,
        request_delay=30,
        key_cooldown=120
    )

    generator.generate_track_a_submission(
        test_file=TRACK_A_TEST,
        output_file=f"{OUTPUT_DIR}/track_a.jsonl",
        save_interval=5
    )


if __name__ == "__main__":
    main()