"""
Gemini批量生成训练数据 - 高效版本
目标: 快速生成10,000高质量样本
"""
import json
import time
from typing import List
from collections import deque, defaultdict
from tqdm import tqdm
from google import genai
from google.genai import types


def load_api_keys(key_file: str) -> List[str]:
    """加载API keys"""
    with open(key_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]


class GeminiDataGenerator:
    """Gemini批量数据生成器"""

    def __init__(self, api_keys: List[str], output_file: str, target_count: int = 10000):
        self.api_keys = api_keys
        self.output_file = output_file
        self.target_count = target_count
        self.generated = []

        # 加载已有进度
        self._load_progress()

    def _load_progress(self):
        """加载已生成的样本"""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                self.generated = [json.loads(line) for line in f if line.strip()]
            print(f"📂 加载已有样本: {len(self.generated)}")
        except:
            self.generated = []

    def _save_samples(self, samples: List[dict]):
        """保存样本"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

    def generate(self, request_delay: int = 5):
        """生成数据"""

        if len(self.generated) >= self.target_count:
            print(f"✅ 已完成! 共{len(self.generated)}样本")
            return

        needed = self.target_count - len(self.generated)
        print(f"\n🚀 开始生成 {needed} 个样本...")
        print(f"   API Keys: {len(self.api_keys)}")
        print(f"   预计耗时: {needed * request_delay / 3600 / len(self.api_keys):.1f} 小时\n")

        # 多样化的prompt模板
        prompts = self._create_diverse_prompts()

        available_keys = deque(range(len(self.api_keys)))
        pbar = tqdm(total=needed, desc="生成中")

        consecutive_failures = 0
        attempts = 0
        max_attempts = needed * 3

        while len(self.generated) < self.target_count and attempts < max_attempts:
            attempts += 1

            if consecutive_failures >= 50:
                print("\n⚠️  连续失败过多,暂停30秒...")
                time.sleep(30)
                consecutive_failures = 0

            if not available_keys:
                time.sleep(10)
                available_keys = deque(range(len(self.api_keys)))
                continue

            key_index = available_keys.popleft()
            api_key = self.api_keys[key_index]

            # 随机选择prompt
            prompt = prompts[len(self.generated) % len(prompts)]

            try:
                time.sleep(request_delay)

                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(
                    temperature=1.0,
                    top_p=0.95,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=config
                )

                if not response or not response.text:
                    raise ValueError("Empty response")

                text = response.text.strip()

                # 清理JSON
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.replace('```', '').strip()

                data = json.loads(text)

                # 验证字段
                required = ['anchor_text', 'text_a', 'text_b', 'text_a_is_closer']
                if not all(f in data for f in required):
                    raise ValueError("Missing fields")

                # 验证长度
                for field in ['anchor_text', 'text_a', 'text_b']:
                    if len(data[field].split()) < 15:
                        raise ValueError(f"{field} too short")

                # 成功
                self.generated.append(data)
                consecutive_failures = 0
                pbar.update(1)

                available_keys.append(key_index)

                # 定期保存
                if len(self.generated) % 50 == 0:
                    self._save_samples(self.generated)

            except Exception as e:
                consecutive_failures += 1
                available_keys.append(key_index)

        pbar.close()
        self._save_samples(self.generated)

        print(f"\n✅ 生成完成!")
        print(f"   总样本: {len(self.generated)}")
        print(f"   成功率: {len(self.generated) / attempts * 100:.1f}%")

    def _create_diverse_prompts(self) -> List[str]:
        """创建多样化的prompt"""

        # 基础模板
        base_template = """Generate a narrative similarity training example.

Create 3 Wikipedia-style story summaries where text_a is more similar to anchor than text_b.

GENRE: {genre}
FOCUS: {focus}

REQUIREMENTS:
1. anchor_text: Complete story (30-60 words)
2. text_a: More similar story (shares {similarity_aspect})
3. text_b: Less similar story

Output valid JSON only:
{{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "...",
  "text_a_is_closer": true
}}"""

        # 多样化配置
        configs = [
            # 主题相似
            {"genre": "Science Fiction", "focus": "Space exploration", "similarity_aspect": "themes"},
            {"genre": "Mystery", "focus": "Detective solving crimes", "similarity_aspect": "themes"},
            {"genre": "Romance", "focus": "Love and relationships", "similarity_aspect": "themes"},
            {"genre": "Historical", "focus": "War and conflict", "similarity_aspect": "themes"},
            {"genre": "Fantasy", "focus": "Magic and quests", "similarity_aspect": "themes"},

            # 情节相似
            {"genre": "Adventure", "focus": "Journey and discovery", "similarity_aspect": "course of action"},
            {"genre": "Thriller", "focus": "Chase and escape", "similarity_aspect": "course of action"},
            {"genre": "Drama", "focus": "Character development", "similarity_aspect": "course of action"},

            # 结局相似
            {"genre": "Comedy", "focus": "Happy endings", "similarity_aspect": "outcomes"},
            {"genre": "Tragedy", "focus": "Sad endings", "similarity_aspect": "outcomes"},

            # 混合
            {"genre": "Biography", "focus": "Real people's lives", "similarity_aspect": "multiple aspects"},
            {"genre": "Horror", "focus": "Fear and survival", "similarity_aspect": "multiple aspects"},
        ]

        prompts = [base_template.format(**cfg) for cfg in configs]

        # 添加困难样本prompt
        hard_prompt = """Generate a CHALLENGING narrative similarity example.

Make text_a and text_b BOTH similar to anchor, but text_a slightly more so.

Requirements:
- All three stories should share some common elements
- The difference should be subtle
- Focus on nuanced narrative similarities

Output valid JSON only:
{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "...",
  "text_a_is_closer": true
}"""

        prompts.append(hard_prompt)

        return prompts


def main():
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    KEY_FILE = f"{PROJECT_ROOT}/config/gemini_api_keys.txt"
    OUTPUT_FILE = f"{PROJECT_ROOT}/GeminiData/gemini_generated_10k.jsonl"

    API_KEYS = load_api_keys(KEY_FILE)
    print(f"✅ 加载 {len(API_KEYS)} 个API keys")

    generator = GeminiDataGenerator(
        api_keys=API_KEYS,
        output_file=OUTPUT_FILE,
        target_count=10000
    )

    generator.generate(request_delay=5)


if __name__ == "__main__":
    main()