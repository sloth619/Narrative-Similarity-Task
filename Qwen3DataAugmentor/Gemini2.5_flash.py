"""
Safe Hard Negative Mining Pipeline
支持全程断点续传,外部API Key配置
"""
import json
import random
import time
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from google import genai
from google.genai import types
import os
import pickle


def load_api_keys(key_file: str) -> List[str]:
    """从文件加载API keys"""
    with open(key_file, 'r', encoding='utf-8') as f:
        keys = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return keys


class SafeHardNegativePipeline:
    """安全的Hard Negative挖掘Pipeline"""

    def __init__(self, model_path: str, train_only: bool = True, cache_dir: str = '../cache'):
        self.model = SentenceTransformer(model_path)
        self.train_only = train_only
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, step: str, ext: str = 'pkl') -> str:
        return os.path.join(self.cache_dir, f'{step}.{ext}')

    def _load_cache(self, step: str):
        cache_path = self._get_cache_path(step)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def _save_cache(self, step: str, data):
        cache_path = self._get_cache_path(step)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass

    def step1_analyze_train_patterns(self, train_path: str, output_file: str, resume: bool = True) -> Dict:
        """步骤1: 分析训练集的困难模式"""

        if resume:
            cached = self._load_cache('step1_patterns')
            if cached:
                return cached

        progress_file = self._get_cache_path('step1_progress', 'json')
        processed_indices = set()
        partial_results = {'patterns': defaultdict(list), 'stats': {'similarities': []}}

        if resume and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    processed_indices = set(progress['processed_indices'])
                    partial_results = progress['partial_results']
                    partial_results['patterns'] = defaultdict(list, partial_results['patterns'])
            except:
                pass

        dataset = load_dataset('json', data_files=train_path, split='train')
        patterns = partial_results['patterns']
        stats = partial_results['stats']

        for idx, item in enumerate(tqdm(dataset, desc="Analyzing", ncols=80)):
            if idx in processed_indices:
                continue

            anchor = item.get('anchor_story') or item.get('anchor_text')
            similar = item.get('similar_story') or item.get('text_a')
            dissimilar = item.get('dissimilar_story') or item.get('text_b')

            if not all([anchor, similar, dissimilar]):
                processed_indices.add(idx)
                continue

            try:
                texts = ['passage: ' + t for t in [anchor, similar, dissimilar]]
                embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

                sim_pos = cosine_similarity([embs[0]], [embs[1]])[0][0]
                sim_neg = cosine_similarity([embs[0]], [embs[2]])[0][0]
                diff = abs(sim_pos - sim_neg)

                stats['similarities'].append({
                    'sim_pos': float(sim_pos),
                    'sim_neg': float(sim_neg),
                    'diff': float(diff)
                })

                sample_info = {
                    'anchor': anchor,
                    'similar': similar,
                    'dissimilar': dissimilar,
                    'sim_pos': float(sim_pos),
                    'sim_neg': float(sim_neg),
                    'diff': float(diff)
                }

                if diff < 0.05:
                    patterns['very_hard'].append(sample_info)
                elif diff < 0.10:
                    patterns['hard'].append(sample_info)
                elif diff < 0.20:
                    patterns['medium'].append(sample_info)

                if sim_pos > 0.7 and sim_neg > 0.6:
                    patterns['high_similarity_both'].append(sample_info)

                processed_indices.add(idx)

                if len(processed_indices) % 50 == 0:
                    progress = {
                        'processed_indices': list(processed_indices),
                        'partial_results': {'patterns': dict(patterns), 'stats': stats}
                    }
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress, f)

            except:
                processed_indices.add(idx)
                continue

        patterns = dict(patterns)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        diffs = [s['diff'] for s in stats['similarities']]

        save_data = {
            'patterns': {k: len(v) for k, v in patterns.items()},
            'stats': {
                'total': len(dataset),
                'diff_mean': float(np.mean(diffs)) if diffs else 0,
                'diff_median': float(np.median(diffs)) if diffs else 0
            },
            'samples': {k: v[:10] for k, v in patterns.items()}
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        self._save_cache('step1_patterns', patterns)

        if os.path.exists(progress_file):
            os.remove(progress_file)

        return patterns

    def step2_create_generation_prompts(self, patterns: Dict, output_file: str, resume: bool = True) -> List[str]:
        """步骤2: 创建生成prompt"""

        if resume:
            cached = self._load_cache('step2_prompts')
            if cached:
                return cached

        if resume and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                self._save_cache('step2_prompts', prompts)
                return prompts
            except:
                pass

        prompts = []
        prompt_templates = {
            'very_hard': {
                'description': 'extremely challenging cases where similarity difference is minimal (< 0.05)',
                'instructions': 'Make both candidates very similar to anchor, with only subtle differences',
                'target_count': 500
            },
            'hard': {
                'description': 'hard cases with small similarity difference (0.05-0.10)',
                'instructions': 'Both candidates should share some aspects with anchor',
                'target_count': 400
            },
            'high_similarity_both': {
                'description': 'cases where both candidates are highly similar to anchor',
                'instructions': 'Create stories where all three share strong thematic similarities',
                'target_count': 300
            }
        }

        for pattern_type, template in prompt_templates.items():
            if pattern_type not in patterns or not patterns[pattern_type]:
                continue

            samples = patterns[pattern_type]
            example_samples = random.sample(samples, min(3, len(samples)))

            examples_str = ""
            for i, ex in enumerate(example_samples, 1):
                examples_str += f"""
Example {i}:
Anchor: {ex['anchor'][:150]}...
Similar (sim={ex['sim_pos']:.3f}): {ex['similar'][:150]}...
Dissimilar (sim={ex['sim_neg']:.3f}): {ex['dissimilar'][:150]}...
"""

            prompt = f"""Generate a NEW narrative similarity example for the "{pattern_type}" difficulty level.

DIFFICULTY LEVEL: {template['description']}
INSTRUCTIONS: {template['instructions']}

REFERENCE EXAMPLES:
{examples_str}

TASK: Generate a COMPLETELY NEW story triplet with different characters and settings.

REQUIREMENTS:
1. anchor_text: A complete story (30-60 words)
2. text_a: More similar story
3. text_b: Less similar story

Output valid JSON only:
{{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "...",
  "text_a_is_closer": true,
  "pattern_type": "{pattern_type}"
}}"""

            prompts.append({
                'pattern_type': pattern_type,
                'prompt': prompt,
                'target_count': template['target_count']
            })

        diversity_prompt = """Generate a diverse narrative similarity example.

TASK: Create 3 Wikipedia-style story summaries

REQUIREMENTS:
- Random genre and diverse settings
- text_a matches anchor in 2-3 aspects
- text_b differs from anchor significantly

Output valid JSON only:
{
  "anchor_text": "Story (30-60 words)...",
  "text_a": "Similar story...",
  "text_b": "Different story...",
  "text_a_is_closer": true,
  "pattern_type": "diversity"
}"""

        prompts.append({
            'pattern_type': 'diversity',
            'prompt': diversity_prompt,
            'target_count': 300
        })

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

        self._save_cache('step2_prompts', prompts)
        return prompts

    def step3_generate_with_gemini(
        self, prompts: List[Dict], api_keys: List[str],
        output_file: str, request_delay: int = 6, resume: bool = True
    ):
        """步骤3: 使用Gemini生成数据"""

        from collections import deque

        available_keys = deque(range(len(api_keys)))
        total_target = sum(p['target_count'] for p in prompts)

        generated_samples = []
        if resume and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    generated_samples = [json.loads(line) for line in f if line.strip()]
            except:
                generated_samples = []

        existing_counts = defaultdict(int)
        for s in generated_samples:
            existing_counts[s.get('pattern_type', 'unknown')] += 1

        if len(generated_samples) >= total_target:
            return

        for prompt_info in prompts:
            pattern_type = prompt_info['pattern_type']
            prompt = prompt_info['prompt']
            target = prompt_info['target_count']
            existing = existing_counts[pattern_type]

            if existing >= target:
                continue

            needed = target - existing
            pbar = tqdm(total=needed, desc=f"{pattern_type}", ncols=80)

            attempts = 0
            max_attempts = needed * 5
            consecutive_failures = 0
            pattern_success = 0

            while pattern_success < needed and attempts < max_attempts:
                attempts += 1

                if consecutive_failures >= 20:
                    break

                if not available_keys:
                    time.sleep(30)
                    available_keys = deque(range(len(api_keys)))
                    continue

                key_index = available_keys.popleft()
                api_key = api_keys[key_index]

                try:
                    time.sleep(request_delay)

                    client = genai.Client(api_key=api_key)
                    config = types.GenerateContentConfig(
                        temperature=1.0,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=1024,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )

                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp",
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

                    data = json.loads(text)

                    required = ['anchor_text', 'text_a', 'text_b', 'text_a_is_closer']
                    if not all(f in data for f in required):
                        raise ValueError("Missing fields")

                    for field in ['anchor_text', 'text_a', 'text_b']:
                        if len(data[field].split()) < 15:
                            raise ValueError(f"{field} too short")

                    data['pattern_type'] = pattern_type
                    generated_samples.append(data)
                    existing_counts[pattern_type] += 1
                    pattern_success += 1
                    consecutive_failures = 0
                    pbar.update(1)

                    available_keys.append(key_index)

                    if len(generated_samples) % 10 == 0:
                        self._save_samples(generated_samples, output_file)

                except:
                    consecutive_failures += 1
                    available_keys.append(key_index)

            pbar.close()

        self._save_samples(generated_samples, output_file)

    def step4_validate_quality(self, input_file: str, output_file: str, resume: bool = True) -> List[Dict]:
        """步骤4: 质量验证"""

        if resume and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    valid_samples = [json.loads(line) for line in f if line.strip()]

                with open(input_file, 'r', encoding='utf-8') as f:
                    input_samples = [json.loads(line) for line in f if line.strip()]

                if len(valid_samples) >= len(input_samples):
                    return valid_samples
            except:
                valid_samples = []
        else:
            valid_samples = []

        with open(input_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f if line.strip()]

        validated_anchors = {s['anchor_text'] for s in valid_samples}

        for sample in tqdm(samples, desc="Validating", ncols=80):
            if sample['anchor_text'] in validated_anchors:
                continue

            try:
                texts = [
                    'passage: ' + sample['anchor_text'],
                    'passage: ' + sample['text_a'],
                    'passage: ' + sample['text_b']
                ]

                embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

                sim_a = cosine_similarity([embs[0]], [embs[1]])[0][0]
                sim_b = cosine_similarity([embs[0]], [embs[2]])[0][0]
                diff = abs(sim_a - sim_b)

                checks = {
                    'correct_order': sim_a > sim_b if sample['text_a_is_closer'] else sim_b > sim_a,
                    'positive_similarity': sim_a > 0.3,
                    'reasonable_diff': 0.02 < diff < 0.5,
                    'length_ok': all(len(sample[f].split()) >= 15 for f in ['anchor_text', 'text_a', 'text_b'])
                }

                if all(checks.values()):
                    sample['sim_a'] = float(sim_a)
                    sample['sim_b'] = float(sim_b)
                    sample['diff'] = float(diff)
                    valid_samples.append(sample)
                    validated_anchors.add(sample['anchor_text'])

                    if len(valid_samples) % 50 == 0:
                        self._save_samples(valid_samples, output_file)

            except:
                continue

        self._save_samples(valid_samples, output_file)

        print(f"\n✅ 验证完成: {len(valid_samples)}/{len(samples)} ({len(valid_samples)/len(samples)*100:.1f}%)")
        return valid_samples

    def _save_samples(self, samples: List[Dict], output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

    def run_full_pipeline(self, train_path: str, api_keys: List[str], output_dir: str, resume: bool = True):
        """运行完整Pipeline"""

        patterns = self.step1_analyze_train_patterns(
            train_path=train_path,
            output_file=os.path.join(output_dir, 'train_patterns.json'),
            resume=resume
        )

        prompts = self.step2_create_generation_prompts(
            patterns=patterns,
            output_file=os.path.join(output_dir, 'generation_prompts.json'),
            resume=resume
        )

        self.step3_generate_with_gemini(
            prompts=prompts,
            api_keys=api_keys,
            output_file=os.path.join(output_dir, 'hard_negatives_raw.jsonl'),
            resume=resume
        )

        valid_samples = self.step4_validate_quality(
            input_file=os.path.join(output_dir, 'hard_negatives_raw.jsonl'),
            output_file=os.path.join(output_dir, 'hard_negatives_validated.jsonl'),
            resume=resume
        )

        print(f"\n✅ Pipeline完成: {len(valid_samples)} 样本")
        print(f"   文件: {os.path.join(output_dir, 'hard_negatives_validated.jsonl')}")


def main():
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    TRAIN_PATH = f"{PROJECT_ROOT}/TrainingSet1/synthetic_data_for_contrastive_learning.jsonl"
    OUTPUT_DIR = f"{PROJECT_ROOT}/GeminiData/HardNegatives"
    CACHE_DIR = f"{PROJECT_ROOT}/cache/hard_negatives"
    KEY_FILE = f"{PROJECT_ROOT}/config/gemini_api_keys.txt"

    API_KEYS = load_api_keys(KEY_FILE)
    print(f"✅ 加载 {len(API_KEYS)} 个API keys")

    pipeline = SafeHardNegativePipeline(
        model_path='/mnt/e/model/e5-large-v2',
        train_only=True,
        cache_dir=CACHE_DIR
    )

    pipeline.run_full_pipeline(
        train_path=TRAIN_PATH,
        api_keys=API_KEYS,
        output_dir=OUTPUT_DIR,
        resume=True
    )


if __name__ == "__main__":
    main()