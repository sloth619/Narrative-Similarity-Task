"""
三模型Ensemble - E5 + Gemini 2.5 Pro + Qwen3-Max
"""
import json
import time
import os
from typing import List, Dict, Optional
from tqdm import tqdm
from google import genai
from google.genai import types
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import numpy as np


class TripleModelEnsemble:
    """E5 + Gemini 2.5 Pro + Qwen3-Max 三模型集成"""

    def __init__(
            self,
            e5_model_path: str,
            gemini_api_keys: List[str],
            qwen_api_keys: List[str],
            e5_weight: float = 0.3,
            gemini_weight: float = 0.4,
            qwen_weight: float = 0.3,
            use_gemini_thinking: bool = True
    ):
        """
        Args:
            e5_model_path: E5模型路径
            gemini_api_keys: Gemini API keys
            qwen_api_keys: Qwen API keys
            e5_weight: E5权重
            gemini_weight: Gemini权重
            qwen_weight: Qwen权重
            use_gemini_thinking: Gemini是否使用思考模式
        """
        assert abs(e5_weight + gemini_weight + qwen_weight - 1.0) < 0.001, "权重之和必须为1"

        self.e5_weight = e5_weight
        self.gemini_weight = gemini_weight
        self.qwen_weight = qwen_weight
        self.use_gemini_thinking = use_gemini_thinking

        # 加载E5
        print("📦 加载E5模型...")
        self.e5_model = SentenceTransformer(e5_model_path)
        print(f"   ✅ E5加载成功")

        # API keys
        self.gemini_keys = gemini_api_keys
        self.qwen_keys = qwen_api_keys
        self.gemini_key_index = 0
        self.qwen_key_index = 0

        print(f"   Gemini Keys: {len(gemini_api_keys)}个")
        print(f"   Qwen Keys: {len(qwen_api_keys)}个")

    def _predict_e5(self, anchor: str, text_a: str, text_b: str) -> float:
        """E5预测 - 返回0-1置信度"""
        anchor_prefixed = f"query: {anchor}"
        text_a_prefixed = f"passage: {text_a}"
        text_b_prefixed = f"passage: {text_b}"

        embeddings = self.e5_model.encode(
            [anchor_prefixed, text_a_prefixed, text_b_prefixed],
            convert_to_tensor=True,
            show_progress_bar=False
        )

        sim_a = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        ).item()

        sim_b = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[2].unsqueeze(0)
        ).item()

        diff = sim_a - sim_b
        confidence = 1 / (1 + np.exp(-10 * diff))

        return confidence

    def _predict_gemini(self, anchor: str, text_a: str, text_b: str) -> Optional[float]:
        """Gemini 2.5 Pro预测"""
        prompt = f"""You are an expert in narrative analysis. Compare three stories to determine narrative similarity.

NARRATIVE SIMILARITY:
1. **Abstract Theme** (30%): Core ideas, conflicts, motifs
2. **Course of Action** (40%): Event sequence, plot structure
3. **Outcomes** (30%): Final resolution, character fates

ANCHOR STORY:
{anchor}

CANDIDATE A:
{text_a}

CANDIDATE B:
{text_b}

Which candidate (A or B) is MORE narratively similar to the Anchor?
Respond with ONLY: A or B"""

        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
        ]

        for retry in range(2):
            try:
                api_key = self.gemini_keys[self.gemini_key_index]
                self.gemini_key_index = (self.gemini_key_index + 1) % len(self.gemini_keys)

                time.sleep(2)
                client = genai.Client(api_key=api_key)

                config = types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                    top_k=40,
                    top_p=0.95,
                    safety_settings=safety_settings,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=1024 if self.use_gemini_thinking else 0
                    )
                )

                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=config
                )

                final_text = None
                if response.text:
                    final_text = response.text
                elif response.candidates and response.candidates[0].content.parts:
                    final_text = " ".join([p.text for p in response.candidates[0].content.parts if p.text])

                if not final_text:
                    continue

                answer = final_text.strip().upper()

                if 'A' in answer and 'B' not in answer:
                    return 1.0
                elif 'B' in answer and 'A' not in answer:
                    return 0.0
                elif answer == 'A':
                    return 1.0
                elif answer == 'B':
                    return 0.0

                time.sleep(1)

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(20)
                else:
                    time.sleep(3)

        return None

    def _predict_qwen(self, anchor: str, text_a: str, text_b: str) -> Optional[float]:
        """Qwen3-Max预测"""
        prompt = f"""You are an expert in narrative analysis. Compare three stories to determine narrative similarity.

NARRATIVE SIMILARITY:
1. **Abstract Theme** (30%): Core ideas, conflicts, motifs
2. **Course of Action** (40%): Event sequence, plot structure
3. **Outcomes** (30%): Final resolution, character fates

ANCHOR STORY:
{anchor}

CANDIDATE A:
{text_a}

CANDIDATE B:
{text_b}

Which candidate (A or B) is MORE narratively similar to the Anchor?
Respond with ONLY: A or B"""

        for retry in range(2):
            try:
                api_key = self.qwen_keys[self.qwen_key_index]
                self.qwen_key_index = (self.qwen_key_index + 1) % len(self.qwen_keys)

                time.sleep(1)
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )

                response = client.chat.completions.create(
                    model="qwen3-max",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=8192,
                    top_p=0.95,
                )

                final_text = response.choices[0].message.content

                if not final_text:
                    continue

                answer = final_text.strip().upper()

                if 'A' in answer and 'B' not in answer:
                    return 1.0
                elif 'B' in answer and 'A' not in answer:
                    return 0.0
                elif answer == 'A':
                    return 1.0
                elif answer == 'B':
                    return 0.0

                time.sleep(1)

            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(15)
                else:
                    time.sleep(3)

        return None

    def predict(self, anchor: str, text_a: str, text_b: str) -> Dict:
        """三模型集成预测"""
        # E5预测
        e5_conf = self._predict_e5(anchor, text_a, text_b)

        # Gemini预测
        gemini_conf = self._predict_gemini(anchor, text_a, text_b)

        # Qwen预测
        qwen_conf = self._predict_qwen(anchor, text_a, text_b)

        # 统计成功的模型数
        valid_models = []
        if e5_conf is not None:
            valid_models.append('e5')
        if gemini_conf is not None:
            valid_models.append('gemini')
        if qwen_conf is not None:
            valid_models.append('qwen')

        # 根据可用模型动态调整权重
        if len(valid_models) == 3:
            # 三个都成功
            ensemble_conf = (
                    self.e5_weight * e5_conf +
                    self.gemini_weight * gemini_conf +
                    self.qwen_weight * qwen_conf
            )
            method = 'triple_ensemble'

        elif len(valid_models) == 2:
            # 两个成功
            if 'e5' in valid_models and 'gemini' in valid_models:
                # E5 + Gemini
                total = self.e5_weight + self.gemini_weight
                ensemble_conf = (self.e5_weight / total * e5_conf +
                                 self.gemini_weight / total * gemini_conf)
                method = 'e5_gemini'

            elif 'e5' in valid_models and 'qwen' in valid_models:
                # E5 + Qwen
                total = self.e5_weight + self.qwen_weight
                ensemble_conf = (self.e5_weight / total * e5_conf +
                                 self.qwen_weight / total * qwen_conf)
                method = 'e5_qwen'

            else:  # gemini + qwen
                # Gemini + Qwen
                total = self.gemini_weight + self.qwen_weight
                ensemble_conf = (self.gemini_weight / total * gemini_conf +
                                 self.qwen_weight / total * qwen_conf)
                method = 'gemini_qwen'

        elif len(valid_models) == 1:
            # 只有一个成功
            if 'e5' in valid_models:
                ensemble_conf = e5_conf
                method = 'e5_only'
            elif 'gemini' in valid_models:
                ensemble_conf = gemini_conf
                method = 'gemini_only'
            else:
                ensemble_conf = qwen_conf
                method = 'qwen_only'
        else:
            # 全部失败,默认True
            ensemble_conf = 0.51
            method = 'fallback'

        # 计算一致性
        predictions = []
        if e5_conf is not None:
            predictions.append(e5_conf > 0.5)
        if gemini_conf is not None:
            predictions.append(gemini_conf > 0.5)
        if qwen_conf is not None:
            predictions.append(qwen_conf > 0.5)

        # 判断是否一致 (至少2/3一致)
        if len(predictions) >= 2:
            agreement = sum(predictions) >= len(predictions) / 2
        else:
            agreement = True

        return {
            'prediction': ensemble_conf > 0.5,
            'e5_confidence': e5_conf,
            'gemini_confidence': gemini_conf,
            'qwen_confidence': qwen_conf,
            'ensemble_confidence': ensemble_conf,
            'agreement': agreement,
            'method': method,
            'valid_models': len(valid_models)
        }

    def generate_submission(
            self,
            test_file: str,
            output_file: str,
            save_interval: int = 5
    ):
        """生成提交文件"""
        dataset = load_dataset('json', data_files=test_file, split='train')

        results = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = [json.loads(line) for line in f if line.strip()]
            except:
                results = []

        start_idx = len(results)
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="三模型Ensemble")

        stats = {
            'total': 0,
            'triple': 0,
            'double': 0,
            'single': 0,
            'fallback': 0,
            'agreement': 0,
            'disagreement': 0
        }

        for idx in range(start_idx, len(dataset)):
            item = dataset[idx]

            anchor = item.get('anchor_text') or item.get('anchor_story')
            text_a = item.get('text_a') or item.get('similar_story')
            text_b = item.get('text_b') or item.get('dissimilar_story')

            if not all([anchor, text_a, text_b]):
                results.append({
                    'text_a_is_closer': True,
                    'method': 'default'
                })
                pbar.update(1)
                continue

            # 三模型预测
            pred_result = self.predict(anchor, text_a, text_b)

            # 统计
            stats['total'] += 1
            if pred_result['valid_models'] == 3:
                stats['triple'] += 1
            elif pred_result['valid_models'] == 2:
                stats['double'] += 1
            elif pred_result['valid_models'] == 1:
                stats['single'] += 1
            else:
                stats['fallback'] += 1

            if pred_result['agreement']:
                stats['agreement'] += 1
            else:
                stats['disagreement'] += 1

            # 保存结果
            results.append({
                'text_a_is_closer': pred_result['prediction'],
                'e5_confidence': pred_result['e5_confidence'],
                'gemini_confidence': pred_result['gemini_confidence'],
                'qwen_confidence': pred_result['qwen_confidence'],
                'ensemble_confidence': pred_result['ensemble_confidence'],
                'agreement': pred_result['agreement'],
                'method': pred_result['method'],
                'valid_models': pred_result['valid_models']
            })

            # 定期保存
            if (idx + 1) % save_interval == 0:
                self._save_results(results, output_file)

            # 更新进度
            pbar.set_postfix({
                '3模型': stats['triple'],
                '2模型': stats['double'],
                '一致': f"{stats['agreement']}/{stats['total']}"
            })
            pbar.update(1)

        pbar.close()
        self._save_results(results, output_file)

        # 打印统计
        if stats['total'] > 0:
            print(f"\n{'=' * 60}")
            print("📊 三模型Ensemble统计:")
            print(f"{'=' * 60}")
            print(f"总样本: {stats['total']}")
            print(f"三模型成功: {stats['triple']} ({stats['triple'] / stats['total'] * 100:.1f}%)")
            print(f"两模型成功: {stats['double']} ({stats['double'] / stats['total'] * 100:.1f}%)")
            print(f"单模型成功: {stats['single']} ({stats['single'] / stats['total'] * 100:.1f}%)")
            print(f"降级处理: {stats['fallback']}")
            print(f"\n一致性: {stats['agreement']} ({stats['agreement'] / stats['total'] * 100:.1f}%)")
            print(f"不一致: {stats['disagreement']} ({stats['disagreement'] / stats['total'] * 100:.1f}%)")

            a_count = sum(1 for r in results if r['text_a_is_closer'])
            print(f"\n最终分布: A={a_count}, B={len(results) - a_count}")
            print(f"提交文件: {output_file}")
            print(f"{'=' * 60}\n")

    def _save_results(self, results: List[Dict], filepath: str):
        """保存结果"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        def to_python(val):
            if val is None:
                return None
            if hasattr(val, 'item'):
                return val.item()
            if hasattr(val, 'tolist'):
                return val.tolist()
            return val

        # 保存提交文件
        with open(filepath, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps({
                    'text_a_is_closer': to_python(r['text_a_is_closer'])
                }, ensure_ascii=False) + '\n')

        # 保存详细结果
        detail_file = filepath.replace('.jsonl', '_detail.jsonl')
        with open(detail_file, 'w', encoding='utf-8') as f:
            for r in results:
                clean = {k: to_python(v) for k, v in r.items()}
                f.write(json.dumps(clean, ensure_ascii=False) + '\n')


def load_api_keys(key_file: str) -> List[str]:
    """加载API keys"""
    with open(key_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]


def main():
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    # 模型路径
    E5_MODEL_PATH = f"{PROJECT_ROOT}/output/GoodModel/E5_0.695"

    # API keys
    GEMINI_KEY_FILE = f"{PROJECT_ROOT}/config/gemini_api_keys.txt"
    QWEN_KEY_FILE = f"{PROJECT_ROOT}/config/qwen_api_keys.txt"

    GEMINI_KEYS = load_api_keys(GEMINI_KEY_FILE)
    QWEN_KEYS = load_api_keys(QWEN_KEY_FILE)

    # 测试集
    TEST_FILE = f"{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl"
    # TEST_FILE = f"{PROJECT_ROOT}/test/track_a.jsonl"  # 真实提交

    # 输出
    OUTPUT_DIR = f"{PROJECT_ROOT}/submissions/triple_ensemble"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'=' * 60}")
    print("🎯 三模型Ensemble: E5 + Gemini + Qwen")
    print(f"{'=' * 60}")
    print(f"E5模型: {E5_MODEL_PATH}")
    print(f"Gemini Keys: {len(GEMINI_KEYS)}个")
    print(f"Qwen Keys: {len(QWEN_KEYS)}个")
    print(f"测试集: {TEST_FILE}")
    print(f"{'=' * 60}\n")

    # 创建Ensemble
    ensemble = TripleModelEnsemble(
        e5_model_path=E5_MODEL_PATH,
        gemini_api_keys=GEMINI_KEYS,
        qwen_api_keys=QWEN_KEYS,
        e5_weight=0.3,
        gemini_weight=0.4,
        qwen_weight=0.3,
        use_gemini_thinking=True
    )

    # 生成预测
    ensemble.generate_submission(
        test_file=TEST_FILE,
        output_file=f"{OUTPUT_DIR}/track_a.jsonl",
        save_interval=5
    )

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()