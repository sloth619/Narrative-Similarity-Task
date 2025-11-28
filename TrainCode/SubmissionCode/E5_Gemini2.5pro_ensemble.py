"""
Ensemble预测 - E5-large-v2 + Gemini 2.5 Pro
"""
import json
import time
import os
from typing import List, Dict, Optional
from tqdm import tqdm
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import numpy as np


class E5Gemini2ProEnsemble:
    """E5和Gemini 2.5 Pro的集成预测器"""

    def __init__(
            self,
            e5_model_path: str,
            gemini_api_keys: List[str],
            e5_weight: float = 0.5,
            gemini_weight: float = 0.5,
            use_thinking: bool = True
    ):
        """
        Args:
            e5_model_path: E5模型路径
            gemini_api_keys: Gemini API keys列表
            e5_weight: E5的权重
            gemini_weight: Gemini的权重
            use_thinking: Gemini是否使用思考模式
        """
        self.e5_weight = e5_weight
        self.gemini_weight = gemini_weight
        self.use_thinking = use_thinking

        # 加载E5模型
        print("📦 加载E5模型...")
        self.e5_model = SentenceTransformer(e5_model_path)
        print(f"   ✅ E5模型加载成功")

        # Gemini API配置
        self.gemini_keys = gemini_api_keys
        self.current_key_index = 0

    def _predict_e5(self, anchor: str, text_a: str, text_b: str) -> float:
        """
        E5预测
        Returns: 0.0-1.0的置信度 (1.0表示完全倾向A, 0.0表示完全倾向B)
        """
        # 添加E5前缀
        anchor_prefixed = f"query: {anchor}"
        text_a_prefixed = f"passage: {text_a}"
        text_b_prefixed = f"passage: {text_b}"

        # 编码
        embeddings = self.e5_model.encode(
            [anchor_prefixed, text_a_prefixed, text_b_prefixed],
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # 计算余弦相似度
        sim_a = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        ).item()

        sim_b = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[2].unsqueeze(0)
        ).item()

        # 归一化到0-1 (sigmoid)
        diff = sim_a - sim_b
        confidence = 1 / (1 + np.exp(-10 * diff))

        return confidence

    def _predict_gemini(self, anchor: str, text_a: str, text_b: str) -> Optional[float]:
        """
        Gemini 2.5 Pro预测
        Returns: 0.0-1.0的置信度 (1.0表示A, 0.0表示B), None表示失败
        """
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

        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
        ]

        for retry in range(3):
            try:
                api_key = self.gemini_keys[self.current_key_index]
                self.current_key_index = (self.current_key_index + 1) % len(self.gemini_keys)

                time.sleep(2)
                client = genai.Client(api_key=api_key)

                config = types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                    top_k=40,
                    top_p=0.95,
                    safety_settings=safety_settings,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=1024 if self.use_thinking else 0
                    )
                )

                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=config
                )

                # 解析响应
                final_text = None
                if response.text:
                    final_text = response.text
                elif response.candidates and response.candidates[0].content.parts:
                    final_text = " ".join([p.text for p in response.candidates[0].content.parts if p.text])

                if not final_text:
                    continue

                answer = final_text.strip().upper()

                # 解析答案
                if 'A' in answer and 'B' not in answer:
                    return 1.0
                elif 'B' in answer and 'A' not in answer:
                    return 0.0
                elif answer == 'A':
                    return 1.0
                elif answer == 'B':
                    return 0.0
                else:
                    # 更宽松的解析
                    if "OUTPUT: A" in answer or answer.endswith("A") or "ANSWER: A" in answer:
                        return 1.0
                    elif "OUTPUT: B" in answer or answer.endswith("B") or "ANSWER: B" in answer:
                        return 0.0
                    else:
                        lines = [l.strip() for l in answer.split('\n') if l.strip()]
                        if lines and lines[-1] == 'A':
                            return 1.0
                        elif lines and lines[-1] == 'B':
                            return 0.0

                time.sleep(2)

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    time.sleep(30)
                else:
                    time.sleep(5)

        return None

    def predict(self, anchor: str, text_a: str, text_b: str) -> Dict:
        """
        集成预测
        Returns:
            {
                'prediction': bool,
                'e5_confidence': float,
                'gemini_confidence': float,
                'ensemble_confidence': float,
                'agreement': bool,
                'method': str
            }
        """
        # E5预测
        e5_conf = self._predict_e5(anchor, text_a, text_b)

        # Gemini预测
        gemini_conf = self._predict_gemini(anchor, text_a, text_b)

        # 如果Gemini失败,只用E5
        if gemini_conf is None:
            return {
                'prediction': e5_conf > 0.5,
                'e5_confidence': e5_conf,
                'gemini_confidence': None,
                'ensemble_confidence': e5_conf,
                'agreement': True,
                'method': 'e5_only'
            }

        # 集成 (加权平均)
        ensemble_conf = (
                self.e5_weight * e5_conf +
                self.gemini_weight * gemini_conf
        )

        # 判断是否一致
        e5_pred = e5_conf > 0.5
        gemini_pred = gemini_conf > 0.5
        agreement = e5_pred == gemini_pred

        return {
            'prediction': ensemble_conf > 0.5,
            'e5_confidence': e5_conf,
            'gemini_confidence': gemini_conf,
            'ensemble_confidence': ensemble_conf,
            'agreement': agreement,
            'method': 'ensemble'
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
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="Ensemble预测", ncols=100)

        stats = {
            'total': 0,
            'e5_only': 0,
            'ensemble': 0,
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

            # 集成预测
            pred_result = self.predict(anchor, text_a, text_b)

            # 记录统计
            stats['total'] += 1
            if pred_result['method'] == 'e5_only':
                stats['e5_only'] += 1
            else:
                stats['ensemble'] += 1
                if pred_result['agreement']:
                    stats['agreement'] += 1
                else:
                    stats['disagreement'] += 1

            # 保存结果
            results.append({
                'text_a_is_closer': pred_result['prediction'],
                'e5_confidence': pred_result['e5_confidence'],
                'gemini_confidence': pred_result['gemini_confidence'],
                'ensemble_confidence': pred_result['ensemble_confidence'],
                'agreement': pred_result['agreement'],
                'method': pred_result['method']
            })

            # 定期保存
            if (idx + 1) % save_interval == 0:
                self._save_results(results, output_file)

            # 更新进度条
            pbar.set_postfix({
                'agree': f"{stats['agreement']}/{stats['ensemble']}" if stats['ensemble'] > 0 else "0/0",
                'e5_only': stats['e5_only']
            })
            pbar.update(1)

        pbar.close()

        # 最终保存
        self._save_results(results, output_file)

        # 打印统计
        print(f"\n{'=' * 60}")
        print("📊 Ensemble统计:")
        print(f"{'=' * 60}")
        print(f"总样本: {stats['total']}")
        print(f"使用Ensemble: {stats['ensemble']} ({stats['ensemble'] / stats['total'] * 100:.1f}%)")
        print(f"只用E5: {stats['e5_only']} ({stats['e5_only'] / stats['total'] * 100:.1f}%)")

        if stats['ensemble'] > 0:
            print(f"\nEnsemble详情:")
            print(f"  一致: {stats['agreement']} ({stats['agreement'] / stats['ensemble'] * 100:.1f}%)")
            print(f"  不一致: {stats['disagreement']} ({stats['disagreement'] / stats['ensemble'] * 100:.1f}%)")

        a_count = sum(1 for r in results if r['text_a_is_closer'])
        print(f"\n最终分布: A={a_count}, B={len(results) - a_count}")
        print(f"提交文件: {output_file}")
        print(f"详细文件: {output_file.replace('.jsonl', '_detail.jsonl')}")
        print(f"{'=' * 60}\n")

    def _save_results(self, results: List[Dict], filepath: str):
        """保存结果 - 修复numpy类型序列化问题"""
        import numpy as np

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        def make_serializable(value):
            """转换numpy类型为Python原生类型"""
            if value is None:
                return None
            # 布尔类型
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            # 整数类型
            if isinstance(value, (int, np.integer)):
                return int(value)
            # 浮点类型
            if isinstance(value, (float, np.floating)):
                return float(value)
            # 数组类型
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        # 1. 保存提交文件 (只有text_a_is_closer)
        with open(filepath, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps({
                    'text_a_is_closer': make_serializable(r['text_a_is_closer'])
                }, ensure_ascii=False) + '\n')

        # 2. 保存详细结果 (包含所有信息)
        detail_file = filepath.replace('.jsonl', '_detail.jsonl')
        with open(detail_file, 'w', encoding='utf-8') as f:
            for r in results:
                clean_result = {
                    'text_a_is_closer': make_serializable(r['text_a_is_closer']),
                    'e5_confidence': make_serializable(r.get('e5_confidence')),
                    'gemini_confidence': make_serializable(r.get('gemini_confidence')),
                    'ensemble_confidence': make_serializable(r.get('ensemble_confidence')),
                    'agreement': make_serializable(r.get('agreement')),
                    'method': r.get('method', 'unknown')
                }
                f.write(json.dumps(clean_result, ensure_ascii=False) + '\n')


def load_api_keys(key_file: str) -> List[str]:
    """加载API keys"""
    with open(key_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]


def main():

    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    E5_MODEL_PATH = f"{PROJECT_ROOT}/output/GoodModel/E5_0.695"

    KEY_FILE = f"{PROJECT_ROOT}/config/gemini_api_keys.txt"
    GEMINI_KEYS = load_api_keys(KEY_FILE)

    TEST_FILE = f"{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl"

    OUTPUT_DIR = f"{PROJECT_ROOT}/submissions/e5_gemini_ensemble"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'=' * 60}")
    print("🎯 E5 + Gemini 2.5 Pro Ensemble")
    print(f"{'=' * 60}")
    print(f"E5模型: {E5_MODEL_PATH}")
    print(f"Gemini Keys: {len(GEMINI_KEYS)}个")
    print(f"测试集: {TEST_FILE}")
    print(f"输出: {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    ensemble = E5Gemini2ProEnsemble(
        e5_model_path=E5_MODEL_PATH,
        gemini_api_keys=GEMINI_KEYS,
        e5_weight=0.3,
        gemini_weight=0.7,
        use_thinking=True  # 使用思考模式
    )

    # ===== 生成预测 =====
    ensemble.generate_submission(
        test_file=TEST_FILE,
        output_file=f"{OUTPUT_DIR}/track_a.jsonl",
        save_interval=5
    )

    print("\n✅ 完成!")


if __name__ == "__main__":
    main()