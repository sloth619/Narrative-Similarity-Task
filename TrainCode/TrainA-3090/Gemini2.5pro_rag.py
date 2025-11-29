"""
改进的RAG + Gemini系统
关键改进:
1. 混合检索策略 (anchor + candidates)
2. 案例多样性采样
3. 动态prompt生成
4. 更详细的案例解释
"""
import json
import os
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


class ImprovedRAGGeminiSystem:
    """改进的RAG增强Gemini判断系统"""

    def __init__(
            self,
            embedding_model_path: str,
            case_library_path: str,
            gemini_api_keys: List[str],
            top_k: int = 5,  # 增加到5个
            use_thinking: bool = True
    ):
        self.top_k = top_k
        self.use_thinking = use_thinking
        self.gemini_keys = gemini_api_keys
        self.current_key_idx = 0

        print("📦 加载Embedding模型...")
        self.embedding_model = SentenceTransformer(embedding_model_path)
        print("   ✅ 模型加载完成")

        print("📚 加载案例库...")
        self.cases = self._load_cases(case_library_path)
        print(f"   ✅ 加载 {len(self.cases)} 个案例")

        print("🔨 构建FAISS索引...")
        self._build_index()
        print(f"   ✅ 索引构建完成\n")

    def _load_cases(self, path: str) -> List[Dict]:
        """加载案例库"""
        dataset = load_dataset('json', data_files=path, split='train')

        cases = []
        for item in dataset:
            if all([
                item.get('anchor_text'),
                item.get('text_a'),
                item.get('text_b'),
                item.get('text_a_is_closer') is not None
            ]):
                cases.append({
                    'anchor': item['anchor_text'],
                    'text_a': item['text_a'],
                    'text_b': item['text_b'],
                    'label': item['text_a_is_closer']
                })

        return cases

    def _build_index(self):
        """构建FAISS检索索引"""
        # 为每个案例构建复合embedding (anchor + text_a + text_b的平均)
        all_embeddings = []

        for case in tqdm(self.cases, desc="Building index", ncols=80):
            # 编码三个文本
            texts = [
                f"query: {case['anchor']}",
                f"passage: {case['text_a']}",
                f"passage: {case['text_b']}"
            ]
            embs = self.embedding_model.encode(texts, show_progress_bar=False)

            # 使用加权平均 (anchor权重更高)
            weighted_emb = 0.5 * embs[0] + 0.25 * embs[1] + 0.25 * embs[2]
            all_embeddings.append(weighted_emb)

        embeddings = np.array(all_embeddings).astype('float32')

        # 归一化
        faiss.normalize_L2(embeddings)
        self.anchor_embeddings = embeddings

        # 创建索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def retrieve_similar_cases(
            self,
            query_anchor: str,
            query_a: str,
            query_b: str,
            k: int = None
    ) -> List[Dict]:
        """
        改进的检索策略 - 考虑完整的三元组相似度
        """
        if k is None:
            k = self.top_k

        # 编码查询
        query_texts = [
            f"query: {query_anchor}",
            f"passage: {query_a}",
            f"passage: {query_b}"
        ]
        query_embs = self.embedding_model.encode(query_texts, show_progress_bar=False)

        # 加权平均
        weighted_query = 0.5 * query_embs[0] + 0.25 * query_embs[1] + 0.25 * query_embs[2]
        weighted_query = weighted_query.reshape(1, -1).astype('float32')
        faiss.normalize_L2(weighted_query)

        # 检索更多候选 (2倍k，后续做多样性采样)
        scores, indices = self.index.search(weighted_query, min(k * 2, len(self.cases)))

        # 构建候选案例
        candidates = []
        for idx, score in zip(indices[0], scores[0]):
            case = self.cases[idx].copy()
            case['similarity'] = float(score)
            candidates.append(case)

        # 多样性采样 - MMR (Maximal Marginal Relevance)
        selected = self._mmr_selection(candidates, k)

        return selected

    def _mmr_selection(self, candidates: List[Dict], k: int, lambda_param: float = 0.7) -> List[Dict]:
        """
        MMR多样性采样
        lambda_param: 相关性 vs 多样性的权衡 (越高越看重相关性)
        """
        if len(candidates) <= k:
            return candidates

        # 编码所有候选案例
        candidate_texts = [c['anchor'] for c in candidates]
        candidate_embs = self.embedding_model.encode(candidate_texts, show_progress_bar=False)

        selected = []
        selected_embs = []
        remaining_indices = list(range(len(candidates)))

        # 先选最相似的
        selected.append(candidates[0])
        selected_embs.append(candidate_embs[0])
        remaining_indices.remove(0)

        # 迭代选择
        while len(selected) < k and remaining_indices:
            best_score = -float('inf')
            best_idx = None

            for idx in remaining_indices:
                # 与query的相似度 (已经在candidates[idx]['similarity']中)
                relevance = candidates[idx]['similarity']

                # 与已选案例的最大相似度
                if selected_embs:
                    similarities = sklearn_cosine([candidate_embs[idx]], selected_embs)[0]
                    max_sim = np.max(similarities)
                else:
                    max_sim = 0

                # MMR分数
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(candidates[best_idx])
                selected_embs.append(candidate_embs[best_idx])
                remaining_indices.remove(best_idx)

        return selected

    def _analyze_case_pattern(self, case: Dict) -> str:
        """
        分析案例的模式 - 用embedding推断选择原因
        """
        # 编码
        texts = [
            f"passage: {case['anchor']}",
            f"passage: {case['text_a']}",
            f"passage: {case['text_b']}"
        ]
        embs = self.embedding_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        sim_a = np.dot(embs[0], embs[1])
        sim_b = np.dot(embs[0], embs[2])
        diff = abs(sim_a - sim_b)

        chosen = "A" if case['label'] else "B"
        rejected = "B" if case['label'] else "A"

        # 推断原因
        if diff < 0.05:
            reason = "subtle difference between candidates"
        elif diff < 0.10:
            reason = "moderate similarity gap"
        elif sim_a > 0.7 and sim_b > 0.6:
            reason = "both candidates highly similar, fine-grained distinction"
        elif sim_a > 0.7 or sim_b > 0.7:
            reason = f"strong semantic match with {chosen}"
        else:
            reason = "clear narrative difference"

        return reason

    def _create_improved_prompt(
            self,
            anchor: str,
            text_a: str,
            text_b: str,
            similar_cases: List[Dict]
    ) -> str:
        """改进的prompt - 提供更多上下文和推理线索"""

        # 案例分析
        case_section = "═══ REFERENCE EXAMPLES ═══\n"
        case_section += "Here are similar judgments from past cases. Pay attention to WHY each choice was made:\n\n"

        for i, case in enumerate(similar_cases, 1):
            chosen = "A" if case['label'] else "B"
            rejected = "B" if case['label'] else "A"
            reason = self._analyze_case_pattern(case)

            case_section += f"EXAMPLE {i} (Retrieval Score: {case['similarity']:.3f}):\n"
            case_section += f"Anchor: {case['anchor'][:180]}...\n"
            case_section += f"Option A: {case['text_a'][:180]}...\n"
            case_section += f"Option B: {case['text_b'][:180]}...\n"
            case_section += f"→ Decision: {chosen} (Not {rejected})\n"
            case_section += f"→ Pattern: {reason}\n\n"

        case_section += "═══════════════════════════════\n\n"

        # Guidelines精简版
        guidelines = """NARRATIVE SIMILARITY GUIDELINES (SemEval-2026 Task 4):

Three core dimensions to evaluate (weigh them based on specific context):

1. **Abstract Theme**: Central ideas, core conflicts, motivations
   - Example: "redemption through sacrifice" vs "power corrupts"
   - NOT about concrete setting (time/place/names don't matter)

2. **Course of Action**: Sequence of events and turning points
   - Example: "loss → search → recovery" vs "discovery → investigation → revelation"
   - ORDER matters: same events in different sequence = different course

3. **Outcomes**: Final resolution and character fates
   - Example: tragic ending vs happy ending
   - Even identical events can have opposite outcomes

CRITICAL: Setting, style, names, and length DO NOT affect similarity!
Example: Medieval knight story can be similar to space astronaut story if theme/course/outcome align.
"""

        # 完整prompt
        prompt = f"""{guidelines}

{case_section}

Now analyze this NEW CASE:

═══════════════════════════════════════════════════════════
ANCHOR STORY:
{anchor}

CANDIDATE A:
{text_a}

CANDIDATE B:
{text_b}
═══════════════════════════════════════════════════════════

TASK:
Based on the three dimensions (Theme, Course of Action, Outcomes) and the reference examples above, determine which candidate is MORE narratively similar to the Anchor.

Consider:
- Which shares more aspects (theme/course/outcome) with the anchor?
- Are there any traps? (e.g., same setting but different narrative, or similar events but opposite outcomes)
- What patterns do you see from the reference examples?

Think step-by-step, then respond with ONLY the letter: A or B

Your answer:"""

        return prompt

    def _call_gemini(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """调用Gemini API"""
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
        ]

        for retry in range(max_retries):
            try:
                api_key = self.gemini_keys[self.current_key_idx]
                self.current_key_idx = (self.current_key_idx + 1) % len(self.gemini_keys)

                time.sleep(2)
                client = genai.Client(api_key=api_key)

                config = types.GenerateContentConfig(
                    temperature=0.1,  # 降低temperature提高稳定性
                    max_output_tokens=8192,
                    top_k=20,  # 降低随机性
                    top_p=0.9,
                    safety_settings=safety_settings,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=2048 if self.use_thinking else 0  # 增加thinking
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

                if final_text:
                    return final_text

                time.sleep(1)

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    time.sleep(30)
                else:
                    time.sleep(3)

        return None

    def predict(self, anchor: str, text_a: str, text_b: str) -> Dict:
        """RAG增强的预测"""
        # 1. 混合检索 - 考虑完整三元组
        similar_cases = self.retrieve_similar_cases(anchor, text_a, text_b)

        # 2. 构建改进的prompt
        prompt = self._create_improved_prompt(anchor, text_a, text_b, similar_cases)

        # 3. 调用Gemini
        response_text = self._call_gemini(prompt)

        if response_text is None:
            # 降级: 基于检索案例的加权投票
            weights = [c['similarity'] for c in similar_cases]
            weighted_votes = sum(w if c['label'] else 0 for w, c in zip(weights, similar_cases))
            total_weight = sum(weights)
            prediction = weighted_votes > total_weight / 2

            return {
                'prediction': prediction,
                'retrieved_cases': len(similar_cases),
                'avg_similarity': float(np.mean([c['similarity'] for c in similar_cases])),
                'method': 'weighted_voting_fallback',
                'confidence': abs(weighted_votes / total_weight - 0.5)
            }

        # 4. 解析答案
        answer = response_text.strip().upper()

        # 更鲁棒的解析
        if answer == 'A' or answer.endswith(' A') or answer.endswith('\nA'):
            prediction = True
        elif answer == 'B' or answer.endswith(' B') or answer.endswith('\nB'):
            prediction = False
        elif 'A' in answer[-10:] and 'B' not in answer[-10:]:  # 看最后10个字符
            prediction = True
        elif 'B' in answer[-10:] and 'A' not in answer[-10:]:
            prediction = False
        else:
            # 降级
            weights = [c['similarity'] for c in similar_cases]
            weighted_votes = sum(w if c['label'] else 0 for w, c in zip(weights, similar_cases))
            total_weight = sum(weights)
            prediction = weighted_votes > total_weight / 2

        return {
            'prediction': prediction,
            'retrieved_cases': len(similar_cases),
            'avg_similarity': float(np.mean([c['similarity'] for c in similar_cases])),
            'gemini_response': response_text[-100:],  # 只保存最后100字符
            'method': 'improved_rag_gemini'
        }

    def evaluate(self, test_file: str, output_file: str = None):
        """评估系统"""
        dataset = load_dataset('json', data_files=test_file, split='train')

        results = []
        correct = 0
        total = 0

        pbar = tqdm(dataset, desc="Improved RAG评估", ncols=100)

        for item in pbar:
            anchor = item.get('anchor_text') or item.get('anchor_story')
            text_a = item.get('text_a') or item.get('similar_story')
            text_b = item.get('text_b') or item.get('dissimilar_story')
            label = item.get('text_a_is_closer')

            if not all([anchor, text_a, text_b]) or label is None:
                continue

            pred_result = self.predict(anchor, text_a, text_b)

            if pred_result['prediction'] == label:
                correct += 1
            total += 1

            results.append({
                'text_a_is_closer': pred_result['prediction'],
                'label': label,
                'correct': pred_result['prediction'] == label,
                **pred_result
            })

            # 更新进度
            accuracy = correct / total if total > 0 else 0
            pbar.set_postfix({
                'acc': f"{accuracy:.4f}",
                'n': f"{correct}/{total}"
            })

        pbar.close()

        accuracy = correct / total if total > 0 else 0

        print(f"\n{'=' * 70}")
        print("📊 改进版RAG+Gemini 评估结果:")
        print(f"{'=' * 70}")
        print(f"准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"正确: {correct}/{total}")

        # 统计方法使用
        method_counts = {}
        for r in results:
            method = r.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1

        print(f"\n方法统计:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} ({count / len(results) * 100:.1f}%)")

        print(f"{'=' * 70}\n")

        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps({
                        'text_a_is_closer': r['text_a_is_closer']
                    }, ensure_ascii=False) + '\n')

            detail_file = output_file.replace('.jsonl', '_detail.jsonl')
            with open(detail_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

            print(f"✅ 提交文件: {output_file}")
            print(f"✅ 详细文件: {detail_file}")

        return accuracy

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
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="RAG+Gemini")

        for idx in range(start_idx, len(dataset)):
            item = dataset[idx]

            anchor = item.get('anchor_text') or item.get('anchor_story')
            text_a = item.get('text_a') or item.get('similar_story')
            text_b = item.get('text_b') or item.get('dissimilar_story')

            if not all([anchor, text_a, text_b]):
                results.append({'text_a_is_closer': True})
                pbar.update(1)
                continue

            pred_result = self.predict(anchor, text_a, text_b)

            results.append({
                'text_a_is_closer': pred_result['prediction'],
                'retrieved_cases': pred_result['retrieved_cases'],
                'avg_similarity': pred_result['avg_similarity'],
                'method': pred_result['method']
            })

            # 定期保存
            if (idx + 1) % save_interval == 0:
                self._save_results(results, output_file)

            pbar.update(1)

        pbar.close()
        self._save_results(results, output_file)

        # 统计
        a_count = sum(1 for r in results if r['text_a_is_closer'])
        print(f"\n✅ 完成!")
        print(f"   总样本: {len(results)}")
        print(f"   分布: A={a_count}, B={len(results)-a_count}")
        print(f"   文件: {output_file}")


def load_api_keys(key_file: str) -> List[str]:
    """加载API keys"""
    with open(key_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]


def main():
    PROJECT_ROOT = "/home/songfeiyang/workspace/semeval"

    # 配置
    E5_MODEL = "/home/songfeiyang/workspace/model/E5_0.695"
    CASE_LIBRARY = f"{PROJECT_ROOT}/TrainSet/synthetic_data_for_contrastive_learning.jsonl"
    DEV_FILE = f"{PROJECT_ROOT}/TrainSet/dev_track_a.jsonl"

    GEMINI_KEY_FILE = f"{PROJECT_ROOT}/config/gemini_api_keys.txt"
    GEMINI_KEYS = load_api_keys(GEMINI_KEY_FILE)

    OUTPUT_DIR = f"{PROJECT_ROOT}/submissions/improved_rag_gemini"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("🔍 改进版 RAG + Gemini 2.5 Pro")
    print("=" * 70)
    print(f"案例库: {CASE_LIBRARY}")
    print(f"检索Top-K: 5 (with MMR diversity)")
    print(f"Gemini Keys: {len(GEMINI_KEYS)}个")
    print(f"改进: 混合检索 + MMR采样 + 详细prompt")
    print("=" * 70 + "\n")

    # 创建系统
    rag_system = ImprovedRAGGeminiSystem(
        embedding_model_path=E5_MODEL,
        case_library_path=CASE_LIBRARY,
        gemini_api_keys=GEMINI_KEYS,
        top_k=5,
        use_thinking=True
    )

    # 评估
    print("📊 在dev集上评估...\n")
    accuracy = rag_system.evaluate(
        test_file=DEV_FILE,
        output_file=f"{OUTPUT_DIR}/track_a.jsonl"
    )

    print(f"\n🎯 最终准确率: {accuracy * 100:.2f}%")
    print(f"vs 原版: 0.72 → {accuracy:.4f} (+{(accuracy - 0.72) * 100:.2f}%)")


if __name__ == "__main__":
    main()