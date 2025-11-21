"""
终极方案: 找出E5最佳的单个配置
测试"passage"和"query_similar"在不同参数下的表现
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import numpy as np


def test_e5_with_variations(data_path):
    """
    测试E5的不同配置组合
    1. 不同prompt
    2. 不同normalize方式
    3. 不同相似度计算
    """
    print("🔬 E5-large 深度优化")
    print("=" * 70)

    # 加载数据
    dataset = load_dataset('json', data_files=data_path, split='train')

    clean_data = []
    for item in dataset:
        anchor = item.get('anchor_text')
        text_a = item.get('text_a')
        text_b = item.get('text_b')
        label = item.get('text_a_is_closer')

        if all([anchor, text_a, text_b, label is not None]):
            clean_data.append({
                'anchor': anchor,
                'text_a': text_a,
                'text_b': text_b,
                'label': 'A' if label else 'B'
            })

    print(f"✅ 测试样本: {len(clean_data)}\n")

    # 加载模型
    model = SentenceTransformer('/mnt/e/model/e5-large-v2')

    # 配置矩阵
    configs = [
        # (prompt, normalize, name)
        ('passage: ', True, 'passage_normalized'),
        ('passage: ', False, 'passage_no_norm'),
        ('query: find similar stories: ', True, 'query_similar_normalized'),
        ('query: find similar stories: ', False, 'query_similar_no_norm'),
        ('', True, 'no_prompt_normalized'),
        ('', False, 'no_prompt_no_norm'),
    ]

    results = []

    for prompt, normalize, name in configs:
        print(f"测试配置: {name}")

        correct = 0
        for sample in clean_data:
            # 编码
            texts = [prompt + t for t in [sample['anchor'], sample['text_a'], sample['text_b']]]
            embeddings = model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )

            # 计算相似度
            sim_a = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            sim_b = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

            pred = 'A' if sim_a > sim_b else 'B'
            if pred == sample['label']:
                correct += 1

        accuracy = correct / len(clean_data)
        results.append((name, accuracy, prompt, normalize))

        print(f"  准确率: {accuracy:.4f} ({correct}/{len(clean_data)})\n")

    # 排序显示
    print("=" * 70)
    print("📊 配置排序")
    print("=" * 70)

    results.sort(key=lambda x: x[1], reverse=True)

    for i, (name, acc, prompt, norm) in enumerate(results, 1):
        marker = "🏆" if i == 1 else f"{i}."
        print(f"{marker} {name:30s}: {acc:.4f} ({acc*100:.2f}%)")

    return results


def final_verification(best_config):
    """最终验证最佳配置"""
    print("\n" + "=" * 70)
    print("🎯 最佳配置验证")
    print("=" * 70)

    name, acc, prompt, normalize = best_config

    print(f"配置名称: {name}")
    print(f"Prompt: '{prompt}'")
    print(f"Normalize: {normalize}")
    print(f"准确率: {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 70)

    # 保存最终配置
    import json

    config = {
        'model': 'intfloat/e5-large-v2',
        'model_path': '/mnt/e/model/e5-large-v2',
        'prompt': prompt,
        'normalize_embeddings': normalize,
        'accuracy': float(acc),
        'accuracy_percentage': f"{acc*100:.2f}%",
        'usage': f"""
model = SentenceTransformer('/mnt/e/model/e5-large-v2')

def predict(anchor, text_a, text_b):
    texts = ['{prompt}' + t for t in [anchor, text_a, text_b]]
    embeddings = model.encode(texts, normalize_embeddings={normalize})
    
    sim_a = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    sim_b = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
    
    return 'A' if sim_a > sim_b else 'B'
        """
    }

    output_path = '/mnt/e/Code/python/Narrative-Similarity-Task/output/FINAL_BEST_CONFIG.json'
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 最终配置已保存: {output_path}")

    return config


def create_production_code(config):
    """生成生产代码"""

    code = f'''"""
生产环境最佳配置
模型: E5-large-v2
准确率: {config['accuracy_percentage']}
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class NarrativeSimilarityModel:
    """叙事相似度预测模型 - 最优配置"""
    
    def __init__(self):
        self.model = SentenceTransformer('{config['model_path']}')
        self.prompt = "{config['prompt']}"
        self.normalize = {config['normalize_embeddings']}
    
    def predict(self, anchor: str, text_a: str, text_b: str) -> str:
        """
        预测哪个文本与anchor更相似
        
        Returns:
            'A' or 'B'
        """
        texts = [self.prompt + t for t in [anchor, text_a, text_b]]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        
        sim_a = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        sim_b = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        
        return 'A' if sim_a > sim_b else 'B'
    
    def get_similarity_scores(self, anchor: str, text_a: str, text_b: str):
        """获取详细的相似度分数"""
        texts = [self.prompt + t for t in [anchor, text_a, text_b]]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        
        sim_a = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        sim_b = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        
        return {{
            'similarity_a': float(sim_a),
            'similarity_b': float(sim_b),
            'prediction': 'A' if sim_a > sim_b else 'B',
            'confidence': abs(sim_a - sim_b)
        }}


# 使用示例
if __name__ == "__main__":
    model = NarrativeSimilarityModel()
    
    anchor = "A hero defeats a dragon and saves a princess."
    text_a = "A knight slays a monster and rescues a maiden."
    text_b = "A warrior loses a battle and dies."
    
    prediction = model.predict(anchor, text_a, text_b)
    print(f"预测结果: {{prediction}}")
    
    scores = model.get_similarity_scores(anchor, text_a, text_b)
    print(f"详细分数: {{scores}}")
'''

    output_path = '/mnt/e/Code/python/Narrative-Similarity-Task/output/production_model.py'
    with open(output_path, 'w') as f:
        f.write(code)

    print(f"📝 生产代码已生成: {output_path}")


def main():
    print("🎯 E5-large 终极优化")
    print("=" * 70)

    data_path = '/mnt/e/Code/python/Narrative-Similarity-Task/TrainingSet1/dev_track_a.jsonl'

    # 测试所有配置
    results = test_e5_with_variations(data_path)

    # 验证最佳配置
    best_config = results[0]
    config = final_verification(best_config)

    # 生成生产代码
    create_production_code(config)

    print("\n" + "=" * 70)
    print("🏆 最终结果")
    print("=" * 70)
    print(f"最佳准确率: {config['accuracy_percentage']}")
    print(f"配置: E5-large + '{config['prompt']}' + normalize={config['normalize_embeddings']}")
    print("=" * 70)


if __name__ == "__main__":
    main()