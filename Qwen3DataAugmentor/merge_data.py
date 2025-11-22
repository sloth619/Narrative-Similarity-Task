"""
合并原始数据 + Hard Negatives
"""
import json


def merge_training_data():
    """合并训练数据"""

    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    # 原始数据
    original_path = f"{PROJECT_ROOT}/TrainingSet1/synthetic_data_for_contrastive_learning.jsonl"

    # Hard Negatives
    hard_neg_path = f"{PROJECT_ROOT}/GeminiData/HardNegatives/hard_negatives_validated.jsonl"

    # 输出
    output_path = f"{PROJECT_ROOT}/TrainingSet1/merged_training_data.jsonl"

    all_samples = []

    # 加载原始数据
    print("加载原始数据...")
    with open(original_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # 转换字段名
                if 'anchor_story' in sample:
                    all_samples.append({
                        'anchor_text': sample['anchor_story'],
                        'text_a': sample['similar_story'],
                        'text_b': sample['dissimilar_story'],
                        'text_a_is_closer': True,
                        'source': 'original'
                    })

    print(f"  原始: {len(all_samples)} 样本")

    # 加载Hard Negatives
    print("加载Hard Negatives...")
    with open(hard_neg_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                sample['source'] = 'hard_negative'
                all_samples.append(sample)

    print(f"  Hard Negatives: {len(all_samples) - 1900} 样本")

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\n✅ 合并完成!")
    print(f"   总样本: {len(all_samples)}")
    print(f"   输出: {output_path}")

    # 统计
    from collections import defaultdict
    source_stats = defaultdict(int)
    for s in all_samples:
        source_stats[s.get('source', 'unknown')] += 1

    print(f"\n数据来源:")
    for source, count in source_stats.items():
        print(f"   {source:20s}: {count}")


if __name__ == "__main__":
    merge_training_data()