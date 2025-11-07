"""
Track B训练 - E5-large-v2 (5080)
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses
from datasets import load_dataset, Dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import torch


def build_triplets_from_track_a(data_path, add_prefix=True):
    """从Track A构建训练数据"""
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_data = []
    for item in dataset:
        anchor = item.get('anchor_text') or item.get('anchor_story')
        text_a = item.get('text_a') or item.get('similar_story')
        text_b = item.get('text_b') or item.get('dissimilar_story')
        label_a_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]):
            continue

        if label_a_closer is not None:
            positive = text_a if label_a_closer else text_b
        else:
            positive = text_a

        # E5需要加前缀
        if add_prefix:
            anchor = f"query: {anchor}"
            positive = f"passage: {positive}"

        train_data.append({'sentence1': anchor, 'sentence2': positive})
        train_data.append({'sentence1': anchor, 'sentence2': anchor})
        train_data.append({'sentence1': positive, 'sentence2': positive})

    return Dataset.from_list(train_data)


def main():
    print("🚀 Track B训练 - E5-large-v2 (5080)...")

    # === 加载模型 ===
    print("加载模型: intfloat/e5-large-v2")
    model_path = r'E:\model\e5-large-v2'  # 本地路径
    # 或者直接从HF下载:
    # model_path = 'intfloat/e5-large-v2'

    try:
        model = SentenceTransformer(model_path)
        print("✅ 本地模型加载成功")
    except:
        print("本地加载失败,从HuggingFace下载...")
        model = SentenceTransformer('intfloat/e5-large-v2')
        print("✅ HF模型加载成功")

    print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === 加载数据 ===
    print("\n加载训练数据...")

    print("1. 加载Synthetic数据...")
    train_dataset = build_triplets_from_track_a(
        '../../TrainingSet1/synthetic_data_for_contrastive_learning.jsonl',
        add_prefix=True  # E5需要前缀
    )

    print(f"\n总训练样本: {len(train_dataset):,}")

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="e5_baseline",
        data_path="../../TrainingSet1/dev_track_a.jsonl",
        batch_size=32
    )

    # === 训练配置 ===
    epochs = 5
    output_path = '../../output/track_b_e5_baseline_5080'
    os.makedirs(output_path, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,  # E5推荐2e-5
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=20,
        metric_for_best_model="eval_evaluator",
        bf16=True,
    )

    print(f"\n开始训练:")
    print(f"  - 模型: E5-large-v2")
    print(f"  - 训练数据: Synthetic(带前缀)")
    print(f"  - 总样本: {len(train_dataset):,}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Epochs: {epochs}")

    # === 训练 ===
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=mnrl_loss,
        evaluator=evaluator,
    )

    trainer.train()

    # === 保存 ===
    print("\n保存最终模型...")
    model.save(output_path)
    print(f"✅ 模型已保存到: {output_path}")

    print("✅ 训练完成!")
    print(f"\nE5 baseline预期准确率: 63-65%")


if __name__ == "__main__":
    main()