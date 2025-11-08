"""
Track B训练 - BGE-M3 Baseline
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses
from datasets import load_dataset, Dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import torch


def build_triplets_from_track_a(data_path):
    """从Track A构建训练数据

    支持两种数据格式:
    1. Track A三元组: anchor + text_a + text_b
    2. Track B单文本: text (构建自对比样本)
    """
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_data = []
    skipped = 0

    for item in dataset:
        # 尝试获取anchor(支持多种字段名)
        anchor = item.get('anchor_text') or item.get('anchor_story') or item.get('anchor') or item.get('text')
        text_a = item.get('text_a') or item.get('similar_story') or item.get('positive')
        text_b = item.get('text_b') or item.get('dissimilar_story') or item.get('negative')
        label_a_closer = item.get('text_a_is_closer')

        # 🔥 处理dev_track_b格式:只有单个文本
        if anchor and not text_a and not text_b:
            # 构建自对比样本(文本与自己配对)
            train_data.append({'sentence1': anchor, 'sentence2': anchor})
            continue

        # 🔥 处理Track A三元组格式
        if not all([anchor, text_a, text_b]):
            skipped += 1
            continue

        # 选择正样本
        if label_a_closer is not None:
            positive = text_a if label_a_closer else text_b
        else:
            positive = text_a

        # 生成训练样本
        train_data.append({'sentence1': anchor, 'sentence2': positive})
        train_data.append({'sentence1': anchor, 'sentence2': anchor})
        train_data.append({'sentence1': positive, 'sentence2': positive})

    if skipped > 0:
        print(f"     ⚠️ 跳过了 {skipped} 条数据")

    return Dataset.from_list(train_data)


def main():
    print("🚀 Track B训练 - BGE-M3 Baseline (修复版)...")

    # === 加载模型 ===
    print("加载模型: BGE-M3")
    model = SentenceTransformer('E:/model/bge-m3')

    print(f"✅ 模型加载完成")
    print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === 加载数据 ===
    print("\n加载训练数据...")

    print("1. 加载Synthetic数据...")
    synthetic_dataset = build_triplets_from_track_a(
        '../../TrainingSet1/synthetic_data_for_contrastive_learning.jsonl'
    )
    print(f"   Synthetic: {len(synthetic_dataset)} 个样本")

    print("2. 加载Dev_b数据...")
    dev_b_dataset = build_triplets_from_track_a(
        '../../TrainingSet1/dev_track_b.jsonl'
    )
    print(f"   Dev_b: {len(dev_b_dataset)} 个样本")

    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets([synthetic_dataset, dev_b_dataset])

    print(f"\n总训练样本: {len(train_dataset):,}")
    print(f"  - Synthetic: {len(synthetic_dataset)} ({len(synthetic_dataset)/len(train_dataset)*100:.1f}%)")
    print(f"  - Dev_b: {len(dev_b_dataset)} ({len(dev_b_dataset)/len(train_dataset)*100:.1f}%)")

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="bge_m3_baseline",
        data_path="../../TrainingSet1/dev_track_a.jsonl",
        batch_size=8
    )

    # === 训练配置 ===
    epochs = 5
    output_path = '../../output/track_b_bge_m3_baseline_5080'
    os.makedirs(output_path, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=20,
        metric_for_best_model="eval_evaluator",
        bf16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    print(f"\n" + "=" * 60)
    print("🚀 开始训练")
    print("=" * 60)
    print(f"配置:")
    print(f"  - 硬件: RTX 5080 (16GB)")
    print(f"  - 模型: BGE-M3 (Multi-lingual, Multi-functionality)")
    print(f"  - 训练数据: Synthetic + Dev_b")
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

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断!")
        print("💾 检查点已保存,可以重新运行脚本继续训练")
        return

    # === 保存 ===
    print("\n保存最终模型...")
    model.save(output_path)
    print(f"✅ 模型已保存到: {output_path}")

    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()