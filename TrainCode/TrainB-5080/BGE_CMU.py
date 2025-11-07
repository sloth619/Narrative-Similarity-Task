"""
Track B训练 - BGE-large-en-v1.5 + Synthetic + CMU Movie (5080优化版)
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses
from datasets import load_dataset, Dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import torch


def build_triplets_from_track_a(data_path, max_length=256):
    """从Track A构建训练数据,限制文本长度"""
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_data = []
    skipped = 0

    def truncate_text(text, max_words=max_length):
        if not text:
            return text
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text

    for item in dataset:
        anchor = item.get('anchor_text') or item.get('anchor_story') or item.get('anchor')
        text_a = item.get('text_a') or item.get('similar_story') or item.get('positive')
        text_b = item.get('text_b') or item.get('dissimilar_story') or item.get('negative')
        label_a_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]):
            skipped += 1
            continue

        # 截断文本
        anchor = truncate_text(anchor)
        text_a = truncate_text(text_a)
        text_b = truncate_text(text_b)

        if label_a_closer is not None:
            positive = text_a if label_a_closer else text_b
        else:
            positive = text_a

        train_data.append({'sentence1': anchor, 'sentence2': positive})
        train_data.append({'sentence1': anchor, 'sentence2': anchor})
        train_data.append({'sentence1': positive, 'sentence2': positive})

    if skipped > 0:
        print(f"  ⚠️ 跳过了 {skipped} 条数据")

    return Dataset.from_list(train_data)


def main():
    print("🚀 Track B训练 - BGE + Synthetic + CMU Movie (5080完整优化版)...")

    # === 5080路径配置 ===
    model_path = r'E:\model\BGE-large-en-v1.5'
    output_path = '../../output/track_b_bge_cmu_full_5080'
    os.makedirs(output_path, exist_ok=True)

    # === 检查断点 ===
    checkpoint_path = None
    if os.path.exists(output_path):
        checkpoints = [d for d in os.listdir(output_path) if d.startswith('checkpoint-')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(output_path, checkpoints[-1])
            print(f"✅ 找到检查点: {checkpoint_path}")

    # === 加载模型 ===
    if checkpoint_path:
        print(f"从检查点加载模型...")
        model = SentenceTransformer(checkpoint_path)
        print("✅ 从检查点加载完成")
    else:
        print(f"加载基础模型: BGE-large-en-v1.5")
        model = SentenceTransformer(model_path)
        print(f"✅ 模型加载完成")
        print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === 加载数据 ===
    print("\n加载训练数据 (256词截断)...")

    print("1. 加载Synthetic数据...")
    synthetic_dataset = build_triplets_from_track_a(
        '../../TrainingSet1/synthetic_data_for_contrastive_learning.jsonl',
        max_length=256
    )
    print(f"   Synthetic: {len(synthetic_dataset)} 个样本")

    print("2. 加载CMU Movie数据...")
    cmu_dataset = build_triplets_from_track_a(
        '../../TrainingSet1/cmu_movie_triplets.jsonl',
        max_length=256
    )
    cmu_dataset = cmu_dataset.select(range(min(30000, len(cmu_dataset))))
    print(f"   CMU Movie: {len(cmu_dataset)} 个样本 (采样25%)")

    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets([synthetic_dataset, cmu_dataset])

    print(f"\n总训练样本: {len(train_dataset):,}")
    print(f"  - Synthetic: {len(synthetic_dataset)} ({len(synthetic_dataset) / len(train_dataset) * 100:.1f}%)")
    print(f"  - CMU Movie: {len(cmu_dataset)} ({len(cmu_dataset) / len(train_dataset) * 100:.1f}%)")

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="bge_cmu_full_5080",
        data_path="../../TrainingSet1/dev_track_a.jsonl",
        batch_size=32
    )

    # === 训练配置 ===
    epochs = 5

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="eval_evaluator",
        bf16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        resume_from_checkpoint=checkpoint_path,
    )

    print(f"\n开始训练 (5080完整优化版):")
    print(f"  - 硬件: RTX 5080 (16GB)")
    if checkpoint_path:
        print(f"  - ✅ 断点续传: {checkpoint_path}")
    else:
        print(f"  - 🆕 从头训练")
    print(f"  - 模型: BGE-large-en-v1.5")
    print(f"  - 文本长度: 256词")
    print(f"  - CMU数据: 30k (25%采样)")
    print(f"  - 总样本: {len(train_dataset):,}")
    print(f"  - Batch size: 8 × 8 = 有效64")
    print(f"  - Epochs: {epochs}")
    print(f"  - 预期准确率: 66-68% 🚀")

    # === 训练 ===
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=mnrl_loss,
        evaluator=evaluator,
    )

    try:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断!")
        print("💾 检查点已保存,可以重新运行继续训练")
        return

    # === 保存 ===
    print("\n保存最终模型...")
    model.save(output_path)
    print(f"✅ 模型已保存到: {output_path}")

    print("✅ 训练完成!")
    print(f"\nBGE + CMU Movie 预期准确率: 66-68%")
    print(f"(vs BGE baseline 64%)")


if __name__ == "__main__":
    main()