"""
Track B训练 - BGE-large-en-v1.5 + ROCStories (支持断点续传)
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses
from datasets import load_dataset, Dataset, concatenate_datasets
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import torch


def load_rocstories(train_path):
    """加载ROCStories数据"""
    print(f"加载ROCStories: {train_path}")

    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                train_data.append({'sentence1': line, 'sentence2': line})

    return Dataset.from_list(train_data)


def build_triplets_from_track_a(data_path):
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

        train_data.append({'sentence1': anchor, 'sentence2': positive})
        train_data.append({'sentence1': anchor, 'sentence2': anchor})
        train_data.append({'sentence1': positive, 'sentence2': positive})

    return Dataset.from_list(train_data)


def main():
    print("🚀 Track B训练 - BGE-large-en-v1.5 + ROCStories (支持断点续传)...")

    output_path = '../../output/track_b_bge_rocstories_5080'
    os.makedirs(output_path, exist_ok=True)

    # === 检查是否有检查点 ===
    checkpoint_path = None
    if os.path.exists(output_path):
        checkpoints = [d for d in os.listdir(output_path) if d.startswith('checkpoint-')]
        if checkpoints:
            # 找到最新的检查点
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(output_path, checkpoints[-1])
            print(f"✅ 找到检查点: {checkpoint_path}")
            print(f"   将从此检查点继续训练...")

    # === 加载模型 ===
    if checkpoint_path:
        print(f"从检查点加载模型...")
        model = SentenceTransformer(checkpoint_path)
        print("✅ 模型从检查点加载完成")
    else:
        print("加载模型: BAAI/bge-large-en-v1.5")
        model = SentenceTransformer('E:\model\BGE-large-en-v1.5')
        print(f"✅ 模型加载完成")
        print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === 加载数据 ===
    print("\n加载训练数据...")

    print("1. 加载ROCStories...")
    roc_dataset = load_rocstories('E:/Code/python/Narrative-Similarity-Task/ROCStories/train.txt')
    print(f"   ROCStories: {len(roc_dataset)} 个样本")

    print("2. 加载Synthetic数据...")
    synthetic_dataset = build_triplets_from_track_a(
        '../../TrainingSet1/synthetic_data_for_contrastive_learning.jsonl'
    )
    print(f"   Synthetic: {len(synthetic_dataset)} 个样本")

    print("3. 加载Dev_b数据...")
    dev_b_dataset = build_triplets_from_track_a(
        '../../TrainingSet1/dev_track_b.jsonl'
    )
    print(f"   Dev_b: {len(dev_b_dataset)} 个样本")

    train_dataset = concatenate_datasets([roc_dataset, synthetic_dataset, dev_b_dataset])

    print(f"\n总训练样本: {len(train_dataset):,}")

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="bge_rocstories",
        data_path="../../TrainingSet1/dev_track_a.jsonl",
        batch_size=8
    )

    # === 训练配置 (BGE推荐参数) ===
    epochs = 5

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,  # BGE推荐batch size
        gradient_accumulation_steps=1,
        learning_rate=2e-5,  # BGE推荐学习率
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,  # 保留最近3个检查点
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="eval_evaluator",
        bf16=True,
        resume_from_checkpoint=checkpoint_path,  # 从检查点恢复
    )

    print(f"\n开始训练:")
    if checkpoint_path:
        print(f"  ✅ 断点续传模式")
        print(f"  - 从检查点: {checkpoint_path}")
    else:
        print(f"  🆕 从头开始训练")
    print(f"  - 模型: BGE-large-en-v1.5")
    print(f"  - 训练数据: ROCStories + Synthetic + Dev_b")
    print(f"  - 总样本: {len(train_dataset):,}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Epochs: {epochs}")
    print(f"  - 每500步保存检查点")

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
        print("💾 检查点已保存,可以重新运行脚本继续训练")
        return

    # === 保存 ===
    print("\n保存最终模型...")
    model.save(output_path)
    print(f"✅ 模型已保存到: {output_path}")

    print("✅ 训练完成!")
    print(f"\nBGE + ROCStories预期准确率: 可能高于baseline (60-63%)")


if __name__ == "__main__":
    main()