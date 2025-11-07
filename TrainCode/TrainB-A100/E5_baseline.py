"""
Track B训练 - E5-large-v2 (A100 40GB优化版)
只用Synthetic高质量数据
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
    """从Track A构建训练数据 (带详细调试)"""
    print(f"  正在加载: {data_path}")

    # 检查文件
    if not os.path.exists(data_path):
        print(f"  ⚠️ 文件不存在: {data_path}")
        return Dataset.from_list([])

    dataset = load_dataset('json', data_files=data_path, split='train')
    print(f"  原始数据行数: {len(dataset)}")

    # 看第一条数据的keys
    if len(dataset) > 0:
        print(f"  数据字段: {list(dataset[0].keys())}")

    train_data = []
    skipped = 0

    for item in dataset:
        # 尝试多种字段名
        anchor = item.get('anchor_text') or item.get('anchor_story') or item.get('anchor')
        text_a = item.get('text_a') or item.get('similar_story') or item.get('positive')
        text_b = item.get('text_b') or item.get('dissimilar_story') or item.get('negative')
        label_a_closer = item.get('text_a_is_closer')

        # 如果三个都没有,跳过
        if not all([anchor, text_a, text_b]):
            skipped += 1
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

    if skipped > 0:
        print(f"  ⚠️ 跳过了 {skipped} 条数据 (缺少字段)")

    print(f"  ✅ 生成了 {len(train_data)} 个训练样本")
    return Dataset.from_list(train_data)


def main():
    print("🚀 Track B训练 - E5-large-v2 (A100 40GB优化版)...")

    # === A100路径配置 ===
    model_path = '/root/autodl-tmp/Narrative-Similarity-Task/models/e5-large-v2'
    output_path = '/root/autodl-tmp/Narrative-Similarity-Task/output/track_b_e5_a100'
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
        print("✅ 模型从检查点加载完成")
    else:
        print("加载模型: intfloat/e5-large-v2")
        try:
            model = SentenceTransformer(model_path)
            print("✅ 本地模型加载成功")
        except Exception as e:
            print(f"本地加载失败: {e}")
            print("从HuggingFace下载...")
            model = SentenceTransformer('intfloat/e5-large-v2')
            print("✅ HF模型加载成功")

        print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === 加载数据 ===
    print("\n加载训练数据...")

    print("1. 加载Synthetic数据...")
    train_dataset = build_triplets_from_track_a(
        '/root/autodl-tmp/Narrative-Similarity-Task/TrainingSet1/synthetic_data_for_contrastive_learning.jsonl',
        add_prefix=True  # E5需要前缀
    )

    print(f"\n总训练样本: {len(train_dataset):,}")

    if len(train_dataset) == 0:
        print("❌ 没有训练数据!请检查文件路径和格式")
        return

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="e5_a100",
        data_path="/root/autodl-tmp/Narrative-Similarity-Task/TrainingSet1/dev_track_a.jsonl",
        batch_size=64  # A100可以用更大的评估batch
    )

    # === 训练配置 (A100优化) ===
    epochs = 5

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=64,  # A100大batch
        gradient_accumulation_steps=1,
        learning_rate=2e-5,  # E5推荐2e-5
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_steps=20,
        metric_for_best_model="eval_evaluator",
        bf16=True,
        dataloader_num_workers=4,  # A100多进程
        dataloader_pin_memory=True,
        resume_from_checkpoint=checkpoint_path,
    )

    print(f"\n开始训练:")
    print(f"  - 硬件: A100 40GB")
    if checkpoint_path:
        print(f"  - ✅ 断点续传: {checkpoint_path}")
    else:
        print(f"  - 🆕 从头训练")
    print(f"  - 模型: E5-large-v2")
    print(f"  - 训练数据: Synthetic only (高质量)")
    print(f"  - E5前缀: query: / passage:")
    print(f"  - 总样本: {len(train_dataset):,}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Epochs: {epochs}")
    print(f"  - 预计步数: {len(train_dataset) // training_args.per_device_train_batch_size * epochs}")
    print(f"  - 预计时间: ~6-8分钟")

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
    print(f"\n预期准确率: 64-66%")
    print(f"(E5 + Synthetic + A100大batch)")


if __name__ == "__main__":
    main()