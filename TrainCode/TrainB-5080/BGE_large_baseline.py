"""
Track B训练 - BGE-large-en-v1.5 baseline (5080)
使用官方Synthetic数据测试
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
    """从Track A构建训练数据 (Baseline 原始逻辑)"""
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_data = []
    for item in dataset:
        anchor = item.get('anchor_text') or item.get('anchor_story')
        text_a = item.get('text_a') or item.get('similar_story')
        text_b = item.get('text_b') or item.get('dissimilar_story')
        label_a_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]):
            # 注意: Baseline 逻辑跳过了 dev_track_b 的数据
            continue

        if label_a_closer is not None:
            positive = text_a if label_a_closer else text_b
        else:
            positive = text_a

        # --- Baseline 逻辑 ---
        # 保留了 (anchor, positive)
        train_data.append({'sentence1': anchor, 'sentence2': positive})
        # [BUG] 保留了 (anchor, anchor)
        train_data.append({'sentence1': anchor, 'sentence2': anchor})
        # [BUG] 保留了 (positive, positive)
        train_data.append({'sentence1': positive, 'sentence2': positive})
        # ---------------------

    return Dataset.from_list(train_data)


def main():
    print("🚀 Track B训练 - BGE-large-en-v1.5 Baseline (5080)...")

    # === 路径配置 (已修改为WSL绝对路径) ===
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    # 模型路径
    model_name = '/mnt/e/model/BGE-large-en-v1.5'

    # 输出路径
    output_path = f'{PROJECT_ROOT}/output/track_b_bge_baseline_5080_wsl'
    os.makedirs(output_path, exist_ok=True)

    # 数据集路径
    dev_track_a_path = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'
    synthetic_data_path = f'{PROJECT_ROOT}/TrainingSet1/synthetic_data_for_contrastive_learning.jsonl'
    dev_track_b_path = f'{PROJECT_ROOT}/TrainingSet1/dev_track_b.jsonl'

    # === 加载模型  ===
    print(f"加载模型: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"✅ 模型加载完成")
    print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === 加载数据 ===
    print("\n加载训练数据...")

    print("1. 加载Synthetic数据...")
    synthetic_dataset = build_triplets_from_track_a(
        synthetic_data_path # <-- 使用WSL路径
    )
    print(f"   Synthetic: {len(synthetic_dataset)} 个样本")

    print("2. 加载Dev_b数据...")
    dev_b_dataset = build_triplets_from_track_a(
        dev_track_b_path # <-- 使用WSL路径
    )
    print(f"   Dev_b: {len(dev_b_dataset)} 个样本")

    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets([synthetic_dataset, dev_b_dataset])

    print(f"\n总训练样本: {len(train_dataset):,}")

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="bge_baseline",
        data_path=dev_track_a_path, # <-- 使用WSL路径
        batch_size=8
    )

    # === 训练配置 (BGE推荐参数) ===
    epochs = 5

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
    )

    print(f"\n开始训练:")
    print(f"  - 模型: BGE-large-en-v1.5")
    print(f"  - 训练数据: Synthetic + Dev_b (Baseline-Bug-Logic)")
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
    print(f"\nBGE baseline预期准确率: 60-63%")


if __name__ == "__main__":
    main()