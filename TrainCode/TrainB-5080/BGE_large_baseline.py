"""
Track B训练 - BGE-large-en-v1.5 baseline
固定seed=42，专注训练
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses
from datasets import load_dataset, Dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import torch


def set_seed(seed):
    """固定随机种子"""
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    # 固定种子
    SEED = 42
    set_seed(SEED)

    print(f"🚀 BGE-large-en-v1.5 Full Fine-tuning - Seed {SEED}")

    # 路径配置
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"
    model_name = '/mnt/e/model/BGE-large-en-v1.5'
    output_path = f'{PROJECT_ROOT}/output/bge_full_seed42'

    os.makedirs(output_path, exist_ok=True)

    synthetic_data_path = f'{PROJECT_ROOT}/TrainingSet1/synthetic_data_for_contrastive_learning.jsonl'
    dev_track_a_path = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'

    # 加载模型
    print(f"\n加载模型...")
    model = SentenceTransformer(model_name)

    # 加载数据
    print("\n加载训练数据...")
    train_dataset = build_triplets_from_track_a(synthetic_data_path)
    print(f"训练样本: {len(train_dataset):,}")

    # 损失函数和评估器
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="bge_full_seed42",
        data_path=dev_track_a_path,
        batch_size=8
    )

    # 训练配置
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="eval_evaluator",
        greater_is_better=True,
        bf16=True,
        seed=SEED,
    )

    # 训练
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=mnrl_loss,
        evaluator=evaluator,
    )

    print("\n开始训练...\n")
    trainer.train()

    # 保存
    print("\n保存模型...")
    model.save(output_path)
    print("✅ 完成!")


if __name__ == "__main__":
    main()