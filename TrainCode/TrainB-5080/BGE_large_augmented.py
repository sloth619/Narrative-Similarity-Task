"""
Track B训练 - BGE-large-en-v1.5 (使用优化后的数据)
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses
from datasets import load_dataset, Dataset, concatenate_datasets
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import torch


def build_triplets_from_track_a(data_path):
    """
    从Track A数据构建训练样本
    使用对比学习的三元组格式
    """
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_data = []
    for item in dataset:
        anchor = item.get('anchor_text')
        text_a = item.get('text_a')
        text_b = item.get('text_b')
        label_a_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]):
            continue

        # 确定正样本
        if label_a_closer is not None:
            positive = text_a if label_a_closer else text_b
            negative = text_b if label_a_closer else text_a
        else:
            positive = text_a
            negative = text_b

        # 构建对比学习样本对
        train_data.append({'sentence1': anchor, 'sentence2': positive})
        train_data.append({'sentence1': anchor, 'sentence2': anchor})
        train_data.append({'sentence1': positive, 'sentence2': positive})

    return Dataset.from_list(train_data)


def main():
    print("🚀 Track B训练 - BGE (使用优化后的数据)")
    print("目标: 0.66 → 0.67-0.68\n")
    print("="*60)

    # === 路径配置 ===
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    # 模型路径
    MODEL_NAME = '/mnt/e/model/BGE-large-en-v1.5'

    # 输出路径 (新的实验名称)
    OUTPUT_PATH = f'{PROJECT_ROOT}/output/track_b_bge_optimized_5080'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 数据集路径
    DEV_TRACK_A = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'
    DEV_TRACK_B = f'{PROJECT_ROOT}/TrainingSet1/dev_track_b.jsonl'

    # 🌟 关键: 使用优化后的数据!
    OPTIMIZED_TRAIN_DATA = f'{PROJECT_ROOT}/TrainingSet_optimized/augmented_training_data.jsonl'

    # === 加载模型 ===
    print(f"📦 加载模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"✅ 模型加载完成")
    print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === 加载数据 ===
    print(f"\n📂 加载训练数据...")

    # 1. 优化后的训练数据 (原始1900 + 困难样本150)
    print(f"1. 加载优化后的训练数据...")
    optimized_dataset = build_triplets_from_track_a(OPTIMIZED_TRAIN_DATA)
    print(f"   优化数据: {len(optimized_dataset)} 个样本")

    # 2. Dev_b数据 (可选,作为额外训练数据)
    print(f"2. 加载Dev_b数据...")
    dev_b_dataset = build_triplets_from_track_a(DEV_TRACK_B)
    print(f"   Dev_b: {len(dev_b_dataset)} 个样本")

    # 3. 组合数据
    train_dataset = concatenate_datasets([optimized_dataset, dev_b_dataset])
    print(f"\n✅ 总训练样本: {len(train_dataset):,}")

    # === 损失函数 ===
    print(f"\n⚙️  配置训练...")
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)
    print(f"   损失函数: MultipleNegativesRankingLoss")

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="bge_optimized",
        data_path=DEV_TRACK_A,
        batch_size=8
    )
    print(f"   评估器: TrackB_Accuracy_Evaluator")

    # === 训练配置 ===
    EPOCHS = 5
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5

    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="eval_evaluator",
        bf16=True,
        report_to="none",
    )

    print(f"\n📊 训练配置:")
    print(f"   模型: BGE-large-en-v1.5")
    print(f"   数据: 优化后数据 (含困难样本增强)")
    print(f"   总样本: {len(train_dataset):,}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   预期提升: 0.66 → 0.67-0.68")

    # === 创建Trainer ===
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=mnrl_loss,
        evaluator=evaluator,
    )

    # === 开始训练 ===
    print(f"\n{'='*60}")
    print(f"🎯 开始训练...")
    print(f"{'='*60}\n")

    trainer.train()

    # === 保存最终模型 ===
    print(f"\n💾 保存最终模型...")
    final_model_path = os.path.join(OUTPUT_PATH, 'final_model')
    model.save(final_model_path)
    print(f"✅ 模型已保存到: {final_model_path}")

    # === 训练完成总结 ===
    print(f"\n{'='*60}")
    print(f"✅ 训练完成!")
    print(f"{'='*60}")
    print(f"\n📊 实验对比:")
    print(f"   Baseline (原始数据): 0.66")
    print(f"   Optimized (本次): 待测试")
    print(f"   预期提升: +0.01-0.02")
    print(f"\n💡 下一步:")
    print(f"   1. 在dev set上测试新模型")
    print(f"   2. 如果达到0.67-0.68,生成提交文件")
    print(f"   3. 提交到CodaLab验证")
    print(f"\n📁 模型位置:")
    print(f"   最佳checkpoint: {OUTPUT_PATH}")
    print(f"   最终模型: {final_model_path}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()