"""
Track A训练 - E5-large-v2
3090 24GB + 6300 Gemini + 1900 Synthetic
只保存acc > 68%的模型
"""
import os
import gc
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.evaluation import SimilarityFunction
from datasets import load_dataset, Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Dict


# ===== 自定义评估器 =====
class TrackAEvaluator:
    """Track A评估器 - 返回metrics字典"""

    def __init__(self, dev_data_path: str, name: str = "track_a", threshold: float = 0.69):
        self.dev_data_path = dev_data_path
        self.name = name
        self.threshold = threshold
        self.best_acc = 0.0

        # 加载验证集
        self.dev_dataset = load_dataset('json', data_files=dev_data_path, split='train')
        print(f"✅ 加载验证集: {len(self.dev_dataset)} 样本")

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        """评估模型 - 返回metrics字典"""

        predictions = []
        labels = []

        model.eval()
        device = next(model.parameters()).device

        print(f"\n{'='*60}")
        print(f"📊 开始评估 (Epoch {epoch})...")

        with torch.no_grad():
            for item in self.dev_dataset:
                anchor = item.get('anchor_text') or item.get('anchor_story')
                text_a = item.get('text_a') or item.get('similar_story')
                text_b = item.get('text_b') or item.get('dissimilar_story')
                label = item.get('text_a_is_closer')

                if not all([anchor, text_a, text_b]) or label is None:
                    continue

                # E5需要加前缀
                anchor = f"query: {anchor}"
                text_a = f"passage: {text_a}"
                text_b = f"passage: {text_b}"

                # 编码
                embeddings = model.encode(
                    [anchor, text_a, text_b],
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=32,
                    device=device
                )

                # 计算相似度
                sim_a = torch.nn.functional.cosine_similarity(
                    embeddings[0].unsqueeze(0),
                    embeddings[1].unsqueeze(0)
                ).item()

                sim_b = torch.nn.functional.cosine_similarity(
                    embeddings[0].unsqueeze(0),
                    embeddings[2].unsqueeze(0)
                ).item()

                prediction = sim_a > sim_b
                predictions.append(prediction)
                labels.append(label)

        # 计算准确率
        accuracy = accuracy_score(labels, predictions)

        # 更新最佳
        if accuracy > self.best_acc:
            self.best_acc = accuracy

        # 打印结果
        print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   阈值: {self.threshold:.4f} ({self.threshold*100:.2f}%)")

        if accuracy > self.threshold:
            print(f"   ✅ 超过阈值!")
        else:
            print(f"   ❌ 未达阈值")

        print(f"   历史最佳: {self.best_acc:.4f} ({self.best_acc*100:.2f}%)")
        print(f"{'='*60}\n")

        model.train()

        # 返回metrics字典 (Trainer需要)
        return {
            f"{self.name}_accuracy": accuracy,
            f"{self.name}_best_accuracy": self.best_acc
        }


# ===== 数据加载函数 =====
def load_training_data(data_path: str, add_prefix: bool = True):
    """加载训练数据并构建三元组"""

    dataset = load_dataset('json', data_files=data_path, split='train')

    examples = []

    for item in dataset:
        anchor = item.get('anchor_text') or item.get('anchor_story')
        text_a = item.get('text_a') or item.get('similar_story')
        text_b = item.get('text_b') or item.get('dissimilar_story')
        label = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]):
            continue

        # 确定正负样本
        if label is not None:
            positive = text_a if label else text_b
            negative = text_b if label else text_a
        else:
            positive = text_a
            negative = text_b

        # E5需要加前缀
        if add_prefix:
            anchor = f"query: {anchor}"
            positive = f"passage: {positive}"
            negative = f"passage: {negative}"

        # 构建InputExample (anchor, positive, negative)
        examples.append(InputExample(texts=[anchor, positive, negative]))

    return examples


# ===== 主训练函数 =====
def main():
    print("="*60)
    print("🚀 Track A训练 - E5-large-v2 (Trainer API)")
    print("="*60)

    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()

    # 确认GPU
    print(f"\n🔍 GPU状态:")
    print(f"   可用GPU数: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"   当前GPU: cuda:{torch.cuda.current_device()}")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")

    # ===== 路径配置 =====
    PROJECT_ROOT = "/home/songfeiyang/workspace/semeval"
    MODEL_PATH = "/home/songfeiyang/workspace/model/e5-large-v2"

    SYNTHETIC_DATA = f"{PROJECT_ROOT}/TrainSet/synthetic_data_for_contrastive_learning.jsonl"
    GEMINI_DATA = f"{PROJECT_ROOT}/TrainSet/gemini_generated_10k.jsonl"
    DEV_DATA = f"{PROJECT_ROOT}/TrainSet/dev_track_a.jsonl"
    OUTPUT_PATH = f"{PROJECT_ROOT}/output/track_a_e5_gemini_6k_v2"

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ===== 加载模型 =====
    print("\n📦 加载模型...")
    model = SentenceTransformer(MODEL_PATH, device='cuda')
    print(f"   ✅ 模型加载成功")
    print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # ===== 加载训练数据 =====
    print("\n📂 加载训练数据...")

    print("   1. 加载原始Synthetic数据...")
    synthetic_examples = load_training_data(SYNTHETIC_DATA, add_prefix=True)
    print(f"      ✅ {len(synthetic_examples):,} 个三元组")

    print("   2. 加载Gemini生成数据...")
    gemini_examples = load_training_data(GEMINI_DATA, add_prefix=True)
    print(f"      ✅ {len(gemini_examples):,} 个三元组")

    # 合并数据
    all_examples = synthetic_examples + gemini_examples
    print(f"\n   📊 总训练样本: {len(all_examples):,} 个三元组")

    # 转换为Dataset格式
    train_dataset = Dataset.from_dict({
        'anchor': [ex.texts[0] for ex in all_examples],
        'positive': [ex.texts[1] for ex in all_examples],
        'negative': [ex.texts[2] for ex in all_examples]
    })

    # ===== 损失函数 =====
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    print(f"\n   损失函数: MultipleNegativesRankingLoss")

    # ===== 评估器 =====
    print("\n📊 配置评估器...")
    evaluator = TrackAEvaluator(
        dev_data_path=DEV_DATA,
        name="track_a",
        threshold=0.69
    )

    # ===== 训练配置 =====
    target_lr = 3e-7
    target_warmup = 0.1
    epochs = 5
    batch_size = 16
    steps_per_epoch = len(all_examples) // batch_size

    print(f"\n⚙️  训练配置:")
    print(f"   模型: E5-large-v2")
    print(f"   训练集: Synthetic(1900) + Gemini(10000)")
    print(f"   总样本: {len(all_examples):,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {target_lr}")
    print(f"   Warmup ratio: {target_warmup}")
    print(f"   Epochs: {epochs}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   保存阈值: 69%")
    print(f"   输出路径: {OUTPUT_PATH}")

    # ===== Trainer参数 =====
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        learning_rate=target_lr,
        warmup_ratio=target_warmup,
        fp16=False,
        bf16=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="track_a_accuracy",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    # ===== 创建Trainer =====
    print(f"\n{'='*60}")
    print("🎯 开始训练...")
    print(f"{'='*60}\n")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=[evaluator],
    )

    # ===== 训练 =====
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  训练被中断")

    # ===== 最终评估 =====
    print(f"\n{'='*60}")
    print("✅ 训练完成!")
    print(f"{'='*60}")

    print("\n🔍 最终评估...")
    final_metrics = evaluator(model, OUTPUT_PATH, epoch=-1)
    final_acc = final_metrics['track_a_accuracy']

    print(f"\n📊 最终结果:")
    print(f"   最终准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"   历史最佳: {evaluator.best_acc:.4f} ({evaluator.best_acc*100:.2f}%)")

    # 只在超过阈值时保存
    if final_acc > 0.69:
        print(f"\n💾 保存最终模型...")
        final_model_path = f"{OUTPUT_PATH}/final_model"
        model.save(final_model_path)
        print(f"   ✅ 已保存到: {final_model_path}")
    else:
        print(f"\n⚠️  最终模型未达阈值,不保存")

    # ===== 保存训练日志 =====
    log_file = f"{OUTPUT_PATH}/training_summary.txt"
    with open(log_file, 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"="*60 + "\n")
        f.write(f"Model: E5-large-v2\n")
        f.write(f"Training samples: {len(all_examples):,}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {target_lr}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"Final accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)\n")
        f.write(f"Best accuracy: {evaluator.best_acc:.4f} ({evaluator.best_acc*100:.2f}%)\n")
        f.write(f"Threshold: 69%\n")
        f.write(f"Model saved: {'Yes' if final_acc > 0.69 else 'No'}\n")

    print(f"\n📝 训练摘要已保存到: {log_file}")

    print(f"\n{'='*60}")
    print("🎉 所有任务完成!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()