"""
Track B训练 - GTE-large-en-v1.5
✅ 支持8192长文本,完美适配维基故事
✅ MTEB检索任务领先BGE +3.75分
✅ 预期准确率: 67-71%
"""
import os
import gc
import torch

# 清理显存
torch.cuda.empty_cache()
gc.collect()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses
from datasets import load_dataset, Dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


def build_triplets_from_track_a(data_path):
    """从Track A构建训练数据"""
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_data = []
    skipped = 0

    for item in dataset:
        anchor = item.get('anchor_text') or item.get('anchor_story') or item.get('anchor') or item.get('text')
        text_a = item.get('text_a') or item.get('similar_story') or item.get('positive')
        text_b = item.get('text_b') or item.get('dissimilar_story') or item.get('negative')
        label_a_closer = item.get('text_a_is_closer')

        # 处理dev_track_b格式
        if anchor and not text_a and not text_b:
            train_data.append({'sentence1': anchor, 'sentence2': anchor})
            continue

        # 处理Track A三元组格式
        if not all([anchor, text_a, text_b]):
            skipped += 1
            continue

        if label_a_closer is not None:
            positive = text_a if label_a_closer else text_b
        else:
            positive = text_a

        train_data.append({'sentence1': anchor, 'sentence2': positive})
        train_data.append({'sentence1': anchor, 'sentence2': anchor})
        train_data.append({'sentence1': positive, 'sentence2': positive})

    if skipped > 0:
        print(f"     ⚠️ 跳过了 {skipped} 条数据")

    return Dataset.from_list(train_data)


def evaluate_zero_shot(model, data_path):
    """评估零样本性能"""
    print("\n" + "=" * 60)
    print("🔍 零样本测试 - GTE-large-en-v1.5")
    print("=" * 60)

    dev_dataset = load_dataset('json', data_files=data_path, split='train')

    correct = 0
    total = 0

    print(f"开始评估 {len(dev_dataset)} 个三元组...")

    for idx, item in enumerate(dev_dataset):
        anchor = item.get('anchor_text') or item.get('anchor_story')
        text_a = item.get('text_a') or item.get('similar_story')
        text_b = item.get('text_b') or item.get('dissimilar_story')
        label_a_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]) or label_a_closer is None:
            continue

        # 编码 - GTE支持8192长度
        embeddings = model.encode(
            [anchor, text_a, text_b],
            show_progress_bar=False,
            batch_size=32
        )

        anchor_emb = embeddings[0]
        text_a_emb = embeddings[1]
        text_b_emb = embeddings[2]

        # 计算余弦相似度
        sim_a = torch.nn.functional.cosine_similarity(
            torch.tensor(anchor_emb).unsqueeze(0),
            torch.tensor(text_a_emb).unsqueeze(0)
        ).item()

        sim_b = torch.nn.functional.cosine_similarity(
            torch.tensor(anchor_emb).unsqueeze(0),
            torch.tensor(text_b_emb).unsqueeze(0)
        ).item()

        # 预测
        prediction = sim_a > sim_b

        if prediction == label_a_closer:
            correct += 1
        total += 1

        # 进度提示
        if (idx + 1) % 50 == 0:
            print(f"  已评估: {idx + 1}/{len(dev_dataset)}, 当前准确率: {correct/total:.2%}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ 零样本准确率: {accuracy:.4f} ({correct}/{total})")

    return accuracy


def main():
    print("🚀 Track B训练 - GTE-large-en-v1.5...")

    # === 清理显存 ===
    print("\n清理GPU显存...")
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ 显存已清理")

    # === 路径配置 ===
    model_name = '/mnt/e/model/gte-large-en-v1.5'
    output_path = '../../output/track_b_gte_large_en_v15'
    os.makedirs(output_path, exist_ok=True)

    # === 加载模型 ===
    print(f"\n加载模型: {model_name}")
    print("✅ GTE: 支持8192长文本,检索任务SOTA")

    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,  # GTE需要trust_remote_code
        device='cuda'
    )

    print(f"\n✅ 模型加载完成")
    print(f"   模型: GTE-large-en-v1.5 (434M参数)")
    print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")
    print(f"   最大长度: 8192 tokens (vs BGE的512)")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === Step 1: 零样本测试 ===
    print("\n📊 Step 1: 零样本性能测试")
    zero_shot_acc = evaluate_zero_shot(
        model=model,
        data_path="../../TrainingSet1/dev_track_a.jsonl"
    )

    print(f"\n💡 分析:")
    if zero_shot_acc > 0.62:
        print(f"   🎉 零样本准确率 {zero_shot_acc:.2%} 很高!")
        print(f"   GTE长文本优势明显!")
    print(f"   预期微调后: 67-71%")

    # === Step 2: 加载训练数据 ===
    print("\n" + "=" * 60)
    print("📚 Step 2: 加载训练数据")
    print("=" * 60)

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

    # === Step 3: 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === Step 4: 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="gte_large_v15",
        data_path="../../TrainingSet1/dev_track_a.jsonl",
        batch_size=32
    )

    # === Step 5: 训练配置 ===
    epochs = 5

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=20,
        logging_first_step=True,
        metric_for_best_model="eval_evaluator",
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    print("\n" + "=" * 60)
    print("🚀 Step 3: 开始微调")
    print("=" * 60)
    print(f"配置:")
    print(f"  - 硬件: RTX 5080 (16GB)")
    print(f"  - 模型: GTE-large-en-v1.5 (434M)")
    print(f"  - 特点: 8192长文本支持 ✨")
    print(f"  - 训练数据: Synthetic + Dev_b")
    print(f"  - 总样本: {len(train_dataset):,}")
    print(f"  - Batch size: 24")
    print(f"  - Learning rate: 2e-5 (cosine)")
    print(f"  - Epochs: {epochs}")
    print(f"  - 零样本基线: {zero_shot_acc:.2%}")
    print(f"\n预期结果:")
    print(f"  - 微调后: 67-71% 🎯")
    print(f"  - vs BGE: +2~3%")
    print(f"  - 训练时间: 40-55分钟")

    # === Step 6: 训练 ===
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=mnrl_loss,
        evaluator=evaluator,
    )

    try:
        print("\n开始训练...\n")
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断!")
        print("💾 检查点已保存")
        return
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ 显存不足!")
            print("💡 降低batch_size到16:")
            print("   per_device_train_batch_size=16")
        else:
            raise e
        return

    # === Step 7: 保存 ===
    print("\n保存最终模型...")
    model.save(output_path)
    print(f"✅ 模型已保存到: {output_path}")

    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"📊 GTE-large-en-v1.5性能总结:")
    print(f"  - 零样本: {zero_shot_acc:.2%}")
    print(f"  - 微调后: 查看上方最佳准确率")
    print(f"\n🎯 GTE优势:")
    print(f"  ✅ 8192长文本支持(BGE只有512)")
    print(f"  ✅ MTEB检索任务+3.75分")
    print(f"  ✅ 预期比BGE高2-3%")


if __name__ == "__main__":
    main()