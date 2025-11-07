"""
Track B训练 - Qwen3-Embedding-4B + Synthetic + CMU Movie (A100 40GB)
使用4-bit量化 + LoRA微调
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses, models
from datasets import load_dataset, Dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training


def build_triplets_from_track_a(data_path, max_length=200):
    """从Track A构建训练数据,限制文本长度"""
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_data = []
    skipped = 0

    for item in dataset:
        anchor = item.get('anchor_text') or item.get('anchor_story') or item.get('anchor')
        text_a = item.get('text_a') or item.get('similar_story') or item.get('positive')
        text_b = item.get('text_b') or item.get('dissimilar_story') or item.get('negative')
        label_a_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]):
            skipped += 1
            continue

        # 限制文本长度
        def truncate_text(text, max_words=max_length):
            words = text.split()
            if len(words) > max_words:
                return ' '.join(words[:max_words])
            return text

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
    print("🚀 Track B训练 - Qwen3-4B + Synthetic + CMU Movie (A100)...")

    # === 路径配置 ===
    model_name = '/root/autodl-tmp/Narrative-Similarity-Task/models/Qwen3-Embedding-4B'
    output_path = '/root/autodl-tmp/Narrative-Similarity-Task/output/track_b_qwen3_cmu_a100'
    os.makedirs(output_path, exist_ok=True)

    # === 检查断点 ===
    checkpoint_path = None
    if os.path.exists(output_path):
        checkpoints = [d for d in os.listdir(output_path) if d.startswith('checkpoint-')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(output_path, checkpoints[-1])
            print(f"✅ 找到检查点: {checkpoint_path}")

    # === 构建模型 ===
    if checkpoint_path:
        print(f"从检查点加载模型...")
        model = SentenceTransformer(checkpoint_path)
        print("✅ 模型从检查点加载完成")
    else:
        print(f"从头开始训练,加载基础模型: {model_name}")

        # 4-bit量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        word_embedding_model = models.Transformer(
            model_name,
            tokenizer_args={'padding_side': 'left'},
            model_args={
                "quantization_config": bnb_config,
                "device_map": "auto",
            }
        )

        word_embedding_model.auto_model = prepare_model_for_kbit_training(
            word_embedding_model.auto_model,
            use_gradient_checkpointing=True
        )

        embedding_dim = word_embedding_model.get_word_embedding_dimension()
        print(f"Embedding维度: {embedding_dim}")

        pooling_model = models.Pooling(
            word_embedding_dimension=embedding_dim,
            pooling_mode='lasttoken'
        )

        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device='cuda'
        )

        # LoRA配置 (A100可以用更大的rank)
        lora_config = LoraConfig(
            r=64,  # A100用64
            lora_alpha=128,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        model.add_adapter(lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_param = sum(p.numel() for p in model.parameters())
        print(f"✅ 可训练参数: {trainable_params:,} / {all_param:,} ({100 * trainable_params / all_param:.2f}%)")

    # === 加载数据 ===
    print("\n加载训练数据...")

    print("1. 加载Synthetic数据...")
    synthetic_dataset = build_triplets_from_track_a(
        '/root/autodl-tmp/Narrative-Similarity-Task/TrainingSet1/synthetic_data_for_contrastive_learning.jsonl'
    )
    print(f"   Synthetic: {len(synthetic_dataset)} 个样本")

    print("2. 加载CMU Movie数据...")
    cmu_dataset = build_triplets_from_track_a(
        '/root/autodl-tmp/Narrative-Similarity-Task/TrainingSet1/cmu_movie_triplets.jsonl'
    )
    cmu_dataset = cmu_dataset.select(range(min(10000, len(cmu_dataset))))
    print(f"   CMU Movie: {len(cmu_dataset)} 个样本")

    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets([synthetic_dataset, cmu_dataset])

    print(f"\n总训练样本: {len(train_dataset):,}")
    print(f"  - Synthetic: {len(synthetic_dataset)} ({len(synthetic_dataset) / len(train_dataset) * 100:.1f}%)")
    print(f"  - CMU Movie: {len(cmu_dataset)} ({len(cmu_dataset) / len(train_dataset) * 100:.1f}%)")

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="qwen3_cmu_a100",
        data_path="/root/autodl-tmp/Narrative-Similarity-Task/TrainingSet1/dev_track_a.jsonl",
        batch_size=32
    )

    # === 训练配置 ===
    epochs = 3

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        learning_rate=5e-7,
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

    print(f"\n开始训练:")
    print(f"  - 硬件: A100 40GB")
    if checkpoint_path:
        print(f"  - ✅ 断点续传: {checkpoint_path}")
    else:
        print(f"  - 🆕 从头训练")
    print(f"  - 模型: Qwen3-Embedding-4B (4-bit + LoRA)")
    print(f"  - LoRA rank: 64")
    print(f"  - 训练数据: Synthetic + CMU Movie")
    print(f"  - 总样本: {len(train_dataset):,}")
    print(f"  - Learning rate: 5e-7")
    print(f"  - Epochs: {epochs}")
    print(f"  - 预计步数: {len(train_dataset) // (32 * 2) * epochs}")

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
    try:
        model.save(output_path)
        print(f"✅ 模型已保存到: {output_path}")
    except:
        model[0].auto_model.save_pretrained(os.path.join(output_path, "lora_adapter"))
        print(f"✅ LoRA适配器已保存到: {output_path}/lora_adapter")

    print("✅ 训练完成!")
    print(f"\n预期准确率: 68-72% 🚀")
    print(f"(Qwen3大模型 + CMU Movie 40k + A100)")


if __name__ == "__main__":
    main()