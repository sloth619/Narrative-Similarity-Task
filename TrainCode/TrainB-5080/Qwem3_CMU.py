"""
Track B训练 - Qwen3-Embedding-4B (V13 - RTX 5080 + 20k CMU)
"""
import os
import gc
import torch

# 清理显存
torch.cuda.empty_cache()
gc.collect()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses, models
from datasets import load_dataset, Dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training


def build_triplets_baseline_logic(data_path, max_length=200):
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

        # 限制文本长度 (保持 VRAM 占用可控)
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

        # --- Baseline 的数据增强 (1 -> 3) ---
        train_data.append({'sentence1': anchor, 'sentence2': positive})
        train_data.append({'sentence1': anchor, 'sentence2': anchor})
        train_data.append({'sentence1': positive, 'sentence2': positive})
        # -------------------------------------

    if skipped > 0:
        print(f"  ⚠️ 跳过了 {skipped} 条数据")

    return Dataset.from_list(train_data)


def main():
    print("🚀 Track B训练 - Qwen3-4B (V13 - RTX 5080 + 20k CMU)...")

    # === 路径配置 (5080 WSL 路径) ===
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    model_name = '/mnt/e/model/Qwen3-Embedding-4B'
    # V10 (bs=16, r=32, 10k data) 的输出路径
    v10_output_path = f'{PROJECT_ROOT}/output/track_b_qwen3_cmu_5080_v9_bs128'
    # V13 (bs=16, r=32, 20k data) 的新输出路径
    output_path = f'{PROJECT_ROOT}/output/track_b_qwen3_cmu_5080_v13_20k_fix' # V13 新名字
    os.makedirs(output_path, exist_ok=True)

    # 数据集路径
    dev_track_a_path = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'
    synthetic_data_path = f'{PROJECT_ROOT}/TrainingSet1/synthetic_data_for_contrastive_learning.jsonl'
    cmu_data_path = f'{PROJECT_ROOT}/TrainingSet1/cmu_movie_triplets.jsonl'

    # === 检查断点 ===
    # 我们从 V13 自己的路径加载断点
    checkpoint_path = None
    if os.path.exists(output_path):
        checkpoints = [d for d in os.listdir(output_path) if d.startswith('checkpoint-')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(output_path, checkpoints[-1])
            print(f"✅ 找到 V13 检查点: {checkpoint_path}")

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
            model_name, # <-- 总是加载基础模型
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

        # LoRA配置 (r=32 保持不变)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        print("✅ 无条件添加 LoRA (r=32) 适配器...")
        model.add_adapter(lora_config)

        v10_best_checkpoint = os.path.join(v10_output_path, "checkpoint-246")
        if os.path.exists(v10_best_checkpoint):
             print(f"✅ 正在从 V10 断点 ({v10_best_checkpoint}) 热启动 LoRA 权重...")
             try:
                model.load_adapter(v10_best_checkpoint, "default")
                print("✅ V10 LoRA 权重加载成功!")
                checkpoint_path = v10_best_checkpoint
             except Exception as e:
                print(f"⚠️ 加载 V10 LoRA 权重失败 (可能是 V10 OOM 导致): {e}")
                print("   将从头训练...")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"✅ 可训练参数: {trainable_params:,} / {all_param:,} ({100 * trainable_params / all_param:.2f}%)")
    print("✅ 模型已成功加载到 VRAM。")


    # === 加载数据 ===
    print("\n加载训练数据 (Baseline 逻辑)...")

    print("1. 加载Synthetic数据...")
    synthetic_dataset = build_triplets_baseline_logic(
        synthetic_data_path,
        max_length=200
    )
    print(f"   Synthetic: {len(synthetic_dataset)} 个样本")

    print("2. 加载CMU Movie数据...")
    cmu_dataset = build_triplets_baseline_logic(
        cmu_data_path,
        max_length=200
    )

    # 🔥 [V12 优化] 增加数据量
    cmu_sample_size = 20000
    cmu_dataset = cmu_dataset.select(range(min(cmu_sample_size, len(cmu_dataset))))
    print(f"   CMU Movie: {len(cmu_dataset)} 个样本 (已增加至 {cmu_sample_size})")

    from datasets import concatenate_datasets
    train_dataset = concatenate_datasets([synthetic_dataset, cmu_dataset])
    print(f"\n总训练样本: {len(train_dataset):,}") # 约 25,691

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="qwen3_cmu_5080_v13_20k",
        data_path=dev_track_a_path,
        batch_size=32
    )

    # === 训练配置 (保持 V10 配置) ===
    epochs = 3
    current_batch_size = 24
    current_grad_steps = 6
    effective_batch_size = current_batch_size * current_grad_steps # 16 * 8 = 128
    current_learning_rate = 5e-7

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=current_batch_size,
        gradient_accumulation_steps=current_grad_steps,

        learning_rate=current_learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        optim="paged_adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1e-8,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,

        # 🔥 OOM 修复
        load_best_model_at_end=False,

        logging_steps=50,
        metric_for_best_model="eval_evaluator",
        bf16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        # resume_from_checkpoint=checkpoint_path, # <-- V13 中断/恢复逻辑
    )

    print(f"\n开始训练 (V13 - 20k CMU 数据版):")
    resume_from_v13_checkpoint = checkpoint_path if "v13" in str(checkpoint_path) else None
    if resume_from_v13_checkpoint:
        print(f"  - ✅ 断点续传: {resume_from_v13_checkpoint}")
    elif "v9" in str(checkpoint_path): # (v9_bs128 是 V10)
        print(f"  - 🔥 V10 热启动: {checkpoint_path}")
    else:
        print(f"  - 🆕 从头训练")

    print(f"  - 模型: Qwen3-Embedding-4B (4-bit + LoRA r=32)")
    print(f"  - 总样本: {len(train_dataset):,}")
    print(f"  - BS (有效): {effective_batch_size}")
    print(f"  - 学习率: {current_learning_rate}")
    print(f"  - 预期: 训练时间更长, 性能有望超越 65.0%")

    # === 训练 ===
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=mnrl_loss,
        evaluator=evaluator,
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from_v13_checkpoint)
    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断!")
        print("💾 检查点已保存,可以重新运行脚本继续训练")
        return
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ 显存不足 (OOM)! r=32 + 20k 数据失败。")
            print("💡 抱歉, 16GB VRAM 无法处理 r=32 + 20k 数据。请坚持 V10 (10k 数据)。")
        else:
            raise e
        return

    # === 保存 ===
    print("\n保存最终模型...")
    try:
        model.save(output_path)
        print(f"✅ 最终模型已保存到: {output_path}")
    except:
        lora_path = os.path.join(output_path, "lora_adapter")
        model[0].auto_model.save_pretrained(lora_path)
        print(f"✅ 完整保存失败, 仅 LoRA 适配器已保存到: {lora_path}")

    print("✅ 训练完成!")
    print(f"📊 Qwen3-4B (5080 V13) 性能总结:")
    print(f"  - V10 (r=32, 10k) 峰值: 65.0%")
    print(f"  - V13 (r=32, 20k) 微调后: 查看上方最佳准确率 (目标 > 65.0%)")


if __name__ == "__main__":
    main()