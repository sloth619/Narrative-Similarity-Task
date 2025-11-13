"""
Track B训练 - Qwen3-Embedding-8B (WSL on 5080)
使用Synthetic数据 + QLoRA微调 - 优化显存版本
"""
import os

# 解决tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses, models
from datasets import load_dataset, Dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training


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
    print("🚀 Track B训练 - Qwen3-Embedding-8B with QLoRA (WSL on 5080)...")
    print("优化显存使用版本")

    # === WSL路径配置 ===
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    model_name = '/mnt/e/model/Qwen3-Embedding-8B'
    output_path = f'{PROJECT_ROOT}/output/track_b_qwen3_8B_qlora_wsl'
    os.makedirs(output_path, exist_ok=True)

    dev_track_a_path = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'
    synthetic_data_path = f'{PROJECT_ROOT}/TrainingSet1/synthetic_data_for_contrastive_learning.jsonl'

    # === 构建模型 with QLoRA ===
    print(f"加载模型: {model_name}")
    print("使用4-bit量化配置...")

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
            "attn_implementation": "flash_attention_2",
        }
    )

    # 开启梯度检查点以节省显存
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

    # === LoRA配置 - 减小r值节省显存 ===
    print("\n配置LoRA适配器...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
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
    print("\n从synthetic数据构建训练集...")
    train_dataset = build_triplets_from_track_a(synthetic_data_path)
    print(f"训练样本: {len(train_dataset):,}")

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="qwen3_8B_synthetic",
        data_path=dev_track_a_path,
        batch_size=8
    )

    # === 优化训练配置 ===
    epochs = 5

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="eval_evaluator",
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
    )

    print(f"\n开始训练:")
    print(f"  - 模型: Qwen3-Embedding-8B with QLoRA")
    print(f"  - Batch size: {training_args.per_device_train_batch_size} (减小以节省显存)")
    print(f"  - Gradient Accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Epochs: {epochs}")
    print(f"  - LoRA r: {lora_config.r} (减小以节省显存)")
    print(f"  - Gradient Checkpointing: ✅")

    # === 清理显存 ===
    torch.cuda.empty_cache()

    # === 训练 ===
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=mnrl_loss,
        evaluator=evaluator,
    )

    print("\n开始训练...")
    trainer.train()

    # === 保存 ===
    print("\n保存最终模型...")
    try:
        model.save(output_path)
        print(f"✅ 模型已保存到: {output_path}")
    except Exception as e:
        print(f"完整模型保存失败: {e}")
        lora_adapter_path = os.path.join(output_path, "lora_adapter")
        model[0].auto_model.save_pretrained(lora_adapter_path)
        print(f"✅ LoRA适配器已保存到: {lora_adapter_path}")

    print("\n✅ 训练完成!")


if __name__ == "__main__":
    main()