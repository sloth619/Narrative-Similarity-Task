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


def build_triplets_from_track_a(data_path):
    """从Track A构建训练数据"""
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_data = []
    for item in dataset:
        # 兼容dev_track_a和synthetic格式
        anchor = item.get('anchor_text') or item.get('anchor_story')
        text_a = item.get('text_a') or item.get('similar_story')
        text_b = item.get('text_b') or item.get('dissimilar_story')

        # dev_track_a用label, synthetic没有label就用None
        label_a_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]):
            continue

        # 如果有标签,用标签;否则假设text_a是正样本
        if label_a_closer is not None:
            positive = text_a if label_a_closer else text_b
        else:
            positive = text_a  # synthetic数据,similar_story是正样本

        # 生成训练样本
        train_data.append({'sentence1': anchor, 'sentence2': positive})
        train_data.append({'sentence1': anchor, 'sentence2': anchor})
        train_data.append({'sentence1': positive, 'sentence2': positive})

    return Dataset.from_list(train_data)


def main():
    print("🚀 Track B训练 - 使用synthetic数据 (5080)...")

    model_name = 'E:/model/Qwen3-Embedding-4B'

    # === 构建模型 ===
    print(f"加载模型: {model_name}")
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

    # LoRA配置
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
    train_dataset = build_triplets_from_track_a(
        '../../TrainingSet1/synthetic_data_for_contrastive_learning.jsonl'
    )
    print(f"训练样本: {len(train_dataset)}")

    # === 损失函数 ===
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 评估器 (用dev_track_a,不重叠) ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="synthetic_train",
        data_path="../../TrainingSet1/dev_track_a.jsonl",
        batch_size=32
    )

    # === 训练配置 ===
    epochs = 5
    output_path = '../../output/track_b_from_synthetic_5080'
    os.makedirs(output_path, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,  # 改这里
        learning_rate=5e-7,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=20,  # 改这里
        metric_for_best_model="eval_evaluator",
        bf16=True
    )

    print(f"\n开始训练:")
    print(f"  - 训练数据: synthetic_data_for_contrastive_learning.jsonl")
    print(f"  - 验证数据: dev_track_a.jsonl (不重叠✅)")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
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
    try:
        model.save(output_path)
        print(f"✅ 模型已保存到: {output_path}")
    except:
        model[0].auto_model.save_pretrained(os.path.join(output_path, "lora_adapter"))
        print(f"✅ LoRA适配器已保存到: {output_path}/lora_adapter")

    print("✅ 训练完成!")


if __name__ == "__main__":
    main()