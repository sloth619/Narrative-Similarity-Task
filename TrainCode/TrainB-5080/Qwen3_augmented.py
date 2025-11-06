import os
from sentence_transformers import SentenceTransformer, losses, InputExample, models
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave

from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training


def main():
    print("🚀 开始 Track B [AUGMENTED V2] 训练 (5080 QLoRA + Trainer)...")

    model_name = 'E:/model/Qwen3-Embedding-4B'

    # === 构建模型 ===
    print(f"Manually building model from: {model_name} with QLoRA")
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

    # 启用梯度检查点
    word_embedding_model.auto_model = prepare_model_for_kbit_training(
        word_embedding_model.auto_model,
        use_gradient_checkpointing=True
    )

    embedding_dim = word_embedding_model.get_word_embedding_dimension()
    print(f"Word Embedding Dimension: {embedding_dim}")

    pooling_model = models.Pooling(
        word_embedding_dimension=embedding_dim,
        pooling_mode='lasttoken'
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device='cuda'
    )

    # LoRA 配置 (使用 7 模块)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model.add_adapter(lora_config)

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"✅ Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")

    if trainable_params == 0:
        raise RuntimeError("❌ 没有可训练参数！")

    print("Model build successful with QLoRA.")

    # === 1. 加载增强的训练数据 ===

    print("正在加载: train_track_b_mixed_10k.jsonl (Augmented Pairs)")
    paired_dataset = load_dataset('json', data_files='../../TrainingSet2/train_track_b_mixed_10k.jsonl', split='train')

    # --- 锚点匹配逻辑 ---
    all_originals_map = {}
    print("正在加载 dev_b (用于匹配锚点)...")
    dev_b = load_dataset('json', data_files='../../TrainingSet1/dev_track_b.jsonl', split='train')
    for i, item in enumerate(dev_b):
        text = item.get('text')
        if text:
            all_originals_map[i] = text

    print("正在加载 synthetic_b (用于匹配锚点)...")
    synthetic_b_offset = len(dev_b)
    synthetic_b = load_dataset('json', data_files='../../TrainingSet1/synthetic_data_for_contrastive_learning.jsonl',
                               split='train')
    for i, item in enumerate(synthetic_b):
        text = item.get('anchor_story') or item.get('text')
        if text:
            all_originals_map[i + synthetic_b_offset] = text

    pair_data_list = []
    for item in paired_dataset:
        if item.get('_augmented'):
            source_idx = item.get('_source_index')
            if source_idx in all_originals_map:
                anchor_text = all_originals_map[source_idx]
                positive_text = item.get('text')

                if anchor_text and positive_text:
                    pair_data_list.append({
                        'sentence1': anchor_text,  # 统一为 sentence1
                        'sentence2': positive_text  # 统一为 sentence2
                    })
    print(f"加载了 {len(pair_data_list)} 个干净的增强正样本对 (已修复)")

    train_dataset = Dataset.from_list(pair_data_list)
    print(f"✅ 已转换为 Dataset 格式: {len(train_dataset)} 条记录")

    # === 2. 定义损失函数 ===

    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 3. 定义评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="augmented_low_lr",
        data_path="../../TrainingSet1/dev_track_a.jsonl",
        batch_size=16
    )

    # === 4. 开始训练 ===
    epochs = 3
    output_path = '../../output/track_b_augmented_model_v2_qlora_5080'
    os.makedirs(output_path, exist_ok=True)

    # 1. 定义训练参数
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=5e-7,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=60,
        save_strategy="steps",
        save_steps=60,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=50,
        dataloader_num_workers=0,
        metric_for_best_model="eval_evaluator",
    )

    print(
        f"开始训练，批次大小: {training_args.per_device_train_batch_size}, 梯度累计: {training_args.gradient_accumulation_steps}, epochs=3")

    # 2. 创建训练器
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=mnrl_loss,
        evaluator=evaluator,
    )

    trainer.train()

    print("\n正在保存最终模型 (来自 trainer.state.best_model_checkpoint)...")
    try:
        model[0].auto_model.save_pretrained(os.path.join(output_path, "best_lora_adapter"))
        print(f"✅ 最佳 LoRA 适配器已保存到: {output_path}/best_lora_adapter")
    except Exception as e:
        print(f"❌ 适配器保存失败: {e}")

    print("✅ 训练完成!")


if __name__ == "__main__":
    main()