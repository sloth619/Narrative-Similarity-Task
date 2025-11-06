import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from sentence_transformers import SentenceTransformer, losses, InputExample, models
from torch.utils.data import DataLoader
from datasets import load_dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator_NoSave
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training


def main():
    print("🚀 开始 Track B [AUGMENTED V2] 训练 (QLoRA + 低学习率 + 7 模块)...")

    model_name = '/root/autodl-tmp/Qwen3-Embedding-4B'

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

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 全部 7 个层
    )

    model.add_adapter(lora_config)

    # 打印可训练参数信息
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

    # === 1. 加载增强的训练数据 (已修复 1913 bug) ===

    print("正在加载: train_track_b_mixed_10k.jsonl (Augmented Pairs)")
    paired_dataset = load_dataset('json', data_files='../../TrainingSet2/train_track_b_mixed_10k.jsonl', split='train')

    # --- ❗ [FIX] 修复锚点匹配逻辑 ---
    all_originals_map = {}
    print("正在加载 dev_b (用于匹配锚点)...")
    dev_b = load_dataset('json', data_files='../../TrainingSet1/dev_track_b.jsonl', split='train')
    for i, item in enumerate(dev_b):
        text = item.get('text')
        if text:
            all_originals_map[i] = text  # 索引 0-478

    print("正在加载 synthetic_b (用于匹配锚点)...")
    synthetic_b_offset = len(dev_b)  # 479
    synthetic_b = load_dataset('json', data_files='../../TrainingSet1/synthetic_data_for_contrastive_learning.jsonl',
                               split='train')
    for i, item in enumerate(synthetic_b):
        text = item.get('anchor_story') or item.get('text')  # 确保能读到
        if text:
            all_originals_map[i + synthetic_b_offset] = text  # 索引 479-2375
    # --- End of Fix ---

    pair_examples = []
    for item in paired_dataset:
        if item.get('_augmented'):
            source_idx = item.get('_source_index')
            if source_idx in all_originals_map:
                anchor_text = all_originals_map[source_idx]
                positive_text = item.get('text')

                if anchor_text and positive_text:
                    pair_examples.append(InputExample(
                        texts=[anchor_text, positive_text]
                    ))
    print(f"加载了 {len(pair_examples)} 个干净的增强正样本对 (已修复)")

    # === 2. 定义损失函数 ===

    # [FIX] 只使用 MNRL 损失函数
    pair_loader = DataLoader(pair_examples, shuffle=True, batch_size=64)
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 3. 定义评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="augmented_low_lr",
        data_path="../../TrainingSet1/dev_track_a.jsonl",
        batch_size=64
    )

    # === 4. 开始训练 ===
    epochs = 3  # 训练 3 轮
    warmup_steps = int(len(pair_loader) * epochs * 0.1)  # 10% 预热
    output_path = '../../output/track_b_augmented_model_v2_qlora'
    os.makedirs(output_path, exist_ok=True)

    print(f"开始训练，批次大小: pair=64, epochs=3")

    model.fit(
        train_objectives=[
            (pair_loader, mnrl_loss),
            # [FIX] 移除旧的 triplet 损失
        ],
        evaluator=evaluator,
        evaluation_steps=200,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        save_best_model=False,
        show_progress_bar=True,
        learning_rate=5e-7  # ❗❗ [FIX] 设置一个非常低的微调学习率 ❗❗
    )

    # 手动保存最终模型
    print("\n正在保存最终模型...")
    try:
        model.save(output_path)
        print(f"✅ 模型已保存到: {output_path}")
    except Exception as e:
        print(f"警告: model.save() 失败: {e}")
        print("尝试仅保存 LoRA 适配器...")
        try:
            model[0].auto_model.save_pretrained(os.path.join(output_path, "lora_adapter"))
            print(f"✅ LoRA 适配器已保存")
        except Exception as e2:
            print(f"❌ 适配器保存也失败了: {e2}")

    print("✅ 训练完成!")


if __name__ == "__main__":
    main()