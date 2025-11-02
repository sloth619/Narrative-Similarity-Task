import os

# [FIX] 解决显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses, InputExample, models  # 导入 'models'
from torch.utils.data import DataLoader
from datasets import load_dataset
# 确保你已经创建了 train_b_evaluator_fixed.py
from train_b_evaluator_fixed import TrackB_Accuracy_Evaluator_NoSave

# [FIX] 导入 QLoRA 和梯度检查点所需的库
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training  # 导入 prepare_model_...


def main():
    print("🚀 开始 Track B [AUGMENTED] 训练 (QLoRA 优化版)...")

    model_name = '/root/autodl-tmp/Qwen3-Embedding-4B'

    # === 构建模型 (与 baseline 完全一致) ===
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

    # 增加 LoRA 的训练容量 (与 baseline 一致)
    lora_config = LoraConfig(
        r=32,  # 16 -> 32
        lora_alpha=64,  # 32 -> 64
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 全部层
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

    # === 1. 加载增强的训练数据 (已修复脏数据过滤) ===

    print("正在加载: synthetic_data_for_contrastive_learning.jsonl (Triplets)")
    triplet_dataset = load_dataset('json',
                                   data_files='../../TrainingSet1/synthetic_data_for_contrastive_learning.jsonl',
                                   split='train')

    triplet_examples = []
    for item in triplet_dataset:
        anchor = item.get('anchor_story')
        positive = item.get('similar_story')
        negative = item.get('dissimilar_story')
        if all([anchor, positive, negative]):
            triplet_examples.append(InputExample(texts=[anchor, positive, negative]))
    print(f"加载了 {len(triplet_examples)} 个干净的三元组样本")

    print("正在加载: train_track_b_mixed_10k.jsonl (Augmented Pairs)")
    paired_dataset = load_dataset('json', data_files='../../TrainingSet2/train_track_b_mixed_10k.jsonl', split='train')

    # --- 配对 (Anchor, Positive) ---
    all_originals_map = {}
    print("正在加载: dev_track_b.jsonl (用于匹配锚点)...")
    dev_b = load_dataset('json', data_files='../../TrainingSet1/dev_track_b.jsonl', split='train')
    for i, item in enumerate(dev_b):
        text = item.get('text')
        if text:
            all_originals_map[i] = text

    pair_examples = []
    for item in paired_dataset:
        if item.get('_augmented'):  # 只使用新生成的
            source_idx = item.get('_source_index')
            # 确认 V3.4 脚本只从 dev_b (索引 < 479) 生成了有效数据
            if source_idx in all_originals_map:
                anchor_text = all_originals_map[source_idx]
                positive_text = item.get('text')

                if anchor_text and positive_text:
                    pair_examples.append(InputExample(
                        texts=[anchor_text, positive_text]
                    ))
    print(f"加载了 {len(pair_examples)} 个干净的增强正样本对")

    # === 2. 定义损失函数 ===

    # 使用与 baseline 一致的批处理大小
    triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=64)
    triplet_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE)

    pair_loader = DataLoader(pair_examples, shuffle=True, batch_size=64)
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 3. 定义评估器 ===
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="augmented",
        data_path="../../TrainingSet1/dev_track_a.jsonl"
    )

    # === 4. 开始训练 ===
    epochs = 5  # 增加训练轮数
    warmup_steps = 200  # 增加预热
    output_path = '../../output/track_b_augmented_model_qlora'  # 新的输出路径
    os.makedirs(output_path, exist_ok=True)

    print(f"开始训练，批次大小: triplet=64, pair=64, epochs=5")

    model.fit(
        train_objectives=[
            (triplet_loader, triplet_loss),  # 调换顺序
            (pair_loader, mnrl_loss),
        ],
        evaluator=evaluator,
        evaluation_steps=200,  # 更频繁评估
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        save_best_model=False,  # 禁用自动保存
        show_progress_bar=True,
        # 不再需要梯度累计，因为 batch_size 已经很大
    )

    # 手动保存最终模型
    print("\n正在保存最终模型...")
    try:
        model.save(output_path)
        print(f"✅ 模型已保存到: {output_path}")
    except:
        model[0].auto_model.save_pretrained(os.path.join(output_path, "lora_adapter"))
        print(f"✅ LoRA 适配器已保存")

    print("✅ 训练完成!")


if __name__ == "__main__":
    main()