import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer, losses, InputExample, models
from torch.utils.data import DataLoader
from datasets import load_dataset
from train_b_evaluator import TrackB_Accuracy_Evaluator

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training


def main():
    print("🚀 开始 Track B [优化版] 训练 (QLoRA)...")

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

    # ✅ 增加 LoRA 的训练容量
    lora_config = LoraConfig(
        r=32,  # 16 -> 32，增加秩
        lora_alpha=64,  # 32 -> 64
        lora_dropout=0.1,  # 0.05 -> 0.1，增加正则化
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 全部层
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

    print("Model build successful with QLoRA.")

    # === 加载数据 ===
    print("正在加载: synthetic_data_for_contrastive_learning.jsonl")
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

    print("正在加载: dev_track_b.jsonl")
    dev_b = load_dataset('json', data_files='../../TrainingSet1/dev_track_b.jsonl', split='train')

    pair_examples = []
    for item in dev_b:
        text = item.get('text')
        if text:
            pair_examples.append(InputExample(texts=[text, text]))
    print(f"加载了 {len(pair_examples)} 个干净的正样本对")

    triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=64)
    triplet_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE)

    pair_loader = DataLoader(pair_examples, shuffle=True, batch_size=64)
    mnrl_loss = losses.MultipleNegativesRankingLoss(model=model)

    # === 修改评估器：移除保存功能 ===
    from train_b_evaluator_fixed import TrackB_Accuracy_Evaluator_NoSave
    evaluator = TrackB_Accuracy_Evaluator_NoSave(
        name="optimized",
        data_path="../../TrainingSet1/dev_track_a.jsonl"
    )

    # === 训练配置 ===
    epochs = 5  # 2 -> 5，增加训练轮数
    warmup_steps = 200  # 100 -> 200
    output_path = '../../output/track_b_optimized_model_qlora'
    os.makedirs(output_path, exist_ok=True)

    print(f"开始训练，批次大小: triplet=64, pair=64, epochs=5")

    model.fit(
        train_objectives=[
            (triplet_loader, triplet_loss),  # 调换顺序，先 triplet
            (pair_loader, mnrl_loss),
        ],
        evaluator=evaluator,
        evaluation_steps=200,  # 500 -> 200，更频繁评估
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        save_best_model=False,  # 禁用自动保存
        show_progress_bar=True,
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