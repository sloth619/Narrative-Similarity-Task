"""
Track A训练 - Qwen3-Reranker-4B with QLoRA (WSL on 5080)
使用Synthetic数据 + QLoRA微调 - 优化显存版本
"""
import os

# 解决tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
from tqdm import tqdm
import json


def build_pairs_from_track_a(data_path):
    """从Track A构建训练数据 - 生成正负样本对"""
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_data = []
    for item in dataset:
        anchor = item.get('anchor_text')
        text_a = item.get('text_a')
        text_b = item.get('text_b')
        label_a_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]):
            continue

        # 确定正样本和负样本
        if label_a_closer is not None:
            positive = text_a if label_a_closer else text_b
            negative = text_b if label_a_closer else text_a
        else:
            positive = text_a
            negative = text_b

        # 添加正样本对 (label=1.0)
        train_data.append({
            'text1': anchor,
            'text2': positive,
            'label': 1.0
        })

        # 添加负样本对 (label=0.0)
        train_data.append({
            'text1': anchor,
            'text2': negative,
            'label': 0.0
        })

    return Dataset.from_list(train_data)


class TrackA_Accuracy_Evaluator:
    """Track A准确率评估器 - 三选一分类"""

    def __init__(self, name: str, data_path: str, tokenizer, max_length: int = 512):
        self.name = name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.load_data(data_path)

    def load_data(self, data_path: str):
        """加载验证数据"""
        print(f"Evaluator: 正在加载并清洗 {data_path}...")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    self.samples.append({
                        'anchor': data['anchor_text'],
                        'text_a': data['text_a'],
                        'text_b': data['text_b'],
                        'label': data['text_a_is_closer']
                    })
                except:
                    continue

        print(f"Evaluator: 加载了 {len(self.samples)} 个干净的验证样本。\n")

    def __call__(self, model, device):
        """评估模型并返回准确率"""
        model.eval()
        correct = 0
        total = len(self.samples)

        with torch.no_grad():
            for sample in self.samples:
                # 编码 anchor-text_a
                inputs_a = self.tokenizer(
                    sample['anchor'],
                    sample['text_a'],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(device)

                # 编码 anchor-text_b
                inputs_b = self.tokenizer(
                    sample['anchor'],
                    sample['text_b'],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(device)

                # 获取相关性分数
                score_a = model(**inputs_a).logits.squeeze().item()
                score_b = model(**inputs_b).logits.squeeze().item()

                # 预测: text_a分数更高则为True
                pred = score_a > score_b

                if pred == sample['label']:
                    correct += 1

        accuracy = correct / total
        model.train()
        return accuracy


class EvaluateCallback(TrainerCallback):
    """自定义回调 - 在每个epoch结束时评估"""

    def __init__(self, evaluator, device):
        self.evaluator = evaluator
        self.device = device
        self.best_accuracy = 0.0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Epoch结束时评估"""
        if model is not None:
            print(f"\n[Validation {self.evaluator.name}] Epoch: {state.epoch:.1f}, Steps: {state.global_step}")

            accuracy = self.evaluator(model, self.device)

            # 判断是否为最佳
            is_best = accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = accuracy

            print(f"Accuracy: {accuracy:.4f} ({int(accuracy*len(self.evaluator.samples))}/{len(self.evaluator.samples)}){' ⭐ New best!' if is_best else ''}")

        return control


def main():
    print("🚀 Track A训练 - Qwen3-Reranker-4B with QLoRA (WSL on 5080)...")
    print("优化显存使用版本")

    # === WSL路径配置 ===
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    model_name = '/mnt/e/model/Qwen3-Reranker-4B'
    output_path = f'{PROJECT_ROOT}/output/track_a_qwen3_reranker_4B_qlora_wsl'
    os.makedirs(output_path, exist_ok=True)

    dev_track_a_path = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'
    synthetic_data_path = f'{PROJECT_ROOT}/TrainingSet1/synthetic_data_for_classification.jsonl'

    # === 构建模型 with QLoRA ===
    print(f"加载模型: {model_name}")
    print("使用4-bit量化配置...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # 开启梯度检查点以节省显存
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    print(f"模型加载完成")

    # === LoRA配置 ===
    print("\n配置LoRA适配器...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"✅ 可训练参数: {trainable_params:,} / {all_param:,} ({100 * trainable_params / all_param:.2f}%)")

    # === 加载数据 ===
    print("\n从synthetic数据构建训练集...")
    train_dataset = build_pairs_from_track_a(synthetic_data_path)
    print(f"训练样本: {len(train_dataset):,}")

    # === 数据预处理 ===
    def preprocess_function(examples):
        return tokenizer(
            examples['text1'],
            examples['text2'],
            padding='max_length',
            truncation=True,
            max_length=512,
        )

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing"
    )

    # === 评估器 ===
    evaluator = TrackA_Accuracy_Evaluator(
        name="reranker_4B_synthetic",
        data_path=dev_track_a_path,
        tokenizer=tokenizer,
        max_length=512
    )

    # === 训练配置 ===
    epochs = 3

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="no",  # 改为no,使用callback手动评估
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    print(f"\n开始训练:")
    print(f"  - 模型: Qwen3-Reranker-4B with QLoRA")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Gradient Accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Epochs: {epochs}")
    print(f"  - LoRA r: {lora_config.r}")
    print(f"  - Gradient Checkpointing: ✅")

    # === 数据整理函数 ===
    def data_collator(features):
        batch = {
            'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in features]),
            'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in features]),
            'labels': torch.tensor([f['label'] for f in features], dtype=torch.float32),
        }
        return batch

    # === 清理显存 ===
    torch.cuda.empty_cache()

    # === 创建评估回调 ===
    eval_callback = EvaluateCallback(
        evaluator=evaluator,
        device=training_args.device
    )

    # === 训练 ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[eval_callback],
    )

    print("\n开始训练...")
    trainer.train()

    # === 最后评估一次 ===
    print(f"\n最终评估...")
    final_accuracy = evaluator(model, training_args.device)
    print(f"Final Accuracy: {final_accuracy:.4f}")

    # === 保存 ===
    print("\n保存最终模型...")
    try:
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"✅ 模型已保存到: {output_path}")
    except Exception as e:
        print(f"完整模型保存失败: {e}")
        lora_adapter_path = os.path.join(output_path, "lora_adapter")
        model.save_pretrained(lora_adapter_path)
        tokenizer.save_pretrained(lora_adapter_path)
        print(f"✅ LoRA适配器已保存到: {lora_adapter_path}")

    print("\n✅ 训练完成!")


if __name__ == "__main__":
    main()