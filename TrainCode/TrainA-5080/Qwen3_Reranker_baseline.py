"""
Track A训练 - Qwen3-Reranker-4B
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from datasets import Dataset
from tqdm import tqdm
from transformers import BitsAndBytesConfig


def load_training_data(data_path):
    """加载训练数据 - 返回Dataset对象"""
    samples = []

    print(f"📂 加载训练数据: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                anchor = data['anchor_text']
                text_a = data['text_a']
                text_b = data['text_b']
                label = data['text_a_is_closer']

                positive = text_a if label else text_b
                negative = text_b if label else text_a

                samples.append({
                    'sentence1': str(anchor),
                    'sentence2': str(positive),
                    'label': 1.0
                })
                samples.append({
                    'sentence1': str(anchor),
                    'sentence2': str(negative),
                    'label': 0.0
                })

            except Exception as e:
                if line_num <= 5:
                    print(f"⚠️  Line {line_num} 错误: {e}")
                continue

    print(f"✅ 加载了 {len(samples)} 个训练样本")

    dataset = Dataset.from_list(samples)
    dataset = dataset.map(
        lambda x: {
            'sentence1': str(x['sentence1']) if x['sentence1'] is not None else "",
            'sentence2': str(x['sentence2']) if x['sentence2'] is not None else "",
            'label': float(x['label'])
        }
    )
    return dataset


def load_dev_data(data_path):
    """加载验证数据"""
    samples = []
    print(f"📂 加载验证数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                samples.append({
                    'anchor': data['anchor_text'],
                    'text_a': data['text_a'],
                    'text_b': data['text_b'],
                    'label': data['text_a_is_closer']
                })
            except:
                continue
    print(f"✅ 加载了 {len(samples)} 个验证样本\n")
    return samples


def evaluate_track_a(model, dev_samples):
    """评估Track A准确率"""
    correct = 0
    total = len(dev_samples)
    print("\n🔍 开始评估...")
    for sample in tqdm(dev_samples, desc="Evaluating"):
        score_a = model.predict([[sample['anchor'], sample['text_a']]])[0]
        score_b = model.predict([[sample['anchor'], sample['text_b']]])[0]
        pred = score_a > score_b
        if pred == sample['label']:
            correct += 1
    accuracy = correct / total
    return accuracy


class TrackAEvaluator:
    def __init__(self, dev_samples, name="track_a"):
        self.dev_samples = dev_samples
        self.name = name
        self.best_accuracy = 0.0

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        print(f"\n{'='*60}")
        print(f"[Validation {self.name}] Epoch: {epoch}, Steps: {steps}")
        correct = 0
        total = len(self.dev_samples)
        for sample in self.dev_samples:
            score_a = model.predict([[sample['anchor'], sample['text_a']]])[0]
            score_b = model.predict([[sample['anchor'], sample['text_b']]])[0]
            pred = score_a > score_b
            if pred == sample['label']:
                correct += 1
        accuracy = correct / total
        is_best = accuracy > self.best_accuracy
        if is_best:
            self.best_accuracy = accuracy
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total}){' ⭐ New best!' if is_best else ''}")
        print(f"{'='*60}\n")
        return accuracy


def main():
    print("🚀 Track A训练 - Qwen3-Reranker-4B (新Trainer API + 4bit量化)")
    print("="*60)

    MODEL_NAME = '/mnt/e/model/Qwen3-Reranker-4B'
    TRAIN_DATA = '/mnt/e/Code/python/Narrative-Similarity-Task/TrainingSet1/synthetic_data_for_classification.jsonl'
    DEV_DATA = '/mnt/e/Code/python/Narrative-Similarity-Task/TrainingSet1/dev_track_a.jsonl'
    OUTPUT_DIR = '/mnt/e/Code/python/Narrative-Similarity-Task/output/track_a_trainer_4bit'
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 2
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n🔧 配置4-bit量化...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\n📦 加载模型: {MODEL_NAME}")
    model = CrossEncoder(
        MODEL_NAME,
        num_labels=1,
        max_length=MAX_LENGTH,
        device='cuda',
        model_kwargs={
            'quantization_config': quantization_config,
            'dtype': torch.bfloat16,
        }
    )

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.model.config.pad_token_id = model.tokenizer.pad_token_id
        print(f"✅ 设置 pad_token = eos_token")
    print(f"✅ 模型加载完成 (4-bit量化)")

    train_dataset = load_training_data(TRAIN_DATA)
    dev_samples = load_dev_data(DEV_DATA)

    print(f"\n⚙️  训练配置:")
    print(f"  - 训练样本: {len(train_dataset)}")
    print(f"  - 验证样本: {len(dev_samples)}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Learning rate: {LEARNING_RATE}")

    print(f"\n🔍 Zero-shot评估:")
    zero_shot_acc = evaluate_track_a(model, dev_samples)
    print(f"✅ Zero-shot Accuracy: {zero_shot_acc:.4f}")

    training_args = CrossEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        push_to_hub=False,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_8bit",
        max_grad_norm=0.3,
    )

    evaluator = TrackAEvaluator(dev_samples, name="reranker_4B")
    model.model_card_data.generate_model_card = False

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        evaluator=[evaluator],
    )

    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if not cb.__class__.__name__ == 'ModelCardCallback'
    ]

    torch.cuda.empty_cache()

    print(f"\n🎯 开始训练...")
    trainer.train()

    print(f"\n📊 最终评估:")
    final_accuracy = evaluate_track_a(model, dev_samples)
    print(f"✅ Final Accuracy: {final_accuracy:.4f}")

    print(f"\n💾 保存最终模型...")
    final_model_path = os.path.join(OUTPUT_DIR, 'final_model')
    model.save(final_model_path)
    print(f"✅ 模型已保存到: {final_model_path}")

    print(f"\n{'='*60}")
    print(f"✅ 训练完成!")
    print(f"📊 性能对比:")
    print(f"   Zero-shot: {zero_shot_acc:.4f}")
    print(f"   Fine-tuned: {final_accuracy:.4f}")
    if final_accuracy > zero_shot_acc:
        improvement = final_accuracy - zero_shot_acc
        print(f"   提升: +{improvement:.4f} ({improvement/zero_shot_acc*100:+.1f}%)")
    else:
        change = final_accuracy - zero_shot_acc
        print(f"   变化: {change:.4f} ({change/zero_shot_acc*100:.1f}%)")
    print(f"📁 模型保存位置: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()