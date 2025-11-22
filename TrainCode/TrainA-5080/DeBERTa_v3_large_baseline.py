"""
DeBERTa-v3-large Multiple Choice 训练
"""
import os
import sys
import logging
import datasets
import numpy as np
from sklearn.metrics import accuracy_score
import torch

from transformers import (
    DebertaV2Tokenizer,
    AutoModelForMultipleChoice,
    Trainer,
    TrainingArguments
)

import warnings
warnings.filterwarnings('ignore')  # 忽略overflow警告


program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


class MultipleChoiceDataCollator:
    """Multiple Choice数据整理器"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])

        flattened_features = [
            {k: v[i] for k, v in f.items() if k != 'labels'}
            for f in features
            for i in range(num_choices)
        ]

        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            return_tensors='pt'
        )

        batch = {
            k: v.view(batch_size, num_choices, -1)
            for k, v in batch.items()
        }

        batch['labels'] = torch.tensor([f['labels'] for f in features])

        return batch


def preprocess_function(examples, tokenizer):
    """预处理函数"""

    anchors = examples.get('anchor_text') or examples.get('anchor_story')
    text_a = examples['text_a']
    text_b = examples['text_b']
    labels = examples['text_a_is_closer']

    # 确保是列表
    if not isinstance(anchors, list):
        anchors = list(anchors)
    if not isinstance(text_a, list):
        text_a = list(text_a)
    if not isinstance(text_b, list):
        text_b = list(text_b)

    # 准备输入
    first_sentences = []
    second_sentences = []

    for i in range(len(anchors)):
        first_sentences.append(str(anchors[i]))
        second_sentences.append(str(text_a[i]))

        first_sentences.append(str(anchors[i]))
        second_sentences.append(str(text_b[i]))

    # Tokenize
    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=512,
        padding=False
    )

    # Reshape
    result = {
        k: [v[i:i+2] for i in range(0, len(v), 2)]
        for k, v in tokenized.items()
    }

    result['labels'] = [0 if label else 1 for label in labels]

    return result


def compute_metrics(eval_pred):
    """计算准确率"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}


def main():
    """主函数"""

    logger.info("="*70)
    logger.info("🚀 DeBERTa-v3-large Multiple Choice 训练")
    logger.info("="*70)

    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

    TRAIN_FILE = f"{PROJECT_ROOT}/TrainingSet1/synthetic_data_for_classification.jsonl"
    DEV_FILE = f"{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl"
    OUTPUT_DIR = f"{PROJECT_ROOT}/models/deberta_v3_large"

    # 加载数据
    logger.info("📂 加载数据...")

    train_dataset = datasets.load_dataset('json', data_files=TRAIN_FILE, split='train')
    dev_dataset = datasets.load_dataset('json', data_files=DEV_FILE, split='train')

    logger.info(f"   训练: {len(train_dataset)}, 验证: {len(dev_dataset)}")

    # 加载模型
    logger.info("🤖 加载模型...")
    model_id = "microsoft/deberta-v3-large"

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)
    model = AutoModelForMultipleChoice.from_pretrained(model_id)

    logger.info(f"   参数: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # 预处理
    logger.info("🔧 预处理...")

    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="训练集"
    )

    tokenized_dev = dev_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dev_dataset.column_names,
        desc="验证集"
    )

    logger.info("   ✅ 预处理完成")

    # Data collator
    data_collator = MultipleChoiceDataCollator(tokenizer=tokenizer)

    # 训练配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-6,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,
        gradient_checkpointing=False,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to='none',
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 训练
    logger.info("="*70)
    logger.info("🚀 开始训练")

    trainer.train()

    # 评估
    logger.info("="*70)
    logger.info("📊 评估")
    logger.info("="*70)
    results = trainer.evaluate()
    accuracy = results['eval_accuracy']
    logger.info(f"\n📈 Dev准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy


if __name__ == '__main__':
    accuracy = main()