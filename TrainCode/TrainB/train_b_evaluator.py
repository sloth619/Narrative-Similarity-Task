import json
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import cos_sim
from datasets import load_dataset
import torch
import logging

logger = logging.getLogger(__name__)


class TrackB_Accuracy_Evaluator(SentenceEvaluator):
    """
    [FIXED] 增加了数据清洗，防止 None 值导致崩溃
    """

    def __init__(self, name: str, data_path: str, batch_size: int = 32):
        self.name = name
        self.batch_size = batch_size

        dataset = load_dataset('json', data_files=data_path, split='train')
        self.examples = []

        # === [FIX] 过滤脏数据 ===
        print(f"Evaluator: 正在加载并清洗 {data_path}...")
        for item in dataset:
            anchor = item.get('anchor_text')
            text_a = item.get('text_a')
            text_b = item.get('text_b')
            label = item.get('text_a_is_closer')  # 必须检查 bool?

            # 确保所有文本都不是 None 或空
            if all([anchor, text_a, text_b]) and label is not None:
                self.examples.append({
                    'anchor': anchor,
                    'text_a': text_a,
                    'text_b': text_b,
                    'label_is_a': label
                })
        print(f"Evaluator: 加载了 {len(self.examples)} 个干净的验证样本。")
        # === End of Fix ===

        self.best_score = -1.0

    def __call__(self, model, output_path: str, epoch: int, steps: int) -> float:

        model.eval()

        texts_to_embed = []
        for ex in self.examples:
            texts_to_embed.append(ex['anchor'])
            texts_to_embed.append(ex['text_a'])
            texts_to_embed.append(ex['text_b'])

        if not texts_to_embed:
            print("Evaluator: 警告！没有可用的验证样本。")
            return self.best_score

        print(f"\nEvaluator: 正在编码 {len(texts_to_embed)} 个验证样本...")

        embeddings = model.encode(
            texts_to_embed,
            show_progress_bar=True,
            batch_size=self.batch_size,
            convert_to_tensor=True
        )

        correct = 0
        total = len(self.examples)

        for i in range(total):
            idx = i * 3
            emb_anchor = embeddings[idx]
            emb_a = embeddings[idx + 1]
            emb_b = embeddings[idx + 2]

            sim_a = cos_sim(emb_anchor, emb_a)
            sim_b = cos_sim(emb_anchor, emb_b)

            model_choice_is_a = sim_a > sim_b
            label_is_a = self.examples[i]['label_is_a']

            if model_choice_is_a == label_is_a:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        print(f"--- [Validation {self.name}] ---")
        print(f"Epoch: {epoch}, Steps: {steps}")
        print(f"Accuracy: {accuracy:.4f} ({correct} / {total})")

        if accuracy > self.best_score:
            self.best_score = accuracy
            if output_path:
                print(f"New best score! Saving model to {output_path}")
                model.save(output_path)

        return accuracy