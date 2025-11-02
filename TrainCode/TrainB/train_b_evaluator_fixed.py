import json
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import cos_sim
from datasets import load_dataset
import torch


class TrackB_Accuracy_Evaluator_NoSave(SentenceEvaluator):
    """不保存模型的评估器，避免卡住"""

    def __init__(self, name: str, data_path: str, batch_size: int = 32):
        self.name = name
        self.batch_size = batch_size
        dataset = load_dataset('json', data_files=data_path, split='train')
        self.examples = []

        print(f"Evaluator: 正在加载并清洗 {data_path}...")
        for item in dataset:
            anchor = item.get('anchor_text')
            text_a = item.get('text_a')
            text_b = item.get('text_b')
            label = item.get('text_a_is_closer')

            if all([anchor, text_a, text_b]) and label is not None:
                self.examples.append({
                    'anchor': anchor,
                    'text_a': text_a,
                    'text_b': text_b,
                    'label_is_a': label
                })
        print(f"Evaluator: 加载了 {len(self.examples)} 个干净的验证样本。")
        self.best_score = -1.0

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        texts_to_embed = []
        for ex in self.examples:
            texts_to_embed.append(ex['anchor'])
            texts_to_embed.append(ex['text_a'])
            texts_to_embed.append(ex['text_b'])

        if not texts_to_embed:
            return self.best_score

        embeddings = model.encode(
            texts_to_embed,
            show_progress_bar=False,  # 关闭进度条，减少输出
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

        print(f"\n[Validation {self.name}] Epoch: {epoch:.1f}, Steps: {steps}")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})", end="")

        if accuracy > self.best_score:
            self.best_score = accuracy
            print(" ⭐ New best!")
        else:
            print()

        return accuracy