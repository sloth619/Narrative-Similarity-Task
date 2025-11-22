"""
Qwen3-Embedding-4B 提示词测试 - 修复版
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import torch


class Qwen3PromptTester:

    def __init__(self, model_path: str = '/mnt/e/model/Qwen3-Embedding-4B'):
        print(f"加载模型: {model_path}")

        # 先定义prompts (在任何分支之前)
        self.prompts = {
            'none': "",
            'query': "query: ",
            'passage': "passage: ",
        }

        # 方法1: 尝试直接加载
        try:
            self.model = SentenceTransformer(
                model_path,
                trust_remote_code=True,
                device='cuda'
            )
            print("✅ 使用SentenceTransformer加载成功\n")
            self.use_hf = False

        except Exception as e:
            print(f"SentenceTransformer加载失败: {e}")
            print("尝试使用transformers加载...\n")

            # 方法2: 使用transformers
            from transformers import AutoTokenizer, AutoModel

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model_hf = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.float16,
                device_map='cuda'
            )
            self.model_hf.eval()
            self.use_hf = True
            print("✅ 使用Transformers加载成功\n")

    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_hf(self, texts, prompt=""):
        """使用transformers编码"""
        prompted_texts = [prompt + t for t in texts]

        inputs = self.tokenizer(
            prompted_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to('cuda')

        with torch.no_grad():
            outputs = self.model_hf(**inputs)
            embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def encode(self, texts, prompt=""):
        """统一的编码接口"""
        if self.use_hf:
            return self.encode_hf(texts, prompt)
        else:
            prompted_texts = [prompt + t for t in texts]
            return self.model.encode(
                prompted_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )

    def predict_single(self, anchor: str, text_a: str, text_b: str, prompt_key: str) -> bool:
        prompt = self.prompts[prompt_key]
        embeddings = self.encode([anchor, text_a, text_b], prompt)

        sim_a = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        sim_b = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]

        return bool(sim_a > sim_b)

    def test_all_prompts(self, data_path: str):
        print("=" * 70)
        print("Qwen3-Embedding-4B Prompt测试")
        print("=" * 70)

        dataset = load_dataset('json', data_files=data_path, split='train')

        clean_data = []
        for item in dataset:
            anchor = item.get('anchor_text')
            text_a = item.get('text_a')
            text_b = item.get('text_b')
            label = item.get('text_a_is_closer')

            if all([anchor, text_a, text_b, label is not None]):
                clean_data.append({
                    'anchor': anchor,
                    'text_a': text_a,
                    'text_b': text_b,
                    'label': label
                })

        print(f"测试样本: {len(clean_data)}\n")

        results = {}

        for prompt_key in self.prompts.keys():
            print(f"测试: {prompt_key}")

            correct = 0
            for i, sample in enumerate(clean_data, 1):
                pred = self.predict_single(
                    sample['anchor'],
                    sample['text_a'],
                    sample['text_b'],
                    prompt_key
                )

                if pred == sample['label']:
                    correct += 1

                if i % 50 == 0:
                    print(f"  进度: {i}/{len(clean_data)}, 准确率: {correct/i:.4f}")

            accuracy = correct / len(clean_data)
            results[prompt_key] = accuracy

            print(f"✅ '{prompt_key}': {accuracy:.4f} ({correct}/{len(clean_data)})\n")

        print("\n" + "=" * 70)
        print("所有结果排序")
        print("=" * 70)

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        for i, (prompt_key, acc) in enumerate(sorted_results, 1):
            marker = "🏆" if i == 1 else f"{i:2d}."
            print(f"{marker} {prompt_key:25s}: {acc:.4f} ({acc*100:.2f}%)")

        best_prompt = sorted_results[0][0]
        best_acc = sorted_results[0][1]

        print("\n" + "=" * 70)
        print(f"最佳配置: {best_prompt}")
        print(f"准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"Zero-shot参考: 62.5%")
        print("=" * 70)

        return best_prompt, results


def main():
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"
    dev_track_a_path = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'

    tester = Qwen3PromptTester()
    best_prompt, results = tester.test_all_prompts(dev_track_a_path)

    print(f"\n推荐使用: {best_prompt}")


if __name__ == "__main__":
    main()