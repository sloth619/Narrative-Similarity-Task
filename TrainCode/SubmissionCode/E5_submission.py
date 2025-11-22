"""
生成CodaBench提交文件
"""
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import zipfile


class CodaBenchSubmissionGenerator:

    def __init__(self, model_path: str = '/mnt/e/model/e5-large-v2'):
        self.model = SentenceTransformer(model_path)
        self.prompt = "passage: "
        self.normalize = True

    def predict_track_a(self, anchor: str, text_a: str, text_b: str) -> bool:
        texts = [self.prompt + t for t in [anchor, text_a, text_b]]
        embeddings = self.model.encode(texts, normalize_embeddings=self.normalize, show_progress_bar=False)
        sim_a = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        sim_b = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        return bool(sim_a > sim_b)  # 转换为Python bool

    def generate_track_b_embedding(self, text: str) -> np.ndarray:
        prompted_text = self.prompt + text
        embedding = self.model.encode([prompted_text], normalize_embeddings=self.normalize, show_progress_bar=False)[0]
        return embedding

    def generate_track_a_jsonl(self, input_path: str, output_path: str):
        dataset = load_dataset('json', data_files=input_path, split='train')

        with open(output_path, 'w') as f:
            for item in dataset:
                anchor = item.get('anchor_text') or item.get('anchor_story')
                text_a = item.get('text_a') or item.get('similar_story')
                text_b = item.get('text_b') or item.get('dissimilar_story')

                if not all([anchor, text_a, text_b]):
                    continue

                text_a_is_closer = self.predict_track_a(anchor, text_a, text_b)
                f.write(json.dumps({"text_a_is_closer": text_a_is_closer}) + '\n')

    def generate_track_b_npy(self, input_path: str, output_path: str):
        dataset = load_dataset('json', data_files=input_path, split='train')

        embeddings_list = []
        for item in dataset:
            text = item.get('text')
            if not text:
                continue
            embedding = self.generate_track_b_embedding(text)
            embeddings_list.append(embedding)

        embeddings_array = np.array(embeddings_list)
        np.save(output_path, embeddings_array)


def create_submission_zip(generator, dev_track_a_path: str, dev_track_b_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    track_a_path = os.path.join(output_dir, 'track_a.jsonl')
    track_b_path = os.path.join(output_dir, 'track_b.npy')
    zip_path = os.path.join(output_dir, 'submission.zip')

    generator.generate_track_a_jsonl(dev_track_a_path, track_a_path)
    generator.generate_track_b_npy(dev_track_b_path, track_b_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(track_a_path, arcname='track_a.jsonl')
        zipf.write(track_b_path, arcname='track_b.npy')

    return zip_path


def main():
    PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"
    OUTPUT_DIR = f"{PROJECT_ROOT}/codabench_submission"

    dev_track_a_path = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'
    dev_track_b_path = f'{PROJECT_ROOT}/TrainingSet1/dev_track_b.jsonl'

    generator = CodaBenchSubmissionGenerator()

    zip_path = create_submission_zip(
        generator=generator,
        dev_track_a_path=dev_track_a_path,
        dev_track_b_path=dev_track_b_path,
        output_dir=OUTPUT_DIR
    )

    print(f"提交文件已生成: {zip_path}")


if __name__ == "__main__":
    main()