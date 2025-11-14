"""
Track A预测 - 使用Track B训练好的Embedding模型
"""
import json
import zipfile
from sentence_transformers import SentenceTransformer, models, util
from datasets import load_dataset
import torch
from transformers import BitsAndBytesConfig
from tqdm import tqdm

# 配置
BASE_MODEL_PATH = '/mnt/e/model/Qwen3-Embedding-4B'
ADAPTER_PATH = '/mnt/e/Code/python/Narrative-Similarity-Task/output/track_b_from_synthetic_5080/checkpoint-356'
INPUT_DATA_FILE = '/mnt/e/Code/python/Narrative-Similarity-Task/TrainingSet1/dev_track_a.jsonl'
OUTPUT_DIR = '/mnt/e/Code/python/Narrative-Similarity-Task/submissions/track_a_from_embedding'


def main():
    print("🚀 使用Track B的Embedding模型做Track A预测")

    # 加载Track B训练好的模型
    print("加载模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    word_embedding_model = models.Transformer(
        BASE_MODEL_PATH,
        tokenizer_args={'padding_side': 'left'},
        model_args={
            "quantization_config": bnb_config,
            "device_map": "auto",
        }
    )

    embedding_dim = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(
        word_embedding_dimension=embedding_dim,
        pooling_mode='lasttoken'
    )

    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device='cuda'
    )

    model.load_adapter(ADAPTER_PATH)
    print("✅ 模型加载完成")

    # 加载数据
    dataset = load_dataset('json', data_files=INPUT_DATA_FILE, split='train')
    print(f"加载了 {len(dataset)} 个三元组")

    # 预测
    predictions = []
    for item in tqdm(dataset, desc="Predicting"):
        anchor = item['anchor_text']
        text_a = item['text_a']
        text_b = item['text_b']

        # 编码
        embeddings = model.encode(
            [anchor, text_a, text_b],
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # 计算余弦相似度
        sim_a = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_b = util.cos_sim(embeddings[0], embeddings[2]).item()

        # 预测
        pred = sim_a > sim_b

        predictions.append({
            'anchor_text': anchor,
            'text_a': text_a,
            'text_b': text_b,
            'text_a_is_closer': bool(pred)
        })

    # 保存
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_jsonl = os.path.join(OUTPUT_DIR, 'predictions.jsonl')
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    # 打包
    output_zip = os.path.join(OUTPUT_DIR, 'submission.zip')
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_jsonl, arcname='track_a.jsonl')

    print(f"✅ 完成! 提交文件: {output_zip}")

    # 统计
    true_count = sum(1 for p in predictions if p['text_a_is_closer'])
    print(f"\n预测分布:")
    print(f"  text_a更接近: {true_count} ({true_count / len(predictions) * 100:.1f}%)")
    print(
        f"  text_b更接近: {len(predictions) - true_count} ({(len(predictions) - true_count) / len(predictions) * 100:.1f}%)")


if __name__ == "__main__":
    main()