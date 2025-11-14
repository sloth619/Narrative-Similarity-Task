"""
Track A预测 - 使用BGE-large-en-v1.5模型
"""
import os
import json
import zipfile
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from tqdm import tqdm

# --- 配置 ---

# ❗ BGE模型路径
MODEL_PATH = '/mnt/e/Code/python/Narrative-Similarity-Task/output/track_b_bge_optimized_5080/checkpoint-3840'

# 考题文件 (CodaLab 开发集)
INPUT_DATA_FILE = '/mnt/e/Code/python/Narrative-Similarity-Task/TrainingSet1/dev_track_a.jsonl'

# 输出目录
OUTPUT_DIR = '/mnt/e/Code/python/Narrative-Similarity-Task/submissions/track_a_bge_submission'

# CodaLab要求的文件名
OUTPUT_JSONL_FILE = 'track_a.jsonl'
OUTPUT_ZIP_FILE = 'submission.zip'


def main():
    print(f"🚀 开始生成 BGE Track A 提交文件...")
    print(f"   模型路径: {MODEL_PATH}")
    print(f"   输入数据: {INPUT_DATA_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 加载BGE模型 ===
    print("正在加载 BGE 模型...")
    try:
        model = SentenceTransformer(MODEL_PATH)
        print("✅ BGE 模型加载成功 (从本地checkpoint)")
    except Exception as e:
        print(f"本地加载失败: {e}")
        print("尝试从HuggingFace下载原始模型...")
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        print("✅ BGE 模型从HF加载成功")

    print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")

    # === 2. 加载考题数据 ===
    print(f"正在加载考题: {INPUT_DATA_FILE}")
    dataset = load_dataset('json', data_files=INPUT_DATA_FILE, split='train')
    print(f"已加载 {len(dataset)} 个三元组")

    # === 3. 批量预测 ===
    print("开始预测...")
    predictions = []

    for item in tqdm(dataset, desc="Predicting"):
        anchor = item.get('anchor_text')
        text_a = item.get('text_a')
        text_b = item.get('text_b')

        if not all([anchor, text_a, text_b]):
            print(f"⚠️ 警告: 发现缺失字段的样本")
            # 即使缺失,也要添加一个预测以保持顺序
            predictions.append({
                'anchor_text': anchor or "",
                'text_a': text_a or "",
                'text_b': text_b or "",
                'text_a_is_closer': True  # 默认预测
            })
            continue

        # 编码三个文本
        embeddings = model.encode(
            [anchor, text_a, text_b],
            convert_to_tensor=True,
            normalize_embeddings=True,  # BGE推荐归一化
            show_progress_bar=False
        )

        # 计算余弦相似度
        sim_a = util.cos_sim(embeddings[0], embeddings[1]).item()
        sim_b = util.cos_sim(embeddings[0], embeddings[2]).item()

        # 预测: text_a相似度更高则为True
        pred = sim_a > sim_b

        predictions.append({
            'anchor_text': anchor,
            'text_a': text_a,
            'text_b': text_b,
            'text_a_is_closer': bool(pred)  # 确保是bool类型
        })

    print(f"✅ 预测完成，共 {len(predictions)} 个样本")

    # === 4. 写入 track_a.jsonl ===
    output_jsonl_path = os.path.join(OUTPUT_DIR, OUTPUT_JSONL_FILE)
    print(f"正在写入 {output_jsonl_path} ...")

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f"✅ {OUTPUT_JSONL_FILE} 写入成功")

    # === 5. 打包 .zip 文件 ===
    output_zip_path = os.path.join(OUTPUT_DIR, OUTPUT_ZIP_FILE)
    print(f"正在创建 {output_zip_path} ...")

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_jsonl_path, arcname=OUTPUT_JSONL_FILE)

    print(f"🎉 提交文件已生成！")
    print(f"📁 输出位置: {output_zip_path}")
    print(f"请在 CodaLab 上传这个文件: {output_zip_path}")

    # === 6. 验证预测分布 ===
    true_count = sum(1 for p in predictions if p['text_a_is_closer'])
    false_count = len(predictions) - true_count
    print(f"\n📊 预测分布:")
    print(f"   text_a更接近: {true_count} ({true_count / len(predictions) * 100:.1f}%)")
    print(f"   text_b更接近: {false_count} ({false_count / len(predictions) * 100:.1f}%)")


if __name__ == "__main__":
    main()