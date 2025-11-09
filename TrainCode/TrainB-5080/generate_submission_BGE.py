"""
生成 BGE-large-en-v1.5 的 CodaLab 提交文件
"""
import os
import zipfile
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

# --- 配置 ---

# ❗ BGE模型路径 (训练好的或原始的)
# MODEL_PATH = r'E:\model\BGE-large-en-v1.5'  # 原始模型
# 或者用训练好的:
MODEL_PATH = '../../output/track_b_bge_baseline_5080_wsl/checkpoint-2136'  # 训练后的模型

# 考题文件
INPUT_DATA_FILE = '../../TrainingSet1/dev_track_b.jsonl'

# 输出目录
OUTPUT_DIR = '../../submissions/bge_baseline_submission'

# CodaLab要求的文件名
OUTPUT_NPY_FILE = 'track_b.npy'
OUTPUT_ZIP_FILE = 'submission.zip'


def main():
    print(f"🚀 开始生成 BGE-large-en-v1.5 提交文件...")
    print(f"   模型: {MODEL_PATH}")
    print(f"   输入数据: {INPUT_DATA_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 加载BGE模型 (超简单!) ===
    print("正在加载 BGE 模型...")
    try:
        model = SentenceTransformer(MODEL_PATH)
        print("✅ BGE 模型加载成功!")
    except Exception as e:
        print(f"本地加载失败: {e}")
        print("尝试从HuggingFace下载...")
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        print("✅ BGE 模型从HF加载成功!")

    print(f"   Embedding维度: {model.get_sentence_embedding_dimension()}")

    # === 2. 加载考题数据 ===
    print(f"正在加载考题: {INPUT_DATA_FILE}")
    dataset = load_dataset('json', data_files=INPUT_DATA_FILE, split='train')

    sentences_to_encode = []
    for item in dataset:
        text = item.get('text')
        if text is None:
            print("警告：发现空文本行，将编码为空字符串。")
            sentences_to_encode.append("")
        else:
            sentences_to_encode.append(text)

    print(f"已加载 {len(sentences_to_encode)} 行待编码的文本。")

    # === 3. 批量生成嵌入向量 ===
    print("开始批量编码...")
    embeddings = model.encode(
        sentences_to_encode,
        batch_size=128,  # BGE更小,可以用更大batch
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True  # BGE推荐归一化
    )
    print(f"✅ 编码完成，生成了 {embeddings.shape} 形状的 numpy 数组。")

    # === 4. 写入 track_b.npy ===
    output_npy_path = os.path.join(OUTPUT_DIR, OUTPUT_NPY_FILE)
    print(f"正在写入 {output_npy_path} ...")

    np.save(output_npy_path, embeddings)

    print(f"✅ {OUTPUT_NPY_FILE} 写入成功。")

    # === 5. 打包 .zip 文件 ===
    output_zip_path = os.path.join(OUTPUT_DIR, OUTPUT_ZIP_FILE)
    print(f"正在创建 {output_zip_path} ...")

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_npy_path, arcname=OUTPUT_NPY_FILE)

    print(f"🎉 提交文件已生成！")
    print(f"请在 CodaLab 上传这个文件: {output_zip_path}")


if __name__ == "__main__":
    main()