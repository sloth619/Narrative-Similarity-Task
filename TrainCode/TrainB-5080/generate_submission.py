import os
# (移除了在 Windows 上无效的 PYTORCH_CUDA_ALLOC_CONF)

import json
import zipfile
import numpy as np  # 导入 numpy 用于保存 .npy
from sentence_transformers import SentenceTransformer, models
from datasets import load_dataset
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType
from tqdm import tqdm

# --- 1. 配置你的提交 ---

# ❗ 基础模型 (你在 E 盘上的路径)
BASE_MODEL_PATH = 'E:/model/Qwen3-Embedding-4B'

# ❗ 适配器路径 (你刚刚训练好的增强版模型)
ADAPTER_PATH = '../../output/track_b_baseline_model_v2_qlora_5080/best_lora_adapter'

# 考题文件 (CodaLab 开发集)
INPUT_DATA_FILE = '../../TrainingSet1/dev_track_b.jsonl'

# 输出目录 (我们会在这里创建 track_b.npy 和 submission.zip)
OUTPUT_DIR = '../../submissions/augmented_v2_5080_submission'  # (新文件夹)

# --- 2. CodaLab 要求的文件名 (已修复) ---
OUTPUT_NPY_FILE = 'track_b.npy'  # 目标文件是 .npy
OUTPUT_ZIP_FILE = 'submission.zip'


def main():
    print(f"🚀 开始生成 CodaLab 提交文件 (.npy 格式)...")
    print(f"   基础模型: {BASE_MODEL_PATH}")
    print(f"   适配器 (LoRA): {ADAPTER_PATH}")
    print(f"   输入数据: {INPUT_DATA_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 加载 QLoRA 模型 (和训练时一样) ===
    # 我们必须先加载 4-bit 的基础模型，然后再把 LoRA 适配器“插”上去

    print("正在加载 4-bit 基础模型...")
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

    # [关键] 加载我们训练好的 LoRA 适配器
    print(f"正在加载 LoRA 适配器...")
    model.load_adapter(ADAPTER_PATH)
    print("✅ QLoRA 模型加载成功！")

    # === 2. 加载考题数据 ===
    print(f"正在加载考题: {INPUT_DATA_FILE}")
    dataset = load_dataset('json', data_files=INPUT_DATA_FILE, split='train')

    # (重要!) 必须保持原始顺序，不能过滤
    sentences_to_encode = []
    for item in dataset:
        text = item.get('text')
        if text is None:
            # 如果 CodaLab 的考题有空行，我们也必须为它生成一个“空”向量
            print("警告：发现一个空文本行，将编码为空字符串。")
            sentences_to_encode.append("")
        else:
            sentences_to_encode.append(text)

    print(f"已加载 {len(sentences_to_encode)} 行待编码的文本。")

    # === 3. 批量生成嵌入向量 ===
    print("开始批量编码 (推理)...")
    # 推理时可以使用大批次，你的 16GB 显存足够
    embeddings = model.encode(
        sentences_to_encode,
        batch_size=64,  # 推理时 batch_size 可以大一点
        show_progress_bar=True,
        convert_to_tensor=False  # 直接转为 numpy array
    )
    print(f"✅ 编码完成，生成了 {embeddings.shape} 形状的 numpy 数组。")

    # === 4. 写入 track_b.npy ===
    output_npy_path = os.path.join(OUTPUT_DIR, OUTPUT_NPY_FILE)
    print(f"正在写入 {output_npy_path} ...")

    np.save(output_npy_path, embeddings)  # 使用 np.save

    print(f"✅ {OUTPUT_NPY_FILE} 写入成功。")

    # === 5. 打包 .zip 文件 ===
    output_zip_path = os.path.join(OUTPUT_DIR, OUTPUT_ZIP_FILE)
    print(f"正在创建 {output_zip_path} ...")

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 关键: arcname=OUTPUT_NPY_FILE 确保文件在 zip 的根目录
        zf.write(output_npy_path, arcname=OUTPUT_NPY_FILE)

    print(f"🎉 提交文件已生成！")
    print(f"请在 CodaLab 上传这个文件: {output_zip_path}")


if __name__ == "__main__":
    main()