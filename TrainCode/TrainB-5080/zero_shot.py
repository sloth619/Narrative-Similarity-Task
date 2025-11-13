"""
Track B 零样本 (Zero-Shot) 评估脚本
- 目的: 加载一个未经微调的基础模型, 在 dev_track_a.jsonl 上测试其原始性能。
- 支持: 自动为 Qwen 模型应用 4-bit 量化 (以匹配训练起点)。
"""
import os
import gc
import torch
import time
from sentence_transformers import SentenceTransformer, models
from datasets import load_dataset
from transformers import BitsAndBytesConfig

# 清理显存
torch.cuda.empty_cache()
gc.collect()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. 配置区 ---

# 🔥 在这里选择您想测试的模型
MODEL_TO_TEST = "Qwen3-Embedding-8B"
# MODEL_TO_TEST = "BGE-large-en-v1.5"
# MODEL_TO_TEST = "GTE-large-en-v1.5"
# MODEL_TO_TEST = "Qwen3-Embedding-8B"

# --- 2. 路径配置 (您的 WSL 路径) ---
PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

MODEL_PATHS = {
    "Qwen3-Embedding-4B": '/mnt/e/model/Qwen3-Embedding-4B',
    "Qwen3-Embedding-8B": '/mnt/e/model/Qwen3-Embedding-8B',
    "BGE-large-en-v1.5": '/mnt/e/model/BGE-large-en-v1.5',
    "GTE-large-en-v1.5": '/mnt/e/model/gte-large-en-v1.5',
}

DEV_DATA_PATH = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'
MODEL_PATH = MODEL_PATHS.get(MODEL_TO_TEST)

if MODEL_PATH is None:
    print(f"❌ 错误: 未知的模型名称 '{MODEL_TO_TEST}'。请在 MODEL_PATHS 字典中定义它。")
    exit()


# --- 3. 模型加载 ---

def load_model(model_name, model_path):
    """根据模型名称, 加载标准或 4-bit 量化模型"""
    print(f"\n" + "=" * 60)
    print(f"🔍 正在加载零样本模型: {model_name}")
    print(f"   路径: {model_path}")
    print("=" * 60)

    start_time = time.time()

    if "Qwen" in model_name:
        print("   检测到 Qwen 模型。正在应用 4-bit (QLoRA) 配置...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # 使用 models.Transformer 加载 QLoRA 配置
        word_embedding_model = models.Transformer(
            model_path,
            tokenizer_args={'padding_side': 'left'},
            model_args={
                "quantization_config": bnb_config,
                "device_map": "auto",
                "trust_remote_code": True  # Qwen 必须
            }
        )

        embedding_dim = word_embedding_model.get_word_embedding_dimension()
        pooling_model = models.Pooling(
            word_embedding_dimension=embedding_dim,
            pooling_mode='lasttoken'  # 匹配您训练脚本的池化方式
        )

        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device='cuda'
        )
        print(f"   ✅ 4-bit {model_name} 加载完成。")

    else:
        print("   检测到 BGE/GTE。正在标准加载...")
        model = SentenceTransformer(model_path, device='cuda')
        print(f"   ✅ {model_name} 加载完成。")

    end_time = time.time()
    print(f"   加载耗时: {end_time - start_time:.2f} 秒")
    return model


# --- 4. 评估函数 (来自您的脚本) ---

def evaluate_zero_shot(model, data_path):
    """评估零样本性能"""
    print("\n" + "=" * 60)
    print("📊 开始零样本评估...")
    print("=" * 60)

    try:
        dev_dataset = load_dataset('json', data_files=data_path, split='train')
    except Exception as e:
        print(f"❌ 加载评估文件失败: {data_path}")
        print(f"   错误: {e}")
        return

    correct = 0
    total = 0

    start_time = time.time()
    print(f"开始评估 {len(dev_dataset)} 个三元组...")

    for idx, item in enumerate(dev_dataset):
        anchor = item.get('anchor_text') or item.get('anchor_story')
        text_a = item.get('text_a') or item.get('similar_story')
        text_b = item.get('text_b') or item.get('dissimilar_story')
        label_a_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]) or label_a_closer is None:
            continue

        # 编码
        try:
            embeddings = model.encode(
                [anchor, text_a, text_b],
                show_progress_bar=False,
                batch_size=32  # 评估时使用合理的批次
            )
        except Exception as e:
            print(f"❌ 在第 {idx} 项编码时出错: {e}")
            print(f"   Anchor: {anchor[:50]}...")
            continue

        anchor_emb = embeddings[0]
        text_a_emb = embeddings[1]
        text_b_emb = embeddings[2]

        # 计算余弦相似度
        sim_a = torch.nn.functional.cosine_similarity(
            torch.tensor(anchor_emb).unsqueeze(0),
            torch.tensor(text_a_emb).unsqueeze(0)
        ).item()

        sim_b = torch.nn.functional.cosine_similarity(
            torch.tensor(anchor_emb).unsqueeze(0),
            torch.tensor(text_b_emb).unsqueeze(0)
        ).item()

        # 预测
        prediction = sim_a > sim_b

        if prediction == label_a_closer:
            correct += 1
        total += 1

        # 进度提示
        if (idx + 1) % 50 == 0:
            print(f"  ...已评估: {idx + 1}/{len(dev_dataset)}, 当前准确率: {correct / total:.2%}")

    end_time = time.time()
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("✅ 零样本评估完成!")
    print(f"   模型: {MODEL_TO_TEST}")
    print(f"   准确率: {accuracy:.4f} ({correct}/{total})")
    print(f"   评估耗时: {end_time - start_time:.2f} 秒")
    print("=" * 60)


# --- 5. 执行 ---

def main():
    model = load_model(MODEL_TO_TEST, MODEL_PATH)
    evaluate_zero_shot(model, DEV_DATA_PATH)


if __name__ == "__main__":
    main()