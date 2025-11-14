"""
Track A预测 - 集成BGE和Qwen3-Embedding两个模型
使用加权平均提升性能
"""
import os
import json
import zipfile
from sentence_transformers import SentenceTransformer, models, util
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import BitsAndBytesConfig

# --- 配置 ---

# ❗ BGE模型路径
BGE_MODEL_PATH = '/mnt/e/Code/python/Narrative-Similarity-Task/output/track_b_bge_baseline_5080_wsl/checkpoint-2136'

# ❗ Qwen3模型路径
QWEN_BASE_MODEL = '/mnt/e/model/Qwen3-Embedding-4B'
QWEN_ADAPTER_PATH = '/mnt/e/Code/python/Narrative-Similarity-Task/output/track_b_from_synthetic_5080/checkpoint-356'

# 考题文件
INPUT_DATA_FILE = '/mnt/e/Code/python/Narrative-Similarity-Task/TrainingSet1/dev_track_a.jsonl'

# 输出目录
OUTPUT_DIR = '/mnt/e/Code/python/Narrative-Similarity-Task/submissions/track_a_ensemble_submission'

# CodaLab要求的文件名
OUTPUT_JSONL_FILE = 'track_a.jsonl'
OUTPUT_ZIP_FILE = 'submission.zip'

# ⭐ 集成权重 (可以调整这两个参数)
BGE_WEIGHT = 0.6
QWEN_WEIGHT = 0.4


def load_bge_model(model_path):
    """加载BGE模型"""
    print("🔧 加载 BGE 模型...")
    try:
        model = SentenceTransformer(model_path)
        print(f"✅ BGE 加载成功 (维度: {model.get_sentence_embedding_dimension()})")
        return model
    except Exception as e:
        print(f"⚠️  本地加载失败: {e}")
        print("尝试从HuggingFace下载...")
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        print("✅ BGE 从HF加载成功")
        return model


def load_qwen_model(base_model_path, adapter_path):
    """加载Qwen3-Embedding模型 (QLoRA)"""
    print("🔧 加载 Qwen3-Embedding 模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    word_embedding_model = models.Transformer(
        base_model_path,
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

    model.load_adapter(adapter_path)
    print(f"✅ Qwen3 加载成功 (维度: {embedding_dim})")
    return model


def compute_similarity_scores(bge_model, qwen_model, anchor, text_a, text_b):
    """
    计算集成相似度分数
    返回: (sim_a, sim_b) - anchor与text_a和text_b的加权相似度
    """
    # === 1. BGE相似度 ===
    bge_embeddings = bge_model.encode(
        [anchor, text_a, text_b],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    bge_sim_a = util.cos_sim(bge_embeddings[0], bge_embeddings[1]).item()
    bge_sim_b = util.cos_sim(bge_embeddings[0], bge_embeddings[2]).item()

    # === 2. Qwen3相似度 ===
    qwen_embeddings = qwen_model.encode(
        [anchor, text_a, text_b],
        convert_to_tensor=True,
        show_progress_bar=False
    )

    qwen_sim_a = util.cos_sim(qwen_embeddings[0], qwen_embeddings[1]).item()
    qwen_sim_b = util.cos_sim(qwen_embeddings[0], qwen_embeddings[2]).item()

    # === 3. 加权集成 ===
    ensemble_sim_a = BGE_WEIGHT * bge_sim_a + QWEN_WEIGHT * qwen_sim_a
    ensemble_sim_b = BGE_WEIGHT * bge_sim_b + QWEN_WEIGHT * qwen_sim_b

    return ensemble_sim_a, ensemble_sim_b


def main():
    print(f"🚀 开始生成集成模型 Track A 提交文件...")
    print(f"   集成策略: {BGE_WEIGHT:.1f} × BGE + {QWEN_WEIGHT:.1f} × Qwen3")
    print(f"   输入数据: {INPUT_DATA_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 加载两个模型 ===
    bge_model = load_bge_model(BGE_MODEL_PATH)
    qwen_model = load_qwen_model(QWEN_BASE_MODEL, QWEN_ADAPTER_PATH)

    print("\n✅ 两个模型加载完成!\n")

    # === 2. 加载考题数据 ===
    print(f"正在加载考题: {INPUT_DATA_FILE}")
    dataset = load_dataset('json', data_files=INPUT_DATA_FILE, split='train')
    print(f"已加载 {len(dataset)} 个三元组\n")

    # === 3. 批量预测 ===
    print("开始集成预测...")
    predictions = []

    # 统计单模型正确数 (用于分析)
    bge_correct = 0
    qwen_correct = 0
    ensemble_correct = 0

    for item in tqdm(dataset, desc="Ensemble Predicting"):
        anchor = item.get('anchor_text')
        text_a = item.get('text_a')
        text_b = item.get('text_b')
        label = item.get('text_a_is_closer')  # 真实标签(如果有)

        if not all([anchor, text_a, text_b]):
            print(f"⚠️ 警告: 发现缺失字段的样本")
            predictions.append({
                'anchor_text': anchor or "",
                'text_a': text_a or "",
                'text_b': text_b or "",
                'text_a_is_closer': True
            })
            continue

        # 计算集成相似度
        ensemble_sim_a, ensemble_sim_b = compute_similarity_scores(
            bge_model, qwen_model, anchor, text_a, text_b
        )

        # 预测
        pred = ensemble_sim_a > ensemble_sim_b

        predictions.append({
            'anchor_text': anchor,
            'text_a': text_a,
            'text_b': text_b,
            'text_a_is_closer': bool(pred)
        })

    print(f"\n✅ 预测完成，共 {len(predictions)} 个样本")

    # === 4. 写入 track_a.jsonl ===
    output_jsonl_path = os.path.join(OUTPUT_DIR, OUTPUT_JSONL_FILE)
    print(f"\n正在写入 {output_jsonl_path} ...")

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f"✅ {OUTPUT_JSONL_FILE} 写入成功")

    # === 5. 打包 .zip 文件 ===
    output_zip_path = os.path.join(OUTPUT_DIR, OUTPUT_ZIP_FILE)
    print(f"\n正在创建 {output_zip_path} ...")

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_jsonl_path, arcname=OUTPUT_JSONL_FILE)

    print(f"\n🎉 提交文件已生成！")
    print(f"📁 输出位置: {output_zip_path}")
    print(f"请在 CodaLab 上传这个文件")

    # === 6. 验证预测分布 ===
    true_count = sum(1 for p in predictions if p['text_a_is_closer'])
    false_count = len(predictions) - true_count

    print(f"\n📊 预测分布:")
    print(f"   text_a更接近: {true_count} ({true_count / len(predictions) * 100:.1f}%)")
    print(f"   text_b更接近: {false_count} ({false_count / len(predictions) * 100:.1f}%)")

    # === 7. 显示配置 ===
    print(f"\n⚙️  集成配置:")
    print(f"   BGE权重: {BGE_WEIGHT}")
    print(f"   Qwen3权重: {QWEN_WEIGHT}")
    print(f"   预期提升: 0.66 → 0.67-0.68")


if __name__ == "__main__":
    main()