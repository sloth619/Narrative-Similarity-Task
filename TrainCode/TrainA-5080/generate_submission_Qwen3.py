"""
Track A提交文件生成脚本
使用训练好的Qwen3-Reranker-4B模型生成predictions.jsonl
"""
import os
import json
import zipfile
from sentence_transformers import CrossEncoder
from datasets import load_dataset
from tqdm import tqdm
import torch

# --- 配置 ---

# ❗ 模型路径 (训练好的模型)
MODEL_PATH = '/mnt/e/Code/python/Narrative-Similarity-Task/output/track_a_trainer_4bit/checkpoint-238'

# 考题文件 (CodaLab 开发集)
INPUT_DATA_FILE = '/mnt/e/Code/python/Narrative-Similarity-Task//TrainingSet1/dev_track_a.jsonl'

# 输出目录
OUTPUT_DIR = '/mnt/e/Code/python/Narrative-Similarity-Task//submissions/track_a_submission'

# CodaLab 要求的文件名
OUTPUT_JSONL_FILE = 'track_a.jsonl'
OUTPUT_ZIP_FILE = 'submission.zip'


def main():
    print(f"🚀 开始生成 Track A CodaLab 提交文件...")
    print(f"   模型路径: {MODEL_PATH}")
    print(f"   输入数据: {INPUT_DATA_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 加载模型 ===
    print("正在加载 Reranker 模型...")
    model = CrossEncoder(
        MODEL_PATH,
        num_labels=1,
        max_length=512,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"✅ 模型加载成功 (设备: {model.device})")

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
            print(f"⚠️ 警告: 发现缺失字段的样本,跳过")
            # 即使缺失,也要添加一个预测以保持顺序
            predictions.append({
                'anchor_text': anchor or "",
                'text_a': text_a or "",
                'text_b': text_b or "",
                'text_a_is_closer': True  # 默认预测
            })
            continue

        # 计算两个分数
        score_a = model.predict([[anchor, text_a]])[0]
        score_b = model.predict([[anchor, text_b]])[0]

        # 预测: text_a分数更高则为True
        pred = score_a > score_b

        predictions.append({
            'anchor_text': anchor,
            'text_a': text_a,
            'text_b': text_b,
            'text_a_is_closer': bool(pred)  # 确保是bool类型
        })

    print(f"✅ 预测完成，共 {len(predictions)} 个样本")

    # === 4. 写入 predictions.jsonl ===
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
        # 关键: arcname确保文件在zip的根目录
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