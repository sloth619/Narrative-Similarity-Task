"""
Track A预测 - 使用Qwen3-Instruct-4B (修复解析逻辑)
"""
import os
import json
import zipfile
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- 配置 ---
MODEL_PATH = '/mnt/e/model/Qwen3-4B-Instruct-2507'
INPUT_DATA_FILE = '/mnt/e/Code/python/Narrative-Similarity-Task/TrainingSet1/dev_track_a.jsonl'
OUTPUT_DIR = '/mnt/e/Code/python/Narrative-Similarity-Task/submissions/track_a_qwen_instruct_submission'
OUTPUT_JSONL_FILE = 'track_a.jsonl'
OUTPUT_ZIP_FILE = 'submission.zip'


def load_qwen_instruct_model(model_path):
    """加载Qwen3-Instruct模型 (4bit量化)"""
    print("🔧 加载 Qwen3-Instruct-4B 模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    print(f"✅ Qwen3-Instruct 加载成功")
    return tokenizer, model


def create_prompt(anchor, text_a, text_b):
    """创建简洁的prompt,强制只输出A或B"""
    prompt = f"""You are an expert in narrative analysis. Determine which story is more similar to the anchor story in terms of themes, plot structure, and outcomes.

Anchor Story:
{anchor}

Story A:
{text_a}

Story B:
{text_b}

Question: Which story (A or B) is more similar to the Anchor?
Important: Answer with ONLY the letter A or B, nothing else.

Answer:"""

    return prompt


def predict_with_instruct(tokenizer, model, anchor, text_a, text_b):
    """
    使用Instruct模型进行预测
    修复解析逻辑: 只看第一个字符
    """
    prompt = create_prompt(anchor, text_a, text_b)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer only with A or B."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # 允许更多tokens来生成完整回答
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # 🔧 修复的解析逻辑
    # 1. 首先尝试找到第一个A或B
    first_char = None
    for char in response:
        if char.upper() == 'A':
            first_char = 'A'
            break
        elif char.upper() == 'B':
            first_char = 'B'
            break

    if first_char == 'A':
        return True, response  # text_a更接近
    elif first_char == 'B':
        return False, response  # text_b更接近
    else:
        # 完全无法解析,随机默认
        print(f"⚠️  完全无法解析: '{response[:50]}...', 默认选A")
        return True, response


def main():
    print(f"🚀 开始生成 Qwen3-Instruct Track A 提交文件...")
    print(f"   模型路径: {MODEL_PATH}")
    print(f"   输入数据: {INPUT_DATA_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 加载模型 ===
    tokenizer, model = load_qwen_instruct_model(MODEL_PATH)

    # === 2. 加载考题数据 ===
    print(f"\n正在加载考题: {INPUT_DATA_FILE}")
    dataset = load_dataset('json', data_files=INPUT_DATA_FILE, split='train')
    print(f"已加载 {len(dataset)} 个三元组\n")

    # === 3. 批量预测 ===
    print("开始预测 (修复了解析逻辑)...\n")

    predictions = []
    parse_errors = 0
    response_stats = {'A': 0, 'B': 0, 'error': 0}

    # 保存一些样例用于分析
    sample_responses = []

    for idx, item in enumerate(tqdm(dataset, desc="Predicting with Instruct"), 1):
        anchor = item.get('anchor_text', '')
        text_a = item.get('text_a', '')
        text_b = item.get('text_b', '')

        if not all([anchor, text_a, text_b]):
            predictions.append({
                'anchor_text': anchor,
                'text_a': text_a,
                'text_b': text_b,
                'text_a_is_closer': True
            })
            continue

        try:
            pred, response = predict_with_instruct(tokenizer, model, anchor, text_a, text_b)

            # 统计
            if pred:
                response_stats['A'] += 1
            else:
                response_stats['B'] += 1

            # 保存前5个样例
            if len(sample_responses) < 5:
                sample_responses.append({
                    'idx': idx,
                    'pred': 'A' if pred else 'B',
                    'response': response[:100]
                })

        except Exception as e:
            print(f"⚠️  样本 {idx} 预测出错: {e}, 默认选A")
            pred = True
            parse_errors += 1
            response_stats['error'] += 1

        predictions.append({
            'anchor_text': anchor,
            'text_a': text_a,
            'text_b': text_b,
            'text_a_is_closer': bool(pred)
        })

    print(f"\n✅ 预测完成，共 {len(predictions)} 个样本")
    print(f"\n📊 模型回答统计:")
    print(f"   选择A: {response_stats['A']} ({response_stats['A']/len(predictions)*100:.1f}%)")
    print(f"   选择B: {response_stats['B']} ({response_stats['B']/len(predictions)*100:.1f}%)")
    print(f"   解析错误: {response_stats['error']} ({response_stats['error']/len(predictions)*100:.1f}%)")

    # 显示样例响应
    print(f"\n🔍 模型回答样例:")
    for sample in sample_responses:
        print(f"\n样本 {sample['idx']}: 选择 {sample['pred']}")
        print(f"  回答: {sample['response']}...")

    # === 4. 写入文件 ===
    output_jsonl_path = os.path.join(OUTPUT_DIR, OUTPUT_JSONL_FILE)
    print(f"\n正在写入 {output_jsonl_path} ...")

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f"✅ {OUTPUT_JSONL_FILE} 写入成功")

    # === 5. 打包 ===
    output_zip_path = os.path.join(OUTPUT_DIR, OUTPUT_ZIP_FILE)
    print(f"\n正在创建 {output_zip_path} ...")

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_jsonl_path, arcname=OUTPUT_JSONL_FILE)

    print(f"\n🎉 提交文件已生成！")
    print(f"📁 输出位置: {output_zip_path}")

    # === 6. 最终统计 ===
    true_count = sum(1 for p in predictions if p['text_a_is_closer'])
    false_count = len(predictions) - true_count

    print(f"\n📊 最终预测分布:")
    print(f"   text_a更接近: {true_count} ({true_count / len(predictions) * 100:.1f}%)")
    print(f"   text_b更接近: {false_count} ({false_count / len(predictions) * 100:.1f}%)")


if __name__ == "__main__":
    main()