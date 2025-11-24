"""
Track A 零样本 (Zero-Shot) 评估脚本 - 完整版
支持: Embedding模型 + DeBERTa Multiple Choice
"""
import os
import gc
import torch
import time
import numpy as np
from sentence_transformers import SentenceTransformer, models
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    DebertaV2Tokenizer,
    DebertaV2ForMultipleChoice
)
from sklearn.metrics import accuracy_score

# 清理显存
torch.cuda.empty_cache()
gc.collect()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- 1. 配置区 ---

# 🔥 在这里选择您想测试的模型
MODEL_TO_TEST = "DeBERTa-v3-large"
# MODEL_TO_TEST = "Qwen3-Embedding-4B"
# MODEL_TO_TEST = "BGE-large-en-v1.5"
# MODEL_TO_TEST = "E5-large-v2"
# MODEL_TO_TEST = "jina-embeddings-v3"

# --- 2. 路径配置 ---
PROJECT_ROOT = "/mnt/e/Code/python/Narrative-Similarity-Task"

MODEL_PATHS = {
    # Embedding模型
    "Qwen3-Embedding-4B": '/mnt/e/model/Qwen3-Embedding-4B',
    "Qwen3-Embedding-8B": '/mnt/e/model/Qwen3-Embedding-8B',
    "BGE-large-en-v1.5": '/mnt/e/model/BGE-large-en-v1.5',
    "GTE-large-en-v1.5": '/mnt/e/model/gte-large-en-v1.5',
    "E5-large-v2": '/mnt/e/model/e5-large-v2',
    "jina-embeddings-v3": '/mnt/e/model/jina-embeddings-v3',

    # Multiple Choice模型
    "DeBERTa-v3-large": "microsoft/deberta-v3-large",
    "DeBERTa-v3-base": "microsoft/deberta-v3-base",
    "RoBERTa-large": "roberta-large",
}

DEV_DATA_PATH = f'{PROJECT_ROOT}/TrainingSet1/dev_track_a.jsonl'


# --- 3. 模型加载 ---

def load_embedding_model(model_name, model_path):
    """加载Embedding模型"""
    print(f"\n{'='*60}")
    print(f"🔍 加载Embedding模型: {model_name}")
    print(f"   路径: {model_path}")
    print(f"{'='*60}")

    start_time = time.time()

    if "Qwen" in model_name:
        print("   应用 4-bit 量化...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        word_embedding_model = models.Transformer(
            model_path,
            tokenizer_args={'padding_side': 'left'},
            model_args={
                "quantization_config": bnb_config,
                "device_map": "auto",
                "trust_remote_code": True
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
    elif "jina" in model_name.lower() or "GTE" in model_name or "gte" in model_name.lower():
        model = SentenceTransformer(
            model_path,
            device='cuda',
            trust_remote_code=True
        )
    else:
        model = SentenceTransformer(model_path, device='cuda')

    end_time = time.time()
    print(f"   ✅ 加载完成 ({end_time - start_time:.2f}秒)")
    return model


def load_multiple_choice_model(model_name, model_path):
    """加载Multiple Choice模型 (DeBERTa等)"""
    print(f"\n{'='*60}")
    print(f"🔍 加载Multiple Choice模型: {model_name}")
    print(f"   路径: {model_path}")
    print(f"{'='*60}")

    start_time = time.time()

    # 加载tokenizer和模型
    if "DeBERTa" in model_name or "deberta" in model_name.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        model = DebertaV2ForMultipleChoice.from_pretrained(model_path)
    else:
        from transformers import AutoTokenizer, AutoModelForMultipleChoice
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForMultipleChoice.from_pretrained(model_path)

    model = model.to('cuda')
    model.eval()

    end_time = time.time()
    print(f"   ✅ 加载完成 ({end_time - start_time:.2f}秒)")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    return tokenizer, model


# --- 4. Embedding模型评估 ---

def evaluate_embedding_model(model, data_path, model_name):
    """评估Embedding模型的零样本性能"""
    print(f"\n{'='*60}")
    print("📊 Embedding模型零样本评估")
    print(f"{'='*60}")

    dataset = load_dataset('json', data_files=data_path, split='train')
    correct = 0
    total = 0

    start_time = time.time()

    for idx, item in enumerate(dataset):
        anchor = item.get('anchor_text') or item.get('anchor_story')
        text_a = item.get('text_a') or item.get('similar_story')
        text_b = item.get('text_b') or item.get('dissimilar_story')
        label = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b]) or label is None:
            continue

        try:
            embeddings = model.encode(
                [anchor, text_a, text_b],
                show_progress_bar=False,
                batch_size=32
            )

            anchor_emb = embeddings[0]
            text_a_emb = embeddings[1]
            text_b_emb = embeddings[2]

            sim_a = torch.nn.functional.cosine_similarity(
                torch.tensor(anchor_emb).unsqueeze(0),
                torch.tensor(text_a_emb).unsqueeze(0)
            ).item()

            sim_b = torch.nn.functional.cosine_similarity(
                torch.tensor(anchor_emb).unsqueeze(0),
                torch.tensor(text_b_emb).unsqueeze(0)
            ).item()

            prediction = sim_a > sim_b

            if prediction == label:
                correct += 1
            total += 1

            if (idx + 1) % 50 == 0:
                print(f"   进度: {idx + 1}/{len(dataset)}, 当前准确率: {correct/total:.2%}")

        except Exception as e:
            print(f"   ⚠️  样本{idx}处理失败: {e}")
            continue

    end_time = time.time()
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print("✅ 评估完成!")
    print(f"   模型: {model_name}")
    print(f"   准确率: {accuracy:.4f} ({correct}/{total})")
    print(f"   耗时: {end_time - start_time:.2f}秒")
    print(f"{'='*60}")

    return accuracy


# --- 5. Multiple Choice模型评估 ---

def evaluate_multiple_choice_model(tokenizer, model, data_path, model_name):
    """评估Multiple Choice模型的零样本性能"""
    print(f"\n{'='*60}")
    print("📊 Multiple Choice模型零样本评估")
    print(f"{'='*60}")

    dataset = load_dataset('json', data_files=data_path, split='train')
    predictions = []
    labels = []

    start_time = time.time()

    with torch.no_grad():
        for idx, item in enumerate(dataset):
            anchor = item.get('anchor_text') or item.get('anchor_story')
            text_a = item.get('text_a') or item.get('similar_story')
            text_b = item.get('text_b') or item.get('dissimilar_story')
            label = item.get('text_a_is_closer')

            if not all([anchor, text_a, text_b]) or label is None:
                continue

            try:
                # Tokenize两个选择
                inputs = tokenizer(
                    [anchor, anchor],  # 两次anchor
                    [text_a, text_b],  # 两个选项
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )

                # 移到GPU
                inputs = {k: v.unsqueeze(0).to('cuda') for k, v in inputs.items()}

                # 推理
                outputs = model(**inputs)
                logits = outputs.logits  # [1, 2]

                # 预测 (0=A, 1=B)
                pred = torch.argmax(logits, dim=-1).item()
                pred_bool = (pred == 0)  # True if A, False if B

                predictions.append(pred_bool)
                labels.append(label)

                if (idx + 1) % 50 == 0:
                    acc = accuracy_score(labels, predictions)
                    print(f"   进度: {idx + 1}/{len(dataset)}, 当前准确率: {acc:.2%}")

            except Exception as e:
                print(f"   ⚠️  样本{idx}处理失败: {e}")
                continue

    end_time = time.time()
    accuracy = accuracy_score(labels, predictions)

    print(f"\n{'='*60}")
    print("✅ 评估完成!")
    print(f"   模型: {model_name}")
    print(f"   准确率: {accuracy:.4f} ({sum(np.array(predictions) == np.array(labels))}/{len(labels)})")
    print(f"   耗时: {end_time - start_time:.2f}秒")
    print(f"{'='*60}")

    return accuracy


# --- 6. 主函数 ---

def main():
    MODEL_TO_TEST = "DeBERTa-v3-large"  # 在这里修改要测试的模型

    model_path = MODEL_PATHS.get(MODEL_TO_TEST)

    if model_path is None:
        print(f"❌ 未知模型: {MODEL_TO_TEST}")
        return

    # 判断模型类型
    if any(x in MODEL_TO_TEST for x in ["DeBERTa", "RoBERTa", "deberta", "roberta"]):
        # Multiple Choice模型
        tokenizer, model = load_multiple_choice_model(MODEL_TO_TEST, model_path)
        accuracy = evaluate_multiple_choice_model(
            tokenizer, model, DEV_DATA_PATH, MODEL_TO_TEST
        )
    else:
        # Embedding模型
        model = load_embedding_model(MODEL_TO_TEST, model_path)
        accuracy = evaluate_embedding_model(
            model, DEV_DATA_PATH, MODEL_TO_TEST
        )

    # 最终总结
    print(f"\n{'='*60}")
    print("📊 最终结果总结")
    print(f"{'='*60}")
    print(f"   模型: {MODEL_TO_TEST}")
    print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   对比:")
    print(f"      E5-large:     67.00%")
    print(f"      Gemini Pro:   71.00%")
    print(f"      当前模型:     {accuracy*100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()