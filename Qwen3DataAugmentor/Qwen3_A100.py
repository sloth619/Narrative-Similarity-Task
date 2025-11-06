import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import torch
from datetime import datetime
import time

from unsloth import FastLanguageModel

class Qwen3DataAugmentorOptimized:
    """V3.4 最终修复版 (英文提示词 + 正确采样)"""

    def __init__(
        self,
        model_name: str = "/root/autodl-tmp/Qwen3-4B-Instruct-2507",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,  # 默认使用 4-bit
        dtype=None,
        device: str = "auto",
        checkpoint_dir: str = "./checkpoints"
    ):
        print(f"正在加载模型: {model_name} (4-bit: {load_in_4bit})")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=dtype,
        )

        FastLanguageModel.for_inference(self.model)

        # ✅ 修复 1: 设置 pad token 和 padding_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ✅ 关键修复：设置左填充
        self.tokenizer.padding_side = 'left'

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # <<< [FIXED] 提示词已全部改为英文 >>>

        # 缓存系统消息 (改为英文)
        self.system_message = {"role": "system", "content": "You are a professional data augmentation assistant, skilled at creating similar but not identical text samples."}

        # 策略1：生成“负样本” (改为英文)
        self.track_a_negative_gen_template = """You are an expert story reviewer. Your task is to create a "negative sample".

Please refer to the following two similar stories:

Anchor Story:
{anchor}

Positive Story (more similar to the Anchor Story):
{positive}

Requirement: {instruction}

Please create a **new negative story**. This new story should:
1. Be thematically related to the "Anchor Story".
2. Be clearly **less** similar to the "Anchor Story" than the "Positive Story" is.
3. Be completely different from the "Positive Story".

Please output **only the text content** of your new negative story, without any other explanatory text or JSON."""

        # 策略2：生成“正样本” (改为英文)
        self.track_a_positive_gen_template = """You are an expert story reviewer. Your task is to create a "positive sample".

Please refer to the following two stories:

Anchor Story:
{anchor}

Negative Story (not very similar to the Anchor Story):
{negative}

Requirement: {instruction}

Please create a **new positive story**. This new story should:
1. Be **highly similar** in plot and theme to the "Anchor Story".
2. Be clearly **more** similar to the "Anchor Story" than the "Negative Story" is.
3. Be completely different from the "Negative Story".

Please output **only the text content** of your new positive story, without any other explanatory text or JSON."""

        # Track B 提示词 (改为英文)
        self.track_b_template = """You are an expert story reviewer. Please create a new, similar story based on the following story.

Original Story:
{text}

Requirement: {instruction}

Please output **only the text content** of the new story, without any other explanatory text."""

        # 多样性指令 (改为英文)
        self.diversity_instructions = [
            "Maintain a similar theme and structure, but use different details and wording.",
            "Keep the core plot, but change the story's setting, time period, or cultural context.",
            "Maintain the story's theme, but retell it from a completely different angle or character's perspective."
        ]
        # <<< 英文提示词修复结束 >>>

        print("模型加载完成")
        print(f"Padding side: {self.tokenizer.padding_side}") # 验证 Padding 修复

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """加载 JSONLines 文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def save_checkpoint(self, checkpoint_path: Union[Path, str], data: List[Dict[str, Any]],
                           current_index: int, current_round: int, metadata: Dict[str, Any]):
        """保存检查点"""
        checkpoint = {
            'data': data,
            'current_index': current_index,
            'current_round': current_round,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, checkpoint_path: Union[Path, str]) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            return None
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            print(f"✓ 已加载检查点: 已有 {len(checkpoint['data'])} 个样本")
            return checkpoint
        except Exception as e:
            print(f"⚠ 加载检查点失败: {e}")
            return None

    def create_prompt_track_b(self, item: Dict[str, Any], diversity_level: int = 0) -> str:
        """快速创建 Track B 提示词"""
        return self.track_b_template.format(
            text=item.get('text', ''),
            instruction=self.diversity_instructions[min(diversity_level, 2)]
        )

    # <<< [FIXED] 修复采样参数 >>>
    def generate_text_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        # 修正为 Qwen3-4B 官方推荐参数
        temperature: float = 0.7, # 从 0.8 降为 0.7
        top_p: float = 0.8,       # 从 0.9 降为 0.8
        top_k: int = 20,        # 从 50 降为 20
        # 增加重复惩罚
        repetition_penalty: float = 1.1
    ) -> List[str]:
        """优化的批处理生成 (已修复重复问题)"""

        # 1. 批量准备聊天模板
        messages_batch = [
            [self.system_message, {"role": "user", "content": p}]
            for p in prompts
        ]

        # 2. 转换为文本格式
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_batch
        ]

        # 3. 批量 tokenize（左填充）
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        ).to(self.model.device)

        # 4. 批量生成
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty, # 应用重复惩罚
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # 5. 批量解码
        input_ids_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_ids_len:]
        generated_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return [text.strip() for text in generated_texts]
    # <<< 采样参数修复结束 >>>

    def augment_track_a_batch(
        self,
        data: List[Dict[str, Any]],
        batch_size: int = 8,  # 默认小批处理
        target_total: int = 10000,
        checkpoint_interval: int = 500,
        checkpoint_name: str = "track_a_checkpoint.json",
        resume: bool = True,
        include_original: bool = True
    ) -> List[Dict[str, Any]]:
        """Track A 数据增强 - 批处理优化"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = None
        if resume:
            checkpoint = self.load_checkpoint(checkpoint_path)

        if checkpoint:
            augmented_data = checkpoint['data']
            start_index = checkpoint['current_index']
            start_round = checkpoint['current_round']
            metadata = checkpoint['metadata']
        else:
            augmented_data = []
            start_index = 0
            start_round = 0

            if include_original:
                augmented_data.extend(data)
                print(f"已添加 {len(data)} 个原始样本")

            original_count = len(data) if include_original else 0
            needed_augmentations = target_total - original_count
            if len(data) == 0:
                print("⚠ 警告: 原始数据为空，无法进行增强。")
                return []
            augmentations_per_sample = needed_augmentations // len(data) if len(data) > 0 else 0
            remainder = needed_augmentations % len(data) if len(data) > 0 else 0
            metadata = {
                'target_total': target_total,
                'original_count': len(data),
                'augmentations_per_sample': augmentations_per_sample,
                'remainder': remainder,
                'include_original': include_original
            }
            print(f"\n=== 数据增强计划 ===")
            print(f"目标样本数: {target_total}")
            print(f"需要生成: {needed_augmentations} 个")
            print(f"每个样本生成: {augmentations_per_sample} 个")
            print(f"批处理大小: {batch_size}")

        augmentations_per_sample = metadata['augmentations_per_sample']
        remainder = metadata['remainder']

        generation_tasks = []
        for idx, item in enumerate(data):
            if idx < start_index:
                continue
            current_augmentations = augmentations_per_sample
            if idx < remainder:
                current_augmentations += 1
            start_aug = start_round if idx == start_index else 0
            for aug_round in range(start_aug, current_augmentations):
                if len(augmented_data) + len(generation_tasks) >= target_total:
                    break
                diversity = aug_round % 3
                generation_tasks.append({
                    'item': item,
                    'source_index': idx,
                    'aug_round': aug_round,
                    'diversity': diversity
                })
            if len(augmented_data) + len(generation_tasks) >= target_total:
                break

        print(f"\n准备生成 {len(generation_tasks)} 个样本")

        samples_since_checkpoint = 0
        start_time = time.time()
        success_count = 0

        current_task_for_checkpoint = generation_tasks[0] if generation_tasks else {}

        try:
            for batch_start in tqdm(range(0, len(generation_tasks), batch_size), desc="生成进度"):
                batch_tasks = generation_tasks[batch_start : batch_start + batch_size]
                batch_prompts = []

                for task in batch_tasks:
                    item = task['item']
                    original_label_is_true = item.get('text_a_is_closer', True)
                    anchor = item.get('anchor_text', '')

                    gen_strategy = 'positive' if task['aug_round'] % 2 == 0 else 'negative'

                    if gen_strategy == 'positive':
                        negative_story = item.get('text_b', '') if original_label_is_true else item.get('text_a', '')
                        if not anchor or not negative_story:
                            batch_prompts.append(None)
                            continue
                        prompt = self.track_a_positive_gen_template.format(
                            anchor=anchor,
                            negative=negative_story,
                            instruction=self.diversity_instructions[task['diversity']]
                        )
                        batch_prompts.append(prompt)
                    else:
                        positive_story = item.get('text_a', '') if original_label_is_true else item.get('text_b', '')
                        if not anchor or not positive_story:
                            batch_prompts.append(None)
                            continue
                        prompt = self.track_a_negative_gen_template.format(
                            anchor=anchor,
                            positive=positive_story,
                            instruction=self.diversity_instructions[task['diversity']]
                        )
                        batch_prompts.append(prompt)

                valid_tasks_and_prompts = [
                    (task, prompt) for task, prompt in zip(batch_tasks, batch_prompts) if prompt is not None
                ]
                if not valid_tasks_and_prompts:
                    continue

                valid_tasks, valid_prompts = zip(*valid_tasks_and_prompts)

                generated_texts = self.generate_text_batch(
                    list(valid_prompts),
                    max_new_tokens=512
                )

                for j, generated_text in enumerate(generated_texts):
                    task = valid_tasks[j]
                    item = task['item']
                    original_label_is_true = item.get('text_a_is_closer', True)
                    anchor = item.get('anchor_text', '')
                    gen_strategy = 'positive' if task['aug_round'] % 2 == 0 else 'negative'

                    if not generated_text:
                        continue

                    try:
                        new_item = {"anchor_text": anchor}
                        gen_type_meta = "unknown"

                        if gen_strategy == 'positive':
                            negative_story = item.get('text_b', '') if original_label_is_true else item.get('text_a', '')
                            if original_label_is_true:
                                new_item["text_a"] = generated_text
                                new_item["text_b"] = negative_story
                                new_item["text_a_is_closer"] = True
                            else:
                                new_item["text_a"] = negative_story
                                new_item["text_b"] = generated_text
                                new_item["text_a_is_closer"] = False
                            gen_type_meta = "positive"
                        else:
                            positive_story = item.get('text_a', '') if original_label_is_true else item.get('text_b', '')
                            if original_label_is_true:
                                new_item["text_a"] = positive_story
                                new_item["text_b"] = generated_text
                                new_item["text_a_is_closer"] = True
                            else:
                                new_item["text_a"] = generated_text
                                new_item["text_b"] = positive_story
                                new_item["text_a_is_closer"] = False
                            gen_type_meta = "negative"

                        new_item['_augmented'] = True
                        new_item['_source_index'] = task['source_index']
                        new_item['_augmentation_round'] = task['aug_round'] + 1
                        new_item['_diversity_level'] = task['diversity']
                        new_item['_gen_strategy'] = gen_type_meta

                        augmented_data.append(new_item)
                        success_count += 1

                    except Exception as e:
                        print(f"\n⚠ 警告: 处理结果时发生错误: {e}")
                        continue

                samples_since_checkpoint += len(batch_tasks)
                current_task_for_checkpoint = batch_tasks[0]

                if samples_since_checkpoint >= checkpoint_interval:
                    elapsed = time.time() - start_time
                    speed = success_count / elapsed if elapsed > 0 else 0
                    print(f"\n[检查点] 已生成 {success_count} 个样本, 速度: {speed:.2f} 样本/秒")

                    self.save_checkpoint(
                        checkpoint_path,
                        augmented_data,
                        current_task_for_checkpoint['source_index'],
                        current_task_for_checkpoint['aug_round'] + 1,
                        metadata
                    )
                    samples_since_checkpoint = 0

                if len(augmented_data) >= target_total:
                    break

        except KeyboardInterrupt:
            print("\n⚠ 中断，保存检查点...")
            self.save_checkpoint(
                checkpoint_path,
                augmented_data,
                current_task_for_checkpoint.get('source_index', 0),
                current_task_for_checkpoint.get('aug_round', 0),
                metadata
            )
            return augmented_data

        elapsed = time.time() - start_time
        print(f"\n✅ 生成完成！")
        if elapsed > 0:
            print(f"   总耗时: {elapsed/60:.1f} 分钟")
            print(f"   平均速度: {success_count/elapsed:.2f} 样本/秒")

        self.save_checkpoint(checkpoint_path, augmented_data, len(data), 0, metadata)
        return augmented_data[:target_total]

    def augment_track_b_batch(
        self,
        data: List[Dict[str, Any]],
        batch_size: int = 8,
        target_total: int = 10000,
        checkpoint_interval: int = 500,
        checkpoint_name: str = "track_b_checkpoint.json",
        resume: bool = True,
        include_original: bool = True
    ) -> List[Dict[str, Any]]:
        """Track B 数据增强 - 批处理优化"""
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = None
        if resume:
            checkpoint = self.load_checkpoint(checkpoint_path)
        if checkpoint:
            augmented_data = checkpoint['data']
            start_index = checkpoint['current_index']
            start_round = checkpoint['current_round']
            metadata = checkpoint['metadata']
        else:
            augmented_data = []
            start_index = 0
            start_round = 0
            if include_original:
                augmented_data.extend(data)
            original_count = len(data) if include_original else 0
            needed_augmentations = target_total - original_count
            if len(data) == 0:
                print("⚠ 警告: 原始数据为空，无法进行增强。")
                return []
            augmentations_per_sample = needed_augmentations // len(data) if len(data) > 0 else 0
            remainder = needed_augmentations % len(data) if len(data) > 0 else 0
            metadata = {
                'target_total': target_total,
                'original_count': len(data),
                'augmentations_per_sample': augmentations_per_sample,
                'remainder': remainder,
                'include_original': include_original
            }

        augmentations_per_sample = metadata['augmentations_per_sample']
        remainder = metadata['remainder']

        generation_tasks = []
        for idx, item in enumerate(data):
            if idx < start_index:
                continue
            current_augmentations = augmentations_per_sample
            if idx < remainder:
                current_augmentations += 1
            start_aug = start_round if idx == start_index else 0
            for aug_round in range(start_aug, current_augmentations):
                if len(augmented_data) + len(generation_tasks) >= target_total:
                    break
                diversity = aug_round % 3
                generation_tasks.append({
                    'item': item,
                    'source_index': idx,
                    'aug_round': aug_round,
                    'diversity': diversity
                })
            if len(augmented_data) + len(generation_tasks) >= target_total:
                break

        samples_since_checkpoint = 0
        start_time = time.time()
        success_count = 0

        current_task_for_checkpoint = generation_tasks[0] if generation_tasks else {}

        try:
            for batch_start in tqdm(range(0, len(generation_tasks), batch_size), desc="生成进度"):
                batch_tasks = generation_tasks[batch_start : batch_start + batch_size]

                batch_prompts = [
                    self.create_prompt_track_b(task['item'], task['diversity'])
                    for task in batch_tasks
                ]

                generated_texts = self.generate_text_batch(
                    batch_prompts,
                    max_new_tokens=512
                )

                for j, generated_text in enumerate(generated_texts):
                    task = batch_tasks[j]

                    if not generated_text:
                        continue

                    try:
                        new_item = {
                            'text': generated_text,
                            '_augmented': True,
                            '_source_index': task['source_index'],
                            '_augmentation_round': task['aug_round'] + 1,
                            '_diversity_level': task['diversity']
                        }
                        augmented_data.append(new_item)
                        success_count += 1
                    except Exception as e:
                        print(f"\n⚠ 警告: 处理结果时发生错误: {e}")
                        continue

                samples_since_checkpoint += len(batch_tasks)
                current_task_for_checkpoint = batch_tasks[0]

                if samples_since_checkpoint >= checkpoint_interval:
                    elapsed = time.time() - start_time
                    speed = success_count / elapsed if elapsed > 0 else 0
                    print(f"\n[检查点] 已生成 {success_count} 个样本, 速度: {speed:.2f} 样本/秒")

                    self.save_checkpoint(
                        checkpoint_path,
                        augmented_data,
                        current_task_for_checkpoint['source_index'],
                        current_task_for_checkpoint['aug_round'] + 1,
                        metadata
                    )
                    samples_since_checkpoint = 0

                if len(augmented_data) >= target_total:
                    break

        except KeyboardInterrupt:
            print("\n⚠ 中断，保存检查点...")
            self.save_checkpoint(
                checkpoint_path,
                augmented_data,
                current_task_for_checkpoint.get('source_index', 0),
                current_task_for_checkpoint.get('aug_round', 0),
                metadata
            )
            return augmented_data

        elapsed = time.time() - start_time
        if elapsed > 0:
            print(f"\n✅ 生成完成！总耗时: {elapsed/60:.1f} 分钟")

        self.save_checkpoint(checkpoint_path, augmented_data, len(data), 0, metadata)
        return augmented_data[:target_total]

    def save_to_jsonl(self, data: List[Dict[str, Any]], output_path: str):
        """保存为 JSONLines 格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✓ 已保存到: {output_path}")

    def save_to_csv(self, data: List[Dict[str, Any]], output_path: str):
        """保存为 CSV 格式"""
        if not data:
            print("⚠ 警告: 没有数据可保存到 CSV。")
            return

        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        fieldnames = sorted(list(all_keys))
        if not fieldnames:
            print("⚠ 警告: 未能从数据中提取任何 CSV 字段。")
            return

        try:
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                dict_data = [item for item in data if isinstance(item, dict)]
                writer.writerows(dict_data)
            print(f"✓ 已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存 CSV 失败: {e}")


def main():
    """主函数"""
    import os
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

    # ✅ 使用 4-bit 量化以节省显存
    augmentor = Qwen3DataAugmentorOptimized(
        model_name="/root/autodl-tmp/Qwen3-4B-Instruct-2507",
        max_seq_length=2048,
        load_in_4bit=True,     # 4-bit 量化
        dtype=None,
        checkpoint_dir="./checkpoints"
    )

    print("\n" + "="*70)
    print("🚀 混合数据增强 (V3.4 最终修复版 - 4B @ 4-bit)")
    print("="*70)

    # ✅ 使用保守的批处理大小
    EFFECTIVE_BATCH_SIZE = 100  # 可以根据实际显存使用情况调整

    print("\n📊 Track A: 加载数据")
    dev_a = augmentor.load_jsonl("../TrainData/dev_track_a.jsonl")
    synthetic_a = augmentor.load_jsonl("../TrainData/synthetic_data_for_classification.jsonl")
    mixed_a_data = dev_a + synthetic_a
    print(f"基础样本: {len(mixed_a_data)} 个")

    print("\n🔄 开始生成 Track A...")
    augmented_mixed_a = augmentor.augment_track_a_batch(
        mixed_a_data,
        batch_size=EFFECTIVE_BATCH_SIZE,
        target_total=10000,
        checkpoint_interval=500,
        checkpoint_name="track_a_mixed_10k_opt.json",
        resume=True,
        include_original=True
    )

    print(f"\n✅ Track A 完成: {len(augmented_mixed_a)} 个样本")
    augmentor.save_to_jsonl(augmented_mixed_a, "train_track_a_mixed_10k.jsonl")
    augmentor.save_to_csv(augmented_mixed_a, "train_track_a_mixed_10k.csv")

    print("\n📊 Track B: 加载数据")
    dev_b = augmentor.load_jsonl("../TrainData/dev_track_b.jsonl")
    synthetic_b = augmentor.load_jsonl("../TrainData/synthetic_data_for_contrastive_learning.jsonl")
    mixed_b_data = dev_b + synthetic_b
    print(f"基础样本: {len(mixed_b_data)} 个")

    print("\n🔄 开始生成 Track B...")
    augmented_mixed_b = augmentor.augment_track_b_batch(
        mixed_b_data,
        batch_size=EFFECTIVE_BATCH_SIZE,
        target_total=10000,
        checkpoint_interval=500,
        checkpoint_name="track_b_mixed_10k_opt.json",
        resume=True,
        include_original=True
    )

    print(f"\n✅ Track B 完成: {len(augmented_mixed_b)} 个样本")
    augmentor.save_to_jsonl(augmented_mixed_b, "train_track_b_mixed_10k.jsonl")
    augmentor.save_to_csv(augmented_mixed_b, "train_track_b_mixed_10k.csv")

    print("\n" + "="*70)
    print("🎉 所有任务完成！")
    print("="*70)


if __name__ == "__main__":
    main()