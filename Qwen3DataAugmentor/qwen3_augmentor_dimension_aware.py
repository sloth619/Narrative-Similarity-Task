"""
Qwen3 数据增强脚本 - V4.0 三维度感知版
基于 SemEval-2026 Narrative Similarity Annotation Guidelines

核心改进:
1. 显式建模三个叙事相似度维度:
   - Abstract Theme (抽象主题)
   - Course of Action (行动过程)
   - Outcomes (结果)
2. 生成"只有N个维度相似"的困难样本
3. 参考PDF Examples的高质量提示词
4. 增加质量过滤机制
"""

import json
import csv
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import torch
from datetime import datetime
import time
from collections import Counter

from unsloth import FastLanguageModel


class Qwen3DataAugmentorV4_DimensionAware:
    """V4.0 三维度感知数据增强器"""

    def __init__(
            self,
            model_name: str = "/root/autodl-tmp/Qwen3-4B-Instruct-2507",
            max_seq_length: int = 2048,
            load_in_4bit: bool = True,
            dtype=None,
            device: str = "auto",
            checkpoint_dir: str = "./checkpoints"
    ):
        print(f"🚀 正在加载模型: {model_name} (4-bit: {load_in_4bit})")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=dtype,
        )

        FastLanguageModel.for_inference(self.model)

        # 设置 tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # === 系统消息 ===
        self.system_message = {
            "role": "system",
            "content": "You are an expert narrative analyst specializing in story similarity across three dimensions: Abstract Theme, Course of Action, and Outcomes."
        }

        # === 三维度增强模板 ===
        self._init_dimension_templates()

        print("✅ 模型加载完成")
        print(f"   Padding side: {self.tokenizer.padding_side}")

    def _init_dimension_templates(self):
        """初始化三维度增强模板"""

        # ==========================================
        # Track A: 三维度控制模板
        # ==========================================

        # === 策略1: 只保留Theme相似 ===
        self.track_a_theme_only_template = """You are creating a HARD NEGATIVE sample for narrative similarity training.

**Reference Stories:**
Anchor Story:
{anchor}

Positive Story (highly similar to Anchor):
{positive}

**Your Task:**
Create a NEW story that:
1. ✅ Shares the ABSTRACT THEME with the Anchor (same core problem, conflict, or central idea)
2. ❌ Has DIFFERENT Course of Action (different sequence of events)
3. ❌ Has DIFFERENT Outcome (different ending or resolution)

**Guidelines (from Annotation PDF):**
- Abstract Theme = the defining constellation of problems, central ideas, and core motifs
- Focus on WHAT the story is fundamentally about, not HOW it unfolds
- Example: Both stories about "loneliness leading to isolation", but one is a wizard, one is an explorer

**Instruction:** {instruction}

**Output ONLY the new story text (no explanations):**"""

        # === 策略2: 只保留Action相似 ===
        self.track_a_action_only_template = """You are creating a HARD NEGATIVE sample for narrative similarity training.

**Reference Stories:**
Anchor Story:
{anchor}

Positive Story (highly similar to Anchor):
{positive}

**Your Task:**
Create a NEW story that:
1. ❌ Has DIFFERENT Abstract Theme (different underlying problem or idea)
2. ✅ Shares the COURSE OF ACTION with the Anchor (same sequence of events, turning points)
3. ❌ Has DIFFERENT Outcome (different ending)

**Guidelines (from Annotation PDF Example 5):**
- Course of Action = the sequence and order of events
- Example structure: [Character prepares] → [Character sets up] → [Event happens]
- Keep this A→B→C structure, but change WHY it happens and WHAT the result is

**Instruction:** {instruction}

**Output ONLY the new story text (no explanations):**"""

        # === 策略3: 只保留Outcome相似 ===
        self.track_a_outcome_only_template = """You are creating a HARD NEGATIVE sample for narrative similarity training.

**Reference Stories:**
Anchor Story:
{anchor}

Positive Story (highly similar to Anchor):
{positive}

**Your Task:**
Create a NEW story that:
1. ❌ Has DIFFERENT Abstract Theme (different core problem)
2. ❌ Has DIFFERENT Course of Action (different events)
3. ✅ Shares the OUTCOME with the Anchor (same ending, fate, or resolution)

**Guidelines (from Annotation PDF Example 4):**
- Outcome = final results, character fates, moral lessons
- Example: Both end with "character dies alone, body not found", but completely different themes and events

**Instruction:** {instruction}

**Output ONLY the new story text (no explanations):**"""

        # === 策略4: Theme + Action 相似, Outcome不同 (困难样本!) ===
        self.track_a_theme_action_diff_outcome_template = """You are creating a VERY HARD NEGATIVE sample (this is the most challenging type).

**Reference Stories:**
Anchor Story:
{anchor}

Positive Story (highly similar to Anchor):
{positive}

**Your Task:**
Create a NEW story that:
1. ✅ Shares the ABSTRACT THEME with the Anchor (same core problem)
2. ✅ Shares the COURSE OF ACTION with the Anchor (similar event sequence)
3. ❌ Has OPPOSITE Outcome (completely different ending!)

**Guidelines (from Annotation PDF Example 3):**
- This is like: "Injury → Struggles → Recovery" vs "Injury → Struggles → Death"
- Theme and events are similar, but the ending flips the meaning

**Instruction:** {instruction}

**Output ONLY the new story text (no explanations):**"""

        # === 策略5: 全维度相似 (正样本) ===
        self.track_a_all_similar_template = """You are creating a POSITIVE sample for narrative similarity training.

**Reference Stories:**
Anchor Story:
{anchor}

Negative Story (dissimilar to Anchor):
{negative}

**Your Task:**
Create a NEW story that is HIGHLY SIMILAR to the Anchor across all three dimensions:
1. ✅ Same ABSTRACT THEME (same core problem, conflict, idea)
2. ✅ Similar COURSE OF ACTION (similar event sequence and turning points)
3. ✅ Similar OUTCOME (similar ending or resolution)

**Guidelines:**
- This should be clearly MORE similar to Anchor than the Negative story
- You can change: setting, time period, character names, writing style, level of detail
- You cannot change: the fundamental story being told

**Instruction:** {instruction}

**Output ONLY the new story text (no explanations):**"""

        # ==========================================
        # Track B: 三维度控制模板
        # ==========================================

        self.track_b_dimension_templates = {
            'theme_preserving': """Create a similar story to the original, preserving its ABSTRACT THEME.

**Original Story:**
{text}

**Requirements:**
1. ✅ Keep the SAME abstract theme (core problem, conflict, central idea)
2. ⚡ You may change: specific events, sequence, outcome
3. ⚡ You may change: setting, characters, time period

**Instruction:** {instruction}

**Output ONLY the new story text:**""",

            'action_preserving': """Create a similar story to the original, preserving its COURSE OF ACTION.

**Original Story:**
{text}

**Requirements:**
1. ✅ Keep the SAME sequence of events and turning points
2. ⚡ You may change: the underlying theme/motivation
3. ⚡ You may change: the final outcome

**Instruction:** {instruction}

**Output ONLY the new story text:**""",

            'outcome_preserving': """Create a similar story to the original, preserving its OUTCOME.

**Original Story:**
{text}

**Requirements:**
1. ✅ Keep the SAME final outcome/resolution
2. ⚡ You may change: the theme
3. ⚡ You may change: how characters reach this outcome

**Instruction:** {instruction}

**Output ONLY the new story text:**""",

            'all_similar': """Create a story that is highly similar to the original across all dimensions.

**Original Story:**
{text}

**Requirements:**
1. ✅ Keep similar abstract theme
2. ✅ Keep similar course of action
3. ✅ Keep similar outcome
4. Change only: wording, details, setting, character names

**Instruction:** {instruction}

**Output ONLY the new story text:**"""
        }

        # === 多样性指令库 ===
        self.diversity_instructions = [
            "Use different character types and settings, but maintain the narrative structure.",
            "Retell in a different genre (e.g., modern day, sci-fi, historical), but keep the core narrative.",
            "Change the scale (e.g., personal drama → global event, or vice versa), but preserve the specified dimensions.",
            "Use a different tone (e.g., serious → comedic), but maintain the specified narrative aspects.",
            "Change the cultural context, but keep the fundamental story dimensions intact."
        ]

        print("📚 已加载三维度增强模板")

    # ==========================================
    # 核心生成方法
    # ==========================================

    def generate_text_batch(
            self,
            prompts: List[str],
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.8,
            top_k: int = 20,
            repetition_penalty: float = 1.1
    ) -> List[str]:
        """批量生成文本"""

        messages_batch = [
            [self.system_message, {"role": "user", "content": p}]
            for p in prompts
        ]

        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_batch
        ]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        input_ids_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_ids_len:]
        generated_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return [text.strip() for text in generated_texts]

    # ==========================================
    # Track A: 三维度增强
    # ==========================================

    def augment_track_a_dimension_aware(
            self,
            data: List[Dict[str, Any]],
            batch_size: int = 8,
            target_total: int = 10000,
            checkpoint_interval: int = 500,
            checkpoint_name: str = "track_a_dimension_aware.json",
            resume: bool = True,
            include_original: bool = True,
            dimension_distribution: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Track A 三维度感知的数据增强

        Args:
            dimension_distribution: 各维度策略的分布比例
                {
                    'theme_only': 0.15,           # 只Theme相似
                    'action_only': 0.15,          # 只Action相似
                    'outcome_only': 0.15,         # 只Outcome相似
                    'theme_action_diff_outcome': 0.25,  # Theme+Action相似,Outcome不同(困难!)
                    'all_similar': 0.30           # 全维度相似(正样本)
                }
        """

        if dimension_distribution is None:
            dimension_distribution = {
                'theme_only': 0.15,
                'action_only': 0.15,
                'outcome_only': 0.15,
                'theme_action_diff_outcome': 0.25,
                'all_similar': 0.30
            }

        print(f"\n{'=' * 70}")
        print(f"🎯 Track A 三维度感知增强")
        print(f"{'=' * 70}")
        print(f"维度分布:")
        for dim, ratio in dimension_distribution.items():
            print(f"  - {dim}: {ratio * 100:.1f}%")

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # 加载检查点
        checkpoint = None
        if resume and checkpoint_path.exists():
            checkpoint = self._load_checkpoint(checkpoint_path)

        if checkpoint:
            augmented_data = checkpoint['data']
            start_index = checkpoint['current_index']
            start_round = checkpoint['current_round']
            metadata = checkpoint['metadata']
            print(f"✅ 从检查点恢复: 已有 {len(augmented_data)} 个样本")
        else:
            augmented_data = []
            start_index = 0
            start_round = 0

            if include_original:
                augmented_data.extend(data)
                print(f"✅ 已添加 {len(data)} 个原始样本")

            original_count = len(data) if include_original else 0
            needed_augmentations = target_total - original_count

            if len(data) == 0:
                print("⚠ 警告: 原始数据为空")
                return []

            augmentations_per_sample = needed_augmentations // len(data)
            remainder = needed_augmentations % len(data)

            metadata = {
                'target_total': target_total,
                'original_count': len(data),
                'augmentations_per_sample': augmentations_per_sample,
                'remainder': remainder,
                'dimension_distribution': dimension_distribution
            }

        augmentations_per_sample = metadata['augmentations_per_sample']
        remainder = metadata['remainder']

        # === 生成任务列表 ===
        generation_tasks = []
        dimension_choices = list(dimension_distribution.keys())
        dimension_weights = list(dimension_distribution.values())

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

                # 随机选择维度策略
                dimension_strategy = random.choices(
                    dimension_choices,
                    weights=dimension_weights
                )[0]

                diversity = random.randint(0, len(self.diversity_instructions) - 1)

                generation_tasks.append({
                    'item': item,
                    'source_index': idx,
                    'aug_round': aug_round,
                    'dimension_strategy': dimension_strategy,
                    'diversity': diversity
                })

            if len(augmented_data) + len(generation_tasks) >= target_total:
                break

        print(f"\n📊 准备生成 {len(generation_tasks)} 个样本")

        # 统计维度分布
        strategy_counts = Counter([t['dimension_strategy'] for t in generation_tasks])
        for strategy, count in strategy_counts.items():
            print(f"   - {strategy}: {count} ({count / len(generation_tasks) * 100:.1f}%)")

        # === 批量生成 ===
        samples_since_checkpoint = 0
        start_time = time.time()
        success_count = 0
        failed_count = 0

        current_task_for_checkpoint = generation_tasks[0] if generation_tasks else {}

        try:
            for batch_start in tqdm(range(0, len(generation_tasks), batch_size), desc="🔄 生成进度"):
                batch_tasks = generation_tasks[batch_start: batch_start + batch_size]
                batch_prompts = []

                for task in batch_tasks:
                    prompt = self._create_track_a_prompt(task)
                    batch_prompts.append(prompt if prompt else None)

                # 过滤无效prompts
                valid_tasks_and_prompts = [
                    (task, prompt) for task, prompt in zip(batch_tasks, batch_prompts)
                    if prompt is not None
                ]

                if not valid_tasks_and_prompts:
                    continue

                valid_tasks, valid_prompts = zip(*valid_tasks_and_prompts)

                # 生成
                generated_texts = self.generate_text_batch(
                    list(valid_prompts),
                    max_new_tokens=512
                )

                # 处理结果
                for j, generated_text in enumerate(generated_texts):
                    task = valid_tasks[j]

                    if not generated_text or len(generated_text.strip()) < 50:
                        failed_count += 1
                        continue

                    try:
                        new_item = self._construct_track_a_item(task, generated_text)
                        augmented_data.append(new_item)
                        success_count += 1
                    except Exception as e:
                        print(f"\n⚠ 处理结果错误: {e}")
                        failed_count += 1
                        continue

                samples_since_checkpoint += len(batch_tasks)
                current_task_for_checkpoint = batch_tasks[0]

                # 保存检查点
                if samples_since_checkpoint >= checkpoint_interval:
                    elapsed = time.time() - start_time
                    speed = success_count / elapsed if elapsed > 0 else 0
                    print(f"\n💾 [检查点] 成功: {success_count}, 失败: {failed_count}, 速度: {speed:.2f} 样本/秒")

                    self._save_checkpoint(
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
            self._save_checkpoint(
                checkpoint_path,
                augmented_data,
                current_task_for_checkpoint.get('source_index', 0),
                current_task_for_checkpoint.get('aug_round', 0),
                metadata
            )
            return augmented_data

        # 最终统计
        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"✅ Track A 生成完成!")
        print(f"   总样本: {len(augmented_data)}")
        print(f"   成功生成: {success_count}")
        print(f"   失败: {failed_count}")
        print(f"   总耗时: {elapsed / 60:.1f} 分钟")
        if elapsed > 0:
            print(f"   平均速度: {success_count / elapsed:.2f} 样本/秒")
        print(f"{'=' * 70}\n")

        self._save_checkpoint(checkpoint_path, augmented_data, len(data), 0, metadata)
        return augmented_data[:target_total]

    def _create_track_a_prompt(self, task: Dict) -> Optional[str]:
        """根据维度策略创建Track A提示词"""

        item = task['item']
        strategy = task['dimension_strategy']
        diversity_instruction = self.diversity_instructions[task['diversity']]

        original_label_is_true = item.get('text_a_is_closer', True)
        anchor = item.get('anchor_text', '')

        if strategy == 'all_similar':
            # 生成正样本: 需要negative story
            negative_story = item.get('text_b', '') if original_label_is_true else item.get('text_a', '')
            if not anchor or not negative_story:
                return None

            return self.track_a_all_similar_template.format(
                anchor=anchor,
                negative=negative_story,
                instruction=diversity_instruction
            )
        else:
            # 生成负样本: 需要positive story
            positive_story = item.get('text_a', '') if original_label_is_true else item.get('text_b', '')
            if not anchor or not positive_story:
                return None

            # 选择对应的模板
            template_map = {
                'theme_only': self.track_a_theme_only_template,
                'action_only': self.track_a_action_only_template,
                'outcome_only': self.track_a_outcome_only_template,
                'theme_action_diff_outcome': self.track_a_theme_action_diff_outcome_template
            }

            template = template_map.get(strategy)
            if not template:
                return None

            return template.format(
                anchor=anchor,
                positive=positive_story,
                instruction=diversity_instruction
            )

    def _construct_track_a_item(self, task: Dict, generated_text: str) -> Dict:
        """构造Track A数据项"""

        item = task['item']
        strategy = task['dimension_strategy']
        original_label_is_true = item.get('text_a_is_closer', True)
        anchor = item.get('anchor_text', '')

        new_item = {"anchor_text": anchor}

        if strategy == 'all_similar':
            # 正样本: 生成的故事比negative更相似
            negative_story = item.get('text_b', '') if original_label_is_true else item.get('text_a', '')

            if original_label_is_true:
                new_item["text_a"] = generated_text
                new_item["text_b"] = negative_story
                new_item["text_a_is_closer"] = True
            else:
                new_item["text_a"] = negative_story
                new_item["text_b"] = generated_text
                new_item["text_a_is_closer"] = False
        else:
            # 负样本: 生成的故事比positive更不相似
            positive_story = item.get('text_a', '') if original_label_is_true else item.get('text_b', '')

            if original_label_is_true:
                new_item["text_a"] = positive_story
                new_item["text_b"] = generated_text
                new_item["text_a_is_closer"] = True
            else:
                new_item["text_a"] = generated_text
                new_item["text_b"] = positive_story
                new_item["text_a_is_closer"] = False

        # 元数据
        new_item['_augmented'] = True
        new_item['_source_index'] = task['source_index']
        new_item['_augmentation_round'] = task['aug_round'] + 1
        new_item['_diversity_level'] = task['diversity']
        new_item['_dimension_strategy'] = strategy

        return new_item

    # ==========================================
    # Track B: 三维度增强
    # ==========================================

    def augment_track_b_dimension_aware(
            self,
            data: List[Dict[str, Any]],
            batch_size: int = 8,
            target_total: int = 10000,
            checkpoint_interval: int = 500,
            checkpoint_name: str = "track_b_dimension_aware.json",
            resume: bool = True,
            include_original: bool = True,
            dimension_distribution: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Track B 三维度感知的数据增强

        Args:
            dimension_distribution: 各维度策略的分布比例
                {
                    'theme_preserving': 0.25,
                    'action_preserving': 0.25,
                    'outcome_preserving': 0.25,
                    'all_similar': 0.25
                }
        """

        if dimension_distribution is None:
            dimension_distribution = {
                'theme_preserving': 0.25,
                'action_preserving': 0.25,
                'outcome_preserving': 0.25,
                'all_similar': 0.25
            }

        print(f"\n{'=' * 70}")
        print(f"🎯 Track B 三维度感知增强")
        print(f"{'=' * 70}")
        print(f"维度分布:")
        for dim, ratio in dimension_distribution.items():
            print(f"  - {dim}: {ratio * 100:.1f}%")

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # 加载检查点
        checkpoint = None
        if resume and checkpoint_path.exists():
            checkpoint = self._load_checkpoint(checkpoint_path)

        if checkpoint:
            augmented_data = checkpoint['data']
            start_index = checkpoint['current_index']
            start_round = checkpoint['current_round']
            metadata = checkpoint['metadata']
            print(f"✅ 从检查点恢复: 已有 {len(augmented_data)} 个样本")
        else:
            augmented_data = []
            start_index = 0
            start_round = 0

            if include_original:
                augmented_data.extend(data)
                print(f"✅ 已添加 {len(data)} 个原始样本")

            original_count = len(data) if include_original else 0
            needed_augmentations = target_total - original_count

            if len(data) == 0:
                print("⚠ 警告: 原始数据为空")
                return []

            augmentations_per_sample = needed_augmentations // len(data)
            remainder = needed_augmentations % len(data)

            metadata = {
                'target_total': target_total,
                'original_count': len(data),
                'augmentations_per_sample': augmentations_per_sample,
                'remainder': remainder,
                'dimension_distribution': dimension_distribution
            }

        augmentations_per_sample = metadata['augmentations_per_sample']
        remainder = metadata['remainder']

        # === 生成任务列表 ===
        generation_tasks = []
        dimension_choices = list(dimension_distribution.keys())
        dimension_weights = list(dimension_distribution.values())

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

                dimension_strategy = random.choices(
                    dimension_choices,
                    weights=dimension_weights
                )[0]

                diversity = random.randint(0, len(self.diversity_instructions) - 1)

                generation_tasks.append({
                    'item': item,
                    'source_index': idx,
                    'aug_round': aug_round,
                    'dimension_strategy': dimension_strategy,
                    'diversity': diversity
                })

            if len(augmented_data) + len(generation_tasks) >= target_total:
                break

        print(f"\n📊 准备生成 {len(generation_tasks)} 个样本")

        # 统计维度分布
        strategy_counts = Counter([t['dimension_strategy'] for t in generation_tasks])
        for strategy, count in strategy_counts.items():
            print(f"   - {strategy}: {count} ({count / len(generation_tasks) * 100:.1f}%)")

        # === 批量生成 ===
        samples_since_checkpoint = 0
        start_time = time.time()
        success_count = 0
        failed_count = 0

        current_task_for_checkpoint = generation_tasks[0] if generation_tasks else {}

        try:
            for batch_start in tqdm(range(0, len(generation_tasks), batch_size), desc="🔄 生成进度"):
                batch_tasks = generation_tasks[batch_start: batch_start + batch_size]

                batch_prompts = [
                    self._create_track_b_prompt(task)
                    for task in batch_tasks
                ]

                generated_texts = self.generate_text_batch(
                    batch_prompts,
                    max_new_tokens=512
                )

                for j, generated_text in enumerate(generated_texts):
                    task = batch_tasks[j]

                    if not generated_text or len(generated_text.strip()) < 50:
                        failed_count += 1
                        continue

                    try:
                        new_item = {
                            'text': generated_text,
                            '_augmented': True,
                            '_source_index': task['source_index'],
                            '_augmentation_round': task['aug_round'] + 1,
                            '_diversity_level': task['diversity'],
                            '_dimension_strategy': task['dimension_strategy']
                        }
                        augmented_data.append(new_item)
                        success_count += 1
                    except Exception as e:
                        print(f"\n⚠ 处理结果错误: {e}")
                        failed_count += 1
                        continue

                samples_since_checkpoint += len(batch_tasks)
                current_task_for_checkpoint = batch_tasks[0]

                if samples_since_checkpoint >= checkpoint_interval:
                    elapsed = time.time() - start_time
                    speed = success_count / elapsed if elapsed > 0 else 0
                    print(f"\n💾 [检查点] 成功: {success_count}, 失败: {failed_count}, 速度: {speed:.2f} 样本/秒")

                    self._save_checkpoint(
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
            self._save_checkpoint(
                checkpoint_path,
                augmented_data,
                current_task_for_checkpoint.get('source_index', 0),
                current_task_for_checkpoint.get('aug_round', 0),
                metadata
            )
            return augmented_data

        # 最终统计
        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"✅ Track B 生成完成!")
        print(f"   总样本: {len(augmented_data)}")
        print(f"   成功生成: {success_count}")
        print(f"   失败: {failed_count}")
        print(f"   总耗时: {elapsed / 60:.1f} 分钟")
        if elapsed > 0:
            print(f"   平均速度: {success_count / elapsed:.2f} 样本/秒")
        print(f"{'=' * 70}\n")

        self._save_checkpoint(checkpoint_path, augmented_data, len(data), 0, metadata)
        return augmented_data[:target_total]

    def _create_track_b_prompt(self, task: Dict) -> str:
        """根据维度策略创建Track B提示词"""

        item = task['item']
        strategy = task['dimension_strategy']
        diversity_instruction = self.diversity_instructions[task['diversity']]
        text = item.get('text', '')

        template = self.track_b_dimension_templates[strategy]
        return template.format(
            text=text,
            instruction=diversity_instruction
        )

    # ==========================================
    # 工具方法
    # ==========================================

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """加载 JSONLines 文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _save_checkpoint(
            self,
            checkpoint_path: Path,
            data: List[Dict],
            current_index: int,
            current_round: int,
            metadata: Dict
    ):
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

    def _load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict]:
        """加载检查点"""
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ 加载检查点失败: {e}")
            return None

    def save_to_jsonl(self, data: List[Dict[str, Any]], output_path: str):
        """保存为 JSONLines 格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"💾 已保存到: {output_path}")

    def save_to_csv(self, data: List[Dict[str, Any]], output_path: str):
        """保存为 CSV 格式"""
        if not data:
            print("⚠ 警告: 没有数据可保存到 CSV")
            return

        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        fieldnames = sorted(list(all_keys))

        if not fieldnames:
            print("⚠ 警告: 未能从数据中提取任何 CSV 字段")
            return

        try:
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                dict_data = [item for item in data if isinstance(item, dict)]
                writer.writerows(dict_data)
            print(f"💾 已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存 CSV 失败: {e}")


# ==========================================
# 主函数
# ==========================================

def main():
    """主函数 - 演示用法"""
    import os
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

    # 初始化增强器
    augmentor = Qwen3DataAugmentorV4_DimensionAware(
        model_name="/root/autodl-tmp/Qwen3-4B-Instruct-2507",
        max_seq_length=2048,
        load_in_4bit=True,
        checkpoint_dir="./checkpoints_v4"
    )

    print("\n" + "=" * 70)
    print("🚀 三维度感知数据增强 V4.0")
    print("=" * 70)

    # === Track A 增强 ===
    print("\n📊 Track A: 加载数据")
    dev_a = augmentor.load_jsonl("../TrainData/dev_track_a.jsonl")
    synthetic_a = augmentor.load_jsonl("../TrainData/synthetic_data_for_classification.jsonl")
    mixed_a_data = dev_a + synthetic_a
    print(f"   基础样本: {len(mixed_a_data)} 个")

    print("\n🔄 开始生成 Track A (三维度感知)...")
    augmented_a = augmentor.augment_track_a_dimension_aware(
        mixed_a_data,
        batch_size=8,
        target_total=10000,
        checkpoint_interval=500,
        checkpoint_name="track_a_dimension_10k.json",
        resume=True,
        include_original=True,
        dimension_distribution={
            'theme_only': 0.15,
            'action_only': 0.15,
            'outcome_only': 0.15,
            'theme_action_diff_outcome': 0.25,
            'all_similar': 0.30
        }
    )

    print(f"\n✅ Track A 完成: {len(augmented_a)} 个样本")
    augmentor.save_to_jsonl(augmented_a, "train_track_a_dimension_10k.jsonl")
    augmentor.save_to_csv(augmented_a, "train_track_a_dimension_10k.csv")

    # === Track B 增强 ===
    print("\n📊 Track B: 加载数据")
    dev_b = augmentor.load_jsonl("../TrainData/dev_track_b.jsonl")
    synthetic_b = augmentor.load_jsonl("../TrainData/synthetic_data_for_contrastive_learning.jsonl")
    mixed_b_data = dev_b + synthetic_b
    print(f"   基础样本: {len(mixed_b_data)} 个")

    print("\n🔄 开始生成 Track B (三维度感知)...")
    augmented_b = augmentor.augment_track_b_dimension_aware(
        mixed_b_data,
        batch_size=8,
        target_total=10000,
        checkpoint_interval=500,
        checkpoint_name="track_b_dimension_10k.json",
        resume=True,
        include_original=True,
        dimension_distribution={
            'theme_preserving': 0.25,
            'action_preserving': 0.25,
            'outcome_preserving': 0.25,
            'all_similar': 0.25
        }
    )

    print(f"\n✅ Track B 完成: {len(augmented_b)} 个样本")
    augmentor.save_to_jsonl(augmented_b, "train_track_b_dimension_10k.jsonl")
    augmentor.save_to_csv(augmented_b, "train_track_b_dimension_10k.csv")

    print("\n" + "=" * 70)
    print("🎉 所有任务完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()