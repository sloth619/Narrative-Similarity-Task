## 文件夹内容如下：

**Qwen3DataAugmentor**: 训练集数据增强代码

**TrainCode**: 训练代码

**TrainingSet1**: 官方数据集

**TrainingSet2**: 增强后的一万条数据

**ROCStories**: 70k五句话故事

## Track B：

### A100-40GB：

| # | 模型                 | 训练数据                    | r  | lora_alpha | 学习率  | batch_size | Epochs | 最佳 Acc | Score |
|:--|:-------------------|:------------------------|:---|:-----------|:-----|:-----------|:-------|:-------|:------|
| 1 | Qwen3-Embedding-4B | TrainingSet1            | 32 | 64         | 2e-5 | 64         | 5      | 0.6350 | 0.63  |
|   | Qwen3-Embedding-4B | TrainingSet2            | 32 | 64         | 2e-5 | 64         | 5      | 0.6350 | 0.63  |
| 2 | Qwen3-Embedding-4B | TrainingSet1+ROCStories | 32 | 64         | 5e-7 | 100        | 3      | 0.620  | 0.63  |
| 3 | Qwen3-Embedding-4B | TrainingSet1+CMU        | /  | /          | 5e-7 | 64         | 3      | 0.650  | 0.65  |

### RTX5080-16GB：

| # | 模型                 | 训练数据                    | r  | lora_alpha | 学习率  | batch_size | Epochs | 最佳 Acc | Score |
|:--|:-------------------|:------------------------|:---|:-----------|:-----|:-----------|:-------|:-------|:------|
| 1 | Qwen3-Embedding-4B | TrainingSet1            | 32 | 64         | 5e-7 | 8          | 3      | 0.630  | 0.63  |
|   | Qwen3-Embedding-4B | TrainingSet2            | 32 | 64         | 5e-7 | 8          | 3      | 0.640  | 0.64  | 
| 2 | BGE-large-en-v1.5  | TrainingSet1            | /  | /          | 2e-5 | 8          | 5      | 0.635  | 0.64  |
|   | BGE-large-en-v1.5  | TrainingSet1+ROCStories | /  | /          | 2e-5 | 8          | 5      | 0.630  | 0.63  | 
| 3 | BGE-large-en-v1.5  | TrainingSet1+CMU        | /  | /          | 2e-5 | 8          | 5      | 0.595  | 0.60  |
| 4 | e5-large-v2        | TrainingSet1            | /  | /          | 2e-5 | 16         | 5      | 0.630  | 0.63  | 