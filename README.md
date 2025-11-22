## 文件夹内容如下：

**Qwen3DataAugmentor**: 训练集数据增强代码

**TrainCode**: 训练代码

**TrainingSet1**: 官方数据集

**TrainingSet2**: 增强后的一万条数据

**ROCStories**: 70k五句话故事

## Track A：

### RTX5080-16GB：

| # | 模型                 | 训练数据                         | r  | lora_alpha | 学习率  | batch_size | Epochs | 最佳 Acc | Score |
|:--|:-------------------|:-----------------------------|:---|:-----------|:-----|:-----------|:-------|:-------|:------|
| 1 | Qwen3-Reranker-4B  | TrainingSet1                 | 32 | 64         | 2e-5 | 16         | 3      | 0.555  | 0.56  |
| 2 | Qwen3-Embedding-4B | TrainingSet1                 | 32 | 64         | 5e-7 | 8          | 3      | 0.635  | 0.64  |
| 3 | BGE-large-en-v1.5  | TrainingSet1                 | /  | /          | 2e-5 | 8          | 5      | 0.660  | 0.66  |
| 3 | Qwen3-4B-Instruct  | TrainingSet1                 | 32 | 64         | 2e-5 | 8          | 5      | 0.630  | 0.63  |
| 4 | BGE-large-en-v1.5  | TrainingSet_optimized        | /  | /          | 2e-5 | 8          | 3      | 0.720  | 0.72  | 
| 5 | Deberta-v3-large   | TrainingSet1+Multiple Choice | /  | /          | 2e-6 | 8          | 3      | 0.395  | 0.40  | 

### LLM直接生成：

| # | 模型             | Score |
|:--|:---------------|:------|
| 1 | Gemini-2.5-pro | 0.71  | 

## Track B：

### A100-40GB：

| # | 模型                 | 训练数据                    | r  | lora_alpha | 学习率  | batch_size | Epochs | 最佳 Acc | Score |
|:--|:-------------------|:------------------------|:---|:-----------|:-----|:-----------|:-------|:-------|:------|
| 1 | Qwen3-Embedding-4B | TrainingSet1            | 32 | 64         | 2e-5 | 128        | 5      | 0.635  | 0.63  |
|   | Qwen3-Embedding-4B | TrainingSet2            | 32 | 64         | 2e-5 | 128        | 5      | 0.635  | 0.63  |
| 2 | Qwen3-Embedding-4B | TrainingSet1+ROCStories | 32 | 64         | 5e-7 | 100        | 3      | 0.620  | 0.63  |
| 3 | Qwen3-Embedding-4B | TrainingSet1+10kCMU     | 64 | 128        | 5e-7 | 128        | 3      | 0.650  | 0.65  |
| 4 | Qwen3-Embedding-4B | TrainingSet1+20kCMU     | 64 | 128        | 5e-7 | 144        | 3      | 0.635  | 0.64  |

### RTX5080-16GB：

| # | 模型                 | 训练数据                    | r  | lora_alpha | 学习率  | batch_size | Epochs | 最佳 Acc | Score |
|:--|:-------------------|:------------------------|:---|:-----------|:-----|:-----------|:-------|:-------|:------|
| 1 | Qwen3-Embedding-4B | TrainingSet1            | 32 | 64         | 5e-7 | 8          | 3      | 0.630  | 0.63  |
|   | Qwen3-Embedding-4B | TrainingSet2            | 32 | 64         | 5e-7 | 8          | 3      | 0.640  | 0.64  | 
| 2 | BGE-large-en-v1.5  | TrainingSet1            | /  | /          | 2e-5 | 8          | 5      | 0.660  | 0.66  |
|   | BGE-large-en-v1.5  | TrainingSet1+ROCStories | /  | /          | 2e-5 | 8          | 5      | 0.630  | 0.63  | 
| 3 | BGE-large-en-v1.5  | TrainingSet1+CMU        | /  | /          | 2e-5 | 8          | 5      | 0.595  | 0.60  |
| 4 | e5-large-v2        | TrainingSet1            | /  | /          | 2e-5 | 16         | 5      | 0.630  | 0.63  |
| 5 | BGE-m3             | TrainingSet1            | /  | /          | 2e-5 | 8          | 5      | 0.575  | 0.58  | 
| 6 | jina-v3            | TrainingSet1            | /  | /          | 2e-5 | 16         | 5      | 0.495  | 0.50  | 
| 7 | GTE-large-en-v1.5  | TrainingSet1            | /  | /          | 5e-7 | 16         | 5      | 0.600  | 0.60  | 
| 8 | Qwen3-Embedding-8B | TrainingSet1            | 32 | 64         | 5e-7 | 16         | 5      | 0.580  | 0.58  | 
| 9 | BGE-large-en-v1.5  | TrainingSet_optimized   | /  | /          | 2e-5 | 8          | 3      | 0.720  | 0.72  | 

### 用prompt直接预测：

| # | 模型                | 最佳 Acc | Score |
|:--|:------------------|:-------|:------|
| 1 | GTE-large-en-v1.5 | 0.67   | 0.67  | 
