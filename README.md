## 文件夹内容如下：

**Qwen3DataAugmentor**: 训练集数据增强代码

**TrainCode**: 训练代码

**TrainingSet1**: 官方数据集

**TrainingSet2**: 增强后的一万条数据

## Track B：

| 模型                 | 训练数据         | r  | lora_alpha | 学习率  | batch_size | Epochs | 最佳 Acc | Score |
|:-------------------|:-------------|:---|:-----------|:-----|:-----------|:-------|:-------|:------|
| Qwen3-Embedding-4B | TrainingSet1 | 32 | 64         | 2e-5 | 64         | 5      | 0.6350 | 0.63  |
| Qwen3-Embedding-4B | TrainingSet2 | 32 | 64         | 2e-5 | 64         | 5      | 0.6350 | 0.63  | 
| Qwen3-Embedding-4B | TrainingSet1 | 32 | 64         | 5e-7 | 8          | 3      | 0.6350 | 0.63  |
| Qwen3-Embedding-4B | TrainingSet2 | 32 | 64         | 5e-7 | 8          | 3      | 0.6350 | 0.63  | 