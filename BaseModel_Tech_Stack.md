## 微调

### LoRA

**1. 效果不好**

按优先级排查：数据问题、超参数设定、任务类型适配度

- 数据问题：是否与目标任务对齐；数据量是否充足；数据质量；badcase/boundcase是否充分
- 调整超参：秩、alpha、dropout、扩大LoRA应用范围
- 任务与方法的匹配度：需要注入大量知识时低秩不成立，用DoRA或全参；多任务场景考虑MoLoRA；基座模型能力不足则先continue training后LoRA

### 推理优化

#### vLLM

PagedAttention

KVCache