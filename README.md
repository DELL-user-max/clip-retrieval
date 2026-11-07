\# CLIP-Retrieval · 图文跨模态检索系统(含 LoRA 微调)



基于 OpenAI CLIP(ViT-B/32)的图文跨模态检索系统,完整覆盖:\*\*数据清洗 → 特征抽取 → 向量检索 → 零样本分类 → LoRA 微调 → Web Demo\*\*。



\## 核心结果



\*\*COCO val2017 文搜图检索(候选池 5000 张,查询 5000 条验证描述)\*\*



| 模型 | Recall@1 | Recall@5 | Recall@10 |

|---|---|---|---|

| CLIP ViT-B/32 零样本 | 32.08 | 56.78 | 68.28 |

| \*\*+ LoRA 微调(训练 0.32% 参数)\*\* | \*\*38.58\*\* | \*\*65.14\*\* | \*\*74.16\*\* |



\- 与 CLIP 论文报告的零样本基准(≈31/55/66)一致,验证了实现正确性

\- 零样本图像分类(COCO 80 类):Top-1 85.2% / Top-5 98.4%



\## 方法



1\. \*\*数据清洗\*\*:坏图双重校验(PIL verify + 强制完整解码)、caption 长度过滤(3\~32 词,适配 77 token 上限)、精确去重、\*\*按图片粒度\*\*划分训练/验证(防跨描述数据泄漏)

2\. \*\*特征抽取\*\*:CLIP 双塔编码 → 512 维向量 → L2 归一化(内积 = 余弦相似度)

3\. \*\*向量检索\*\*:FAISS IndexFlatIP;评测协议:5000 条验证描述 × 5000 张候选图

4\. \*\*LoRA 微调\*\*:PEFT,r=8,注入 q\_proj/v\_proj,对比学习损失;20k 图文对 1 epoch,单卡 RTX 3080Ti 约 10 分钟,可训练参数仅 0.32%

5\. \*\*Demo\*\*:Gradio Web 界面,支持任意英文描述检索



\## 快速开始



```bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install transformers peft faiss-cpu pillow pandas scikit-learn matplotlib gradio

python scripts/01\_parse\_captions.py      # 解析 COCO 标注

python scripts/02\_clean\_data.py          # 数据清洗

python scripts/03\_extract\_features.py    # CLIP 特征抽取(CPU 约 10\~30 分钟)

python scripts/04\_build\_index.py         # FAISS 索引 + 检索自测

python scripts/05\_evaluate\_retrieval.py  # Recall@1/5/10 评测

python scripts/06\_zero\_shot\_cls.py       # 零样本分类

python app/demo\_gradio.py                # Web 检索演示

```



微调需单卡 GPU(≥12G,AutoDL 3080Ti 实测):`python scripts/07\_finetune\_lora.py --bs 16 \&\& python scripts/08\_eval\_finetuned.py`



\## 评估协议说明(重要)



\- 训练/验证按\*\*图片粒度\*\* 8:2 划分,避免同一图片的多条描述跨集合造成数据泄漏

\- 检索评测统一使用\*\*全部 5000 张图作为候选池\*\*

\- 开发过程中曾发现并修正一处评测缺陷:微调评测的候选池(1000 张)与基线(5000 张)不一致,修正前微调 R@1 虚高至 58.3%,修正后为 38.6%。本文所有数字均基于修正后的统一协议。



\## 环境



Python 3.9+ · PyTorch(推理可用 CPU 版,微调需 GPU ≥12G)· transformers 4.57 · peft · faiss

