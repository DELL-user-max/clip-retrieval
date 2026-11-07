"""阶段3b:零样本分类 —— 不训练,用 80 句提示词给 5000 张图分类"""
import json, numpy as np, torch
from transformers import CLIPModel, CLIPProcessor

OUT = "data/processed"
img_feats = np.load(f"{OUT}/img_feats.npy")
img_names = open(f"{OUT}/img_names.txt").read().splitlines()

# 读取 COCO 80 类别 + 每张图的真实标签(来自你已下载的标注包)
coco = json.load(open("data/annotations/instances_val2017.json", encoding="utf-8"))
names = [c["name"] for c in coco["categories"]]
cat_id2name = {c["id"]: c["name"] for c in coco["categories"]}
img_id2name = {im["id"]: im["file_name"] for im in coco["images"]}
gt = {}
for a in coco["annotations"]:
    gt.setdefault(img_id2name[a["image_id"]], set()).add(cat_id2name[a["category_id"]])

# 提示词工程:"a photo of a xxx" 比"裸类名"准确率更高(CLIP 论文结论)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
with torch.no_grad():
    tf = model.get_text_features(**processor(text=[f"a photo of a {n}" for n in names],
                                             return_tensors="pt", padding=True)).numpy()
tf /= np.linalg.norm(tf, axis=1, keepdims=True)

sims = img_feats @ tf.T                     # 一次矩阵乘法 = 5000 图 × 80 类全部分类完!
ranked = np.argsort(-sims, axis=1)          # 对 80 类按相似度从高到低排序

hit5 = sum(1 for i, fn in enumerate(img_names)
           if fn in gt and gt[fn] & {names[j] for j in ranked[i, :5]})
hit1 = sum(1 for i, fn in enumerate(img_names)
           if fn in gt and names[ranked[i, 0]] in gt[fn])   # 第 1 名恰好正确
n = sum(1 for fn in img_names if fn in gt)
print(f"评测图片: {n} 张(COCO 80 类)")
print(f"Top-5 命中率: {100.0*hit5/n:.2f}%")
print(f"Top-1 命中率: {100.0*hit1/n:.2f}%  ← 严格口径的准确率")
strict = [(i, fn) for i, fn in enumerate(img_names) if fn in gt and len(gt[fn]) == 1]
hit_s = sum(1 for i, fn in strict if names[ranked[i, 0]] in gt[fn])
print(f"单类别图片 Top-1: {100.0*hit_s/len(strict):.2f}%  (共 {len(strict)} 张,最严格口径)")