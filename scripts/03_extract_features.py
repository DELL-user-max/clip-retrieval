"""阶段2:CLIP 特征抽取 —— 把所有图片和文本编码成 512 维向量"""
import csv, os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

MODEL = "openai/clip-vit-base-patch32"
IMG_DIR, PAIRS, OUT = "data/val2017", "data/processed/pairs_clean.csv", "data/processed"

model = CLIPModel.from_pretrained(MODEL).eval()
processor = CLIPProcessor.from_pretrained(MODEL)

with open(PAIRS, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
img_names = sorted({r["file_name"] for r in rows})
print(f"待编码:图片 {len(img_names)} 张 / 文本 {len(rows)} 条")

# ---- 图片编码(CPU 较慢,预计 15~30 分钟,挂着让它跑)----
img_feats = []
for i in tqdm(range(0, len(img_names), 32), desc="编码图片"):
    batch = [Image.open(os.path.join(IMG_DIR, n)).convert("RGB") for n in img_names[i:i+32]]
    with torch.no_grad():
        f = model.get_image_features(**processor(images=batch, return_tensors="pt"))
    img_feats.append(f.numpy())
img_feats = np.concatenate(img_feats).astype(np.float32)

# ---- 文本编码(快得多,几分钟)----
txt_feats = []
caps = [r["caption"] for r in rows]
for i in tqdm(range(0, len(caps), 128), desc="编码文本"):
    with torch.no_grad():
        f = model.get_text_features(**processor(text=caps[i:i+128], return_tensors="pt", padding=True, truncation=True))
    txt_feats.append(f.numpy())
txt_feats = np.concatenate(txt_feats).astype(np.float32)

# ---- L2 归一化:归一化后"向量内积"就等于"余弦相似度" ----
img_feats /= np.linalg.norm(img_feats, axis=1, keepdims=True)
txt_feats /= np.linalg.norm(txt_feats, axis=1, keepdims=True)

np.save(f"{OUT}/img_feats.npy", img_feats)
np.save(f"{OUT}/txt_feats.npy", txt_feats)
with open(f"{OUT}/img_names.txt", "w") as f:
    f.write("\n".join(img_names))
print("完成!图片特征:", img_feats.shape, "| 文本特征:", txt_feats.shape)