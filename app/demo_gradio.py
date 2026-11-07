"""以文搜图 Web 演示:输入任意英文描述 → 返回最匹配的图片"""
import csv, os
import gradio as gr
import faiss, numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

OUT, IMG_DIR = "data/processed", "data/val2017"
img_feats = np.load(f"{OUT}/img_feats.npy").astype(np.float32)
img_names = open(f"{OUT}/img_names.txt").read().splitlines()
index = faiss.read_index(f"{OUT}/img.index")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

def search(query, topk):
    with torch.no_grad():
        tf = model.get_text_features(**processor(text=[query], return_tensors="pt")).numpy()[0]
    tf /= np.linalg.norm(tf)
    D, I = index.search(tf[None, :], k=int(topk))
    return [(os.path.join(IMG_DIR, img_names[i]), f"{img_names[i]} | 相似度 {s:.3f}")
            for s, i in zip(D[0], I[0])]

demo = gr.Interface(
    fn=search,
    inputs=[gr.Textbox(label="图片描述(英文效果最佳)", value="a dog running on the grass"),
            gr.Slider(1, 50, value=9, step=1, label="返回图片数")],
    outputs=gr.Gallery(label="检索结果", columns=3),
    title="CLIP 图文检索系统 · 项目A",
    description="输入描述,在 5000 张 COCO 图片中检索最匹配的图",
)
demo.launch()