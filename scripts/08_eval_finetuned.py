"""阶段4b:加载 LoRA 适配器重测 Recall@K,与基线对比"""
import csv, os, argparse,  numpy as np, torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="/root/autodl-tmp/lora_adapter")
ap.add_argument("--data", default="/root/autodl-tmp/data/processed/pairs_clean.csv")
ap.add_argument("--img_dir", default="/root/autodl-tmp/data/val2017")
args = ap.parse_args()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = PeftModel.from_pretrained(CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                                  args.adapter).to("cuda").eval()

rows = [r for r in csv.DictReader(open(args.data, encoding="utf-8")) if r["split"] == "val"]
img_names = sorted({r["file_name"] for r in rows})

img_feats = []
for i in tqdm(range(0, len(img_names), 64), desc="编码图片"):
    b = [Image.open(os.path.join(args.img_dir, n)).convert("RGB") for n in img_names[i:i+64]]
    with torch.no_grad():
        f = model.get_image_features(**processor(images=b, return_tensors="pt"))
    img_feats.append(f.cpu().numpy())
img_feats = np.concatenate(img_feats).astype(np.float32)
img_feats /= np.linalg.norm(img_feats, axis=1, keepdims=True)

txt_feats = []
for i in tqdm(range(0, len(rows), 128), desc="编码文本"):
    caps = [r["caption"] for r in rows[i:i+128]]
    with torch.no_grad():
        f = model.get_text_features(**processor(text=caps, return_tensors="pt", padding=True, truncation=True))
    txt_feats.append(f.cpu().numpy())
txt_feats = np.concatenate(txt_feats).astype(np.float32)
txt_feats /= np.linalg.norm(txt_feats, axis=1, keepdims=True)

sims = txt_feats @ img_feats.T
name2idx = {n: i for i, n in enumerate(img_names)}
gt = np.array([name2idx[r["file_name"]] for r in rows])
rank = np.argsort(-sims, axis=1)
for k in (1, 5, 10):
    r = np.mean([gt[q] in rank[q, :k] for q in range(len(rows))]) * 100
    print(f"微调后 Recall@{k}: {r:.2f}%")
print("基线(未微调): R@1 32.08 / R@5 56.78 / R@10 68.28")