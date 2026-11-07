"""数据清洗:坏图过滤 -> 文本规范化 -> 长度过滤 -> 去重 -> 按图划分训练/验证"""
import csv, os
from PIL import Image
from tqdm import tqdm

RAW, OUT = "data/processed/captions_raw.csv", "data/processed/pairs_clean.csv"
IMG_DIR = "data/val2017"
MIN_WORDS, MAX_WORDS = 3, 32   # CLIP 文本编码器上限 77 token,32 词以内基本不会被截断

with open(RAW, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
n_raw, n_bad_img, n_len, n_dup = len(rows), 0, 0, 0

# ---------- 1) 坏图过滤:每张图只检测一次(5000 次,而不是 25 万次) ----------
ok = {}
for fn in tqdm(sorted({r["file_name"] for r in rows}), desc="坏图检测"):
    try:
        with Image.open(os.path.join(IMG_DIR, fn)) as im:
            im.verify()                    # 快速校验文件头
        with Image.open(os.path.join(IMG_DIR, fn)) as im:
            im.convert("RGB").load()       # 强制完整解码,揪出截断/损坏的图
        ok[fn] = True
    except Exception:
        ok[fn] = False

# ---------- 2) 文本规范化 + 长度过滤 + 去重 ----------
seen, clean = set(), []
for r in rows:
    fn, cap = r["file_name"], " ".join(r["caption"].split()).strip()
    if not ok[fn]:
        n_bad_img += 1
        continue
    if not (MIN_WORDS <= len(cap.split()) <= MAX_WORDS):
        n_len += 1
        continue
    key = (fn, cap.lower())                # 大小写不同但内容相同的也算重复
    if key in seen:
        n_dup += 1
        continue
    seen.add(key)
    clean.append((fn, cap))

# ---------- 3) 按图片划分训练/验证(防数据泄漏的关键!) ----------
img_names = sorted({fn for fn, _ in clean})
val_imgs = set(img_names[: len(img_names) // 5])   # 20% 的图片作为验证集

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file_name", "caption", "split"])
    for fn, cap in clean:
        w.writerow([fn, cap, "val" if fn in val_imgs else "train"])

print(f"原始图文对:   {n_raw}")
print(f"坏图剔除:     {n_bad_img}")
print(f"长度越界剔除: {n_len}")
print(f"重复剔除:     {n_dup}")
print(f"清洗后保留:   {len(clean)}  (训练 {len(clean)-sum(1 for fn,_ in clean if fn in val_imgs)} / 验证 {sum(1 for fn,_ in clean if fn in val_imgs)})")