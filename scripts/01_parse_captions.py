"""解析 COCO captions:把 JSON 变成整洁的 图片-描述 对照表"""
import json, csv, os

with open("data/annotations/captions_val2017.json", encoding="utf-8") as f:
    coco = json.load(f)

# id -> 文件名 映射
id2name = {img["id"]: img["file_name"] for img in coco["images"]}

# 展开成 (文件名, caption) 平表
rows = [(id2name[a["image_id"]], a["caption"].strip()) for a in coco["annotations"]]

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/captions_raw.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file_name", "caption"])
    w.writerows(rows)

# 数据体检
lens = [len(c.split()) for _, c in rows]
print(f"图片数: {len(id2name)}")
print(f"图文对数: {len(rows)}  (平均每图 {len(rows)/len(id2name):.1f} 条描述)")
print(f"caption 词数: 最短 {min(lens)} / 平均 {sum(lens)/len(lens):.1f} / 最长 {max(lens)}")