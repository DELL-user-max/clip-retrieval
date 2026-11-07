"""阶段3:在全部验证图文对上计算 Recall@1/5/10"""
import csv, faiss, numpy as np

OUT = "data/processed"
img_feats = np.load(f"{OUT}/img_feats.npy").astype(np.float32)
txt_feats = np.load(f"{OUT}/txt_feats.npy").astype(np.float32)
img_names = open(f"{OUT}/img_names.txt").read().splitlines()
rows = list(csv.DictReader(open(f"{OUT}/pairs_clean.csv", encoding="utf-8")))
name2idx = {n: i for i, n in enumerate(img_names)}

# 只评验证集(训练集的图模型间接"见过",评了会虚高——数据泄漏!)
val_pos = [i for i, r in enumerate(rows) if r["split"] == "val"]
val_txt = txt_feats[val_pos]
gt = np.array([name2idx[rows[i]["file_name"]] for i in val_pos])

index = faiss.read_index(f"{OUT}/img.index")
D, I = index.search(val_txt, k=10)     # 5000 条查询一次完成,FAISS 毫秒级

print(f"评测规模: {len(val_pos)} 条验证描述 × 5000 张候选图")
for k in (1, 5, 10):
    hits = sum(1 for q in range(len(val_pos)) if gt[q] in I[q, :k])
    print(f"Recall@{k}: {100.0*hits/len(val_pos):.2f}%")
print("\n参考:CLIP 论文中同款模型(ViT-B/32)零样本文搜图")
print("     R@1 约 31%,R@5 约 55%,R@10 约 66% —— 你的数字应在此附近")