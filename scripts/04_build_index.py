"""阶段2b:建 FAISS 索引 + 检索自测"""
import csv, faiss, numpy as np

OUT = "data/processed"
img_feats = np.load(f"{OUT}/img_feats.npy").astype(np.float32)
txt_feats = np.load(f"{OUT}/txt_feats.npy").astype(np.float32)
img_names = open(f"{OUT}/img_names.txt").read().splitlines()
rows = list(csv.DictReader(open(f"{OUT}/pairs_clean.csv", encoding="utf-8")))

index = faiss.IndexFlatIP(img_feats.shape[1])   # 内积索引(向量已归一化=余弦相似度)
index.add(img_feats)
faiss.write_index(index, f"{OUT}/img.index")
print(f"索引建好: {index.ntotal} 张图 / {img_feats.shape[1]} 维")

# 自测:拿验证集第一条描述来搜图
D, I = index.search(txt_feats[0][None, :], k=5)
print("\n查询文本:", rows[0]["caption"])
print("描述对应的原图:", rows[0]["file_name"])
print("Top-5 检索结果:")
for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
    hit = "  ★命中原图!" if img_names[idx] == rows[0]["file_name"] else ""
    print(f"  {rank}. {img_names[idx]}  相似度={score:.3f}{hit}")