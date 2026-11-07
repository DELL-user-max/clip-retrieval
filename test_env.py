import torch, transformers, faiss
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)

from transformers import CLIPModel, CLIPProcessor
print("正在下载/加载 CLIP 模型(首次约 600MB,请耐心)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP 加载成功, 参数量:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# 冒烟测试:零样本打分
from PIL import Image
img = Image.new("RGB", (224, 224), color=(128, 200, 128))  # 纯绿色方块图
inputs = processor(text=["a green square", "a red circle"], images=img, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
probs = out.logits_per_image.softmax(dim=1)
print("零样本打分 [绿方块, 红圆]:", probs.numpy())