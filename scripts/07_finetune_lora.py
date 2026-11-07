"""阶段4:LoRA 微调 CLIP(在 AutoDL 4090 上运行)"""
import csv, random, os, argparse
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model

MODEL = "openai/clip-vit-base-patch32"

class PairDS(Dataset):
    def __init__(self, rows, img_dir):
        self.rows, self.img_dir = rows, img_dir
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        im = Image.open(os.path.join(self.img_dir, self.rows[i]["file_name"])).convert("RGB")
        return self.rows[i]["caption"], im

def collate(batch, processor):
    caps = [c for c, _ in batch]; imgs = [im for _, im in batch]
    return processor(text=caps, images=imgs, return_tensors="pt", padding=True, truncation=True)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/root/autodl-tmp/data/processed/pairs_clean.csv")
ap.add_argument("--img_dir", default="/root/autodl-tmp/data/val2017")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--out", default="/root/autodl-tmp/lora_adapter")
args = ap.parse_args()

rows = [r for r in csv.DictReader(open(args.data, encoding="utf-8")) if r["split"] == "train"]
random.seed(42); random.shuffle(rows)
print(f"训练图文对: {len(rows)}")

processor = CLIPProcessor.from_pretrained(MODEL)
model = CLIPModel.from_pretrained(MODEL)
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                                         target_modules=["q_proj", "v_proj"]))
model.print_trainable_parameters()      # 应显示可训练参数 <1%
model.to("cuda").train()

dl = DataLoader(PairDS(rows, args.img_dir), batch_size=args.bs, shuffle=True,
                collate_fn=lambda b: collate(b, processor), num_workers=4)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dl)*args.epochs)

for ep in range(args.epochs):
    for step, batch in enumerate(dl, 1):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        loss = model(**batch, return_loss=True).loss     # CLIP 对比学习损失
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        if step % 50 == 0:
            print(f"epoch {ep} | step {step}/{len(dl)} | loss {loss.item():.4f}")

model.save_pretrained(args.out)
print("LoRA 适配器已保存:", args.out)