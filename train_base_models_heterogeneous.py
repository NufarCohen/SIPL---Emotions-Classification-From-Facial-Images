import math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import timm
from torch.cuda.amp import autocast, GradScaler
from torch_ema import ExponentialMovingAverage
import random, os
import shutil
from contextlib import nullcontext
from augmentation_data import create_augmentation

# ------------------------- Reproducibility -------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

# ------------------------- Data Loading -------------------------
def split_dataset(src_dir: Path, out_dir: Path, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    for class_dir in src_dir.iterdir():
        if not class_dir.is_dir(): continue
        class_name = class_dir.name
        imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in exts]
        random.shuffle(imgs)
        n_total = len(imgs)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test
        splits = {"train": imgs[:n_train], "val": imgs[n_train:n_train+n_val], "test": imgs[n_train+n_val:]}
        for split, files in splits.items():
            split_dir = out_dir / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in files: shutil.copy2(f, split_dir / f.name)

def load_data(data_dir, batch_size=64, seed=42, num_workers=4, run_id=0):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    img_size = 224

    base_split_dir = Path(f"/home/projects/sipl-prj10219/stacked_models/")
    dup_dir_path = base_split_dir / "dataset_cpy_notUniform"
    split_out = base_split_dir / "final_split_dataset"
    data_root_dir = Path(data_dir)
    
    if not split_out.exists():
        print(f"--- Preparing data for run {run_id} (seed={seed}) ---")
        if dup_dir_path.exists(): shutil.rmtree(dup_dir_path)
        shutil.copytree(data_root_dir, dup_dir_path)
        if split_out.exists(): shutil.rmtree(split_out)
        split_dataset(dup_dir_path, split_out, val_ratio=0.15, test_ratio=0.15, seed=seed)
        create_augmentation(split_out/"train", split_out/"train_augmented", target_count=20000)
    else:
        print(f"--- Using existing data split ---")

    t_common = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    train_dir = split_out / "train_augmented"
    val_dir   = split_out / "val"
    test_dir  = split_out / "test"

    ds_train = datasets.ImageFolder(root=train_dir, transform=t_common)
    ds_val   = datasets.ImageFolder(root=val_dir,   transform=t_common)
    ds_test  = datasets.ImageFolder(root=test_dir,  transform=t_common)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, len(ds_train.classes), 3

# ------------------------- Generic Model Wrapper -------------------------
class GenericModelFER(nn.Module):
    def __init__(self, model_name, num_classes=7, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

# ------------------------- Training Utils -------------------------
def cosine_lr(optimizer, base_lr, epoch, max_epochs, min_lr=1e-6):
    cos = 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
    lr = min_lr + (base_lr - min_lr) * cos
    for pg in optimizer.param_groups: pg['lr'] = lr
    return lr

def run_epoch(model, loader, loss_fn, optimizer=None, scaler=None, device="cuda", ema=None, train=True):
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0
    amp_ctx = autocast if (scaler is not None) else nullcontext
    grad_ctx = torch.enable_grad if train else torch.inference_mode

    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        if train: optimizer.zero_grad(set_to_none=True)
        with grad_ctx():
            with amp_ctx():
                logits = model(x)
                loss = loss_fn(logits, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None: ema.update()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total,1), total_correct / max(total, 1)

def train_specific_model(model_name, train_loader, val_loader, num_classes, save_path, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training {model_name}...")
    model = GenericModelFER(model_name=model_name, num_classes=num_classes).to(device)
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    best_acc = 0.0
    epochs = 5 #num of epoches

    for ep in range(1, epochs+1):
        lr = cosine_lr(optimizer, 1e-4, ep-1, epochs)
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, ema, train=True)
        with ema.average_parameters():
            va_loss, va_acc = run_epoch(model, val_loader, loss_fn, optimizer=None, scaler=None, device=device, ema=None, train=False)
        
        print(f"[{model_name}] Ep {ep}: TrAcc={tr_acc*100:.1f}%, ValAcc={va_acc*100:.1f}%")
        
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), save_path)

    print(f"Finished {model_name}. Best Acc: {best_acc*100:.2f}%")

# ------------------------- Main -------------------------
if __name__ == "__main__":
    DATA_ROOT = r""
    MODELS_DIR = Path("./trained_hetero_models")
    MODELS_DIR.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader, num_classes, _ = load_data(
        data_dir=DATA_ROOT, batch_size=8, seed=42, run_id=1
    )

    models_to_train = [
        {"name": "convnext_xlarge", "seed": 100},
        {"name": "tf_efficientnet_b7", "seed": 200}
    ]

    for m in models_to_train:
        save_path = MODELS_DIR / f"{m['name']}_trained.pt"
        train_specific_model(
            model_name=m['name'],
            train_loader=train_loader, 
            val_loader=val_loader, 
            num_classes=num_classes, 
            save_path=save_path,
            seed=m['seed']
        )

