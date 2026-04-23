import math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import timm
from torch.cuda.amp import autocast, GradScaler
from torch_ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from augmentation_data import create_augmentation
import cv2
import random, os
import shutil
from contextlib import nullcontext
from collections import Counter

# ------------------------- Reproducibility -------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
set_seed(42)


def split_dataset(src_dir: Path, out_dir: Path, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split images per class into train/val/test folders with given ratios.
    """
    random.seed(seed)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    for class_dir in src_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in exts]
        random.shuffle(imgs)

        n_total = len(imgs)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test

        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:]
        }
        for split, files in splits.items():
            split_dir = out_dir / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, split_dir / f.name)

 
def load_data(
    data_dir,
    variant='tf_efficientnet_b0_ns',
    mode='rgb224',
    batch_size=64,
    seed=42,
    num_workers=4
):
    """
    One-folder dataset with class subfolders. Splits 70/15/15 in-memory.
    Returns:
        train_loader, val_loader, test_loader, num_classes, in_chans
    """
    # Read default config from a pretrained model (for Option A transforms)
    tmp = timm.create_model(variant, pretrained=True)
    mean = tmp.default_cfg.get('mean', (0.485, 0.456, 0.406))
    std  = tmp.default_cfg.get('std',  (0.229, 0.224, 0.225))
    img_size = tmp.default_cfg.get('input_size', (3, 224, 224))[1]
    del tmp
   
    dup_dir_path=r"./dataset_cpy_notUniform"
    split_out = Path(r"./final_split_dataset")
    dup_dir = Path(dup_dir_path)
    data_root_dir = Path(data_dir)
    if dup_dir.exists():
        shutil.rmtree(dup_dir)
    shutil.copytree(data_root_dir, dup_dir)

    if split_out.exists():
        shutil.rmtree(split_out)
    split_dataset(dup_dir, split_out, val_ratio=0.15, test_ratio=0.15)
    #augmentataion on the train set
    create_augmentation(split_out/"train", split_out/"train_augmented", target_count=20000)
   
    if mode == 'rgb224':
        t_common = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),                        
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),  # img_size=224
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        ])
        t_train = t_eval = t_common
        in_chans = 3
    else:
        raise ValueError("mode must be 'rgb224' or 'gray48'")
    
    train_dir = split_out / "train_augmented"
    val_dir   = split_out / "val"
    test_dir  = split_out / "test"

    ds_train = datasets.ImageFolder(root=train_dir, transform=t_train)
    ds_val   = datasets.ImageFolder(root=val_dir,   transform=t_eval)
    ds_test  = datasets.ImageFolder(root=test_dir,  transform=t_eval)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    num_classes = len(ds_train.classes)
    return train_loader, val_loader, test_loader, num_classes, in_chans

# ------------------------- Metrics -------------------------
def macro_f1_from_preds(all_preds, all_labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(all_preds, all_labels):
        cm[t, p] += 1
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1= tp/ (tp + fp + fn + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


# ------------------------- Model Wrapper -------------------------
class ConvNeXtFER(nn.Module):
    """
    ConvNeXt wrapper:
      - TorchVision: tiny/small/base/large
      - timm: xlarge (and others)
    """
    def __init__(self, num_classes=7, in_chans=3, pretrained=True, model_name="convnext_base"):
        super().__init__()
        self.model_name = model_name

        # Use timm for xlarge (or any name not covered by torchvision)
        use_timm = model_name.startswith("convnext_xlarge") or model_name.startswith("convnextv2") \
                   or model_name not in {"convnext_tiny","convnext_small","convnext_base","convnext_large"}

        if use_timm:
            # e.g. 'convnext_xlarge' (pretrained on ImageNet-22k / 12k variants inside timm registry)
            import timm
            self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        else:
            from torchvision import models as tv_models
            if model_name == "convnext_tiny":
                weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
                self.net = tv_models.convnext_tiny(weights=weights)
            elif model_name == "convnext_small":
                weights = tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
                self.net = tv_models.convnext_small(weights=weights)
            elif model_name == "convnext_large":
                weights = tv_models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
                self.net = tv_models.convnext_large(weights=weights)
            else:  # base
                weights = tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
                self.net = tv_models.convnext_base(weights=weights)

            # swap classifier for TorchVision models
            in_feats = self.net.classifier[-1].in_features
            self.net.classifier[-1] = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.net(x)

# ------------------------- Training Utils -------------------------
def cosine_lr(optimizer, base_lr, epoch, max_epochs, min_lr=1e-6):
    cos = 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
    lr = min_lr + (base_lr - min_lr) * cos
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def run_epoch(model, loader, loss_fn, optimizer=None, scaler=None, device="cuda",
              ema=None, train=True, return_preds=False, amp_eval=True):
    model.train(train)

    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []  
    # AMP on if training (scaler!=None) OR if we allow AMP during eval
    use_amp = (scaler is not None) or (not train and amp_eval and device == "cuda")
    amp_ctx = autocast if use_amp else nullcontext
    grad_ctx = torch.enable_grad if train else torch.inference_mode

    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with grad_ctx():
            with amp_ctx():
                logits = model(x)
                loss = loss_fn(logits, y)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        total_correct += (pred == y).sum().item()
        total += y.size(0)
        all_preds.append(pred.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    all_probs = np.concatenate(all_probs) if all_probs else np.array([])
    acc = total_correct / max(total, 1)
    f1  = macro_f1_from_preds(all_preds, all_labels, num_classes=int(logits.size(1))) if total > 0 else 0.0

    if return_preds:
        return total_loss / max(total,1), acc, f1, all_preds, all_labels, all_probs
    else:
        return total_loss / max(total,1), acc, f1


# ------------------------- Train Loop -------------------------
def train_efficientfer(
    train_loader, val_loader, test_loader, num_classes, in_chans,
    variant,
    epochs=30,
    base_lr=1e-4,
    batch_size=128,
    weight_decay=0.05,
    num_workers=4,
    label_smoothing=0.1,
    early_stop_patience=7,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Model: ConvNeXt
    model = ConvNeXtFER(
        num_classes=num_classes,
        in_chans=in_chans,
        pretrained=True,
        model_name=variant
    ).to(device)

    # Loss / Optim / AMP / EMA
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    best_val_acc, best_state, epochs_no_improve = 0.0, None, 0
    val_loss_history = []        # Validation loss
    train_loss_history = []            # Store Training loss per epoch  
    val_acc_history = []         # Validation accuracy (classification only)
    train_acc_history = []       # Training accuracy
    for ep in range(1, epochs+1):
        lr_now = cosine_lr(optimizer, base_lr, ep-1, epochs, min_lr=1e-6)
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, ema, train=True)
        train_loss_history.append(tr_loss)
        train_acc_history.append(tr_acc)
        with ema.average_parameters():
            va_loss, va_acc, va_f1 = run_epoch(model, val_loader, loss_fn, optimizer=None, scaler=None, device=device, ema=None, train=False)
        val_acc_history.append(va_acc)
        val_loss_history.append(va_loss)
        dt = time.time()-t0
        print(f"Epoch {ep:03d}/{epochs} | lr={lr_now:.2e} | "
              f"train: loss={tr_loss:.4f}, acc={tr_acc*100:.2f}%, f1={tr_f1:.3f} | "
              f"val: loss={va_loss:.4f}, acc={va_acc*100:.2f}%, f1={va_f1:.3f} | {dt:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    with ema.average_parameters():
        te_loss, te_acc, te_f1, te_preds, te_labels, te_probs = run_epoch(model, test_loader, loss_fn, optimizer=None, scaler=None, device=device, ema=None, train=False, return_preds=True)
    print(f"TEST: loss={te_loss:.4f}, acc={te_acc*100:.2f}%, f1={te_f1:.3f}")

    return te_loss, te_acc, te_f1, te_preds, te_labels, te_probs

def run_multiple_fixed(data_root, variant="tf_efficientnet_b0_ns", runs=3, batch=64):
    # load data only once
    train_loader, val_loader, test_loader, num_classes, in_chans = load_data(
        data_root, variant=variant, mode="rgb224", batch_size=batch
    )

    all_preds_runs = []
    all_probs_runs = []
    all_labels = None

    for i in range(runs):
        print(f"\n=== Run {i+1}/{runs} ===\n")
        te_loss, te_acc, te_f1, te_preds, te_labels, te_probs = train_efficientfer(
            train_loader, val_loader, test_loader,
            num_classes, in_chans,
            variant=variant, epochs=5, base_lr=1e-5, batch_size=batch
        )
        all_preds_runs.append(te_preds)
        all_probs_runs.append(te_probs)
        if all_labels is None:
            all_labels = te_labels

    # majority vote
    # Shape: [N_samples, runs]
    all_preds_runs = np.stack(all_preds_runs, axis=1)

    # Shape: [N_samples, runs, N_classes]
    all_probs_runs = np.stack(all_probs_runs, axis=1)
    final_preds = []
    
    # go over each sample
    for i in range(len(all_labels)):
        pred_row = all_preds_runs[i]  # strong predictions
        prob_row = all_probs_runs[i]  # (Shape: [runs, N_classes])
        
        counts = Counter(pred_row)
        most_common_list = counts.most_common()

        top_count = most_common_list[0][1] 
        is_tie = len(most_common_list) > 1 and most_common_list[1][1] == top_count

        if not is_tie:
            final_pred = most_common_list[0][0]
        else:
            tied_labels = {label for label, count in most_common_list if count == top_count}
            
            best_prob = -1.0
            final_pred = -1
            
            for run_idx, label in enumerate(pred_row):
                if label in tied_labels:
                    prob = prob_row[run_idx, label]
                    
                    if prob > best_prob:
                        best_prob = prob
                        final_pred = label
            
        final_preds.append(final_pred)

    final_preds = np.array(final_preds)
    acc = (final_preds == all_labels).mean()
    print(f"\nMajority-vote Accuracy: {acc*100:.2f}%")
    return final_preds, all_labels


if __name__ == "__main__":
    variant = "convnext_xlarge"  # 'convnext_tiny' | 'convnext_small' | 'convnext_base' | 'convnext_large'
    batch_size=16
    print(print(f"Model version is {variant} with batch size {batch_size}\n"))
    run_multiple_fixed(
        data_root=r"",
        variant=variant,
        runs=5,
        batch=batch_size
    )