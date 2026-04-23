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
from torchvision import models as tv_models 

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

    
# ------------------------- Data Loading -------------------------
def load_data(
    data_dir,
    batch_size=64,
    seed=42,
    img_size=224, 
    num_workers=4
):
    """
    Loads the data once. All models in the ensemble must use the same image size.
    We will use standard ImageNet normalization.
    """
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    
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
    create_augmentation(split_out/"train", split_out/"train_augmented", target_count=20000)
   
    t_common = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    ])
    t_train = t_eval = t_common
    in_chans = 3
    
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


# ------------------------- Model Wrappers -------------------------

class EfficientFER(nn.Module):
  
    def __init__(self, variant="tf_efficientnetv2_l", num_classes=7, in_chans=3, pretrained=True):
        super().__init__()
        self.net = timm.create_model(variant, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

class VGG19FER(nn.Module):
    
    def __init__(self, num_classes=7, in_chans=3, pretrained=True):
        super().__init__()
        self.net = tv_models.vgg19_bn(weights=tv_models.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = self.net.classifier[-1].in_features
        self.net.classifier[-1] = nn.Linear(in_feats, num_classes)
        
        if in_chans == 1:
            first_conv = self.net.features[0]
            self.net.features[0] = nn.Conv2d(1, first_conv.out_channels, 
                                             kernel_size=first_conv.kernel_size, 
                                             stride=first_conv.stride, 
                                             padding=first_conv.padding, 
                                             bias=False)
            if pretrained:
                self.net.features[0].weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)

    def forward(self, x):
        return self.net(x)

class ConvNeXtFER(nn.Module):
    """
    ConvNeXt wrapper:
    """
    def __init__(self, num_classes=7, in_chans=3, pretrained=True, model_name="convnext_base"):
        super().__init__()
        self.model_name = model_name

        use_timm = model_name.startswith("convnext_xlarge") or model_name.startswith("convnextv2") \
                   or model_name not in {"convnext_tiny","convnext_small","convnext_base","convnext_large"}

        if use_timm:
            self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
        else:
            if model_name == "convnext_tiny":
                weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
                self.net = tv_models.convnext_tiny(weights=weights)
            elif model_name == "convnext_small":
                weights = tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
                self.net = tv_models.convnext_small(weights=weights)
            elif model_name == "convnext_large":
                weights = tv_models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
                self.net = tv_models.convnext_large(weights=weights)
            else:  
                weights = tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
                self.net = tv_models.convnext_base(weights=weights)

            in_feats = self.net.classifier[-1].in_features
            self.net.classifier[-1] = nn.Linear(in_feats, num_classes)
            
            if in_chans == 1:
                first_layer = self.net.features[0][0]
                self.net.features[0][0] = nn.Conv2d(1, first_layer.out_channels,
                                                    kernel_size=first_layer.kernel_size,
                                                    stride=first_layer.stride,
                                                    padding=first_layer.padding,
                                                    bias=False)
                if pretrained:
                     self.net.features[0][0].weight.data = first_layer.weight.data.mean(dim=1, keepdim=True)


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
        
        if return_preds:
            all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    
    acc = total_correct / max(total, 1)
    f1  = macro_f1_from_preds(all_preds, all_labels, num_classes=int(logits.size(1))) if total > 0 else 0.0

    if return_preds:
        all_probs = np.concatenate(all_probs) if all_probs else np.array([]) 
        return total_loss / max(total,1), acc, f1, all_preds, all_labels, all_probs
    else:
        return total_loss / max(total,1), acc, f1


# ------------------------- Train Loop -------------------------
def train_model(
    train_loader, val_loader, test_loader, num_classes, in_chans,
    variant, 
    epochs=10, 
    base_lr=1e-4,
    weight_decay=0.05,
    label_smoothing=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "efficientnet" in variant:
        model = EfficientFER(variant=variant, num_classes=num_classes, in_chans=in_chans, pretrained=True).to(device)
    elif "vgg19" in variant:
        model = VGG19FER(num_classes=num_classes, in_chans=in_chans, pretrained=True).to(device)
    elif "convnext" in variant:
        model = ConvNeXtFER(model_name=variant, num_classes=num_classes, in_chans=in_chans, pretrained=True).to(device)
    else:
        raise ValueError(f"Model variant '{variant}' not recognized in train_model function.")

    # Loss / Optim / AMP / EMA
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    best_val_acc, best_state, epochs_no_improve = 0.0, None, 0
    
    for ep in range(1, epochs+1):
        lr_now = cosine_lr(optimizer, base_lr, ep-1, epochs, min_lr=1e-6)
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, ema, train=True)
        
        with ema.average_parameters():
            va_loss, va_acc, va_f1 = run_epoch(model, val_loader, loss_fn, optimizer=None, scaler=None, device=device, ema=None, train=False)
        
        dt = time.time()-t0
        print(f"Epoch {ep:03d}/{epochs} | lr={lr_now:.2e} | "
              f"train: loss={tr_loss:.4f}, acc={tr_acc*100:.2f}% | "
              f"val: loss={va_loss:.4f}, acc={va_acc*100:.2f}% | {dt:.1f}s")

        if va_acc > best_val_acc:
             best_val_acc = va_acc
             # Save the best model state
             best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Check on Test Set with the best model
    with ema.average_parameters():
        te_loss, te_acc, te_f1, te_preds, te_labels, te_probs = run_epoch(model, test_loader, loss_fn, optimizer=None, scaler=None, device=device, ema=None, train=False, return_preds=True)
    
    print(f"--- TEST Result for {variant} ---")
    print(f"TEST: loss={te_loss:.4f}, acc={te_acc*100:.2f}%, f1={te_f1:.3f}")
    print("-" * (28 + len(variant)))

    return te_loss, te_acc, te_f1, te_preds, te_labels, te_probs

def save_graphs(train_loss, val_loss, train_acc, val_acc, out_path):
    pass 

# ------------------------- Ensemble Voting Runner -------------------------
def run_soft_voting(data_root, model_variants_list, batch=64):
    
    train_loader, val_loader, test_loader, num_classes, in_chans = load_data(
        data_root, batch_size=batch, img_size=224
    )

    all_probs_runs = []
    all_preds_runs = [] 
    all_labels = None

    # --- Loop over the different models ---
    for variant_name in model_variants_list:
        print(f"\n{'='*20} TRAINING MODEL: {variant_name} {'='*20}\n")
        
        te_loss, te_acc, te_f1, te_preds, te_labels, te_probs = train_model(
            train_loader, val_loader, test_loader,
            num_classes, in_chans,
            variant=variant_name,
            epochs=5, 
            base_lr=1e-4,
        )
        
        all_probs_runs.append(te_probs)
        all_preds_runs.append(te_preds) 
        
        if all_labels is None:
            all_labels = te_labels

    # --- Voting Logic ---

    # Soft Voting 
    all_probs_stack = np.stack(all_probs_runs, axis=1)
    average_probs = all_probs_stack.mean(axis=1) 
    soft_vote_preds = average_probs.argmax(axis=1)
    soft_vote_acc = (soft_vote_preds == all_labels).mean()
    print(f"\n{'='*20} ENSEMBLE RESULTS {'='*20}")
    print(f"Soft-Vote (Averaging Probabilities) Accuracy: {soft_vote_acc*100:.2f}%")

    return soft_vote_preds, all_labels


# ------------------------- CLI Runner -------------------------
if __name__ == "__main__":
    
    model_list = [
        "convnext_xlarge",            # model 1
        "convnext_xlarge",            # model 2 
        "convnext_xlarge",            # model 3
        "convnext_xlarge",            # model 4
        "convnext_xlarge"             # model 5 
    ]

    batch_size=16
    data_root = r""

    print(f"Starting Ensemble Vote for {len(model_list)} models...")
    print(f"Models: {model_list}")
    print(f"Batch Size: {batch_size}\n")
    
    run_soft_voting(
        data_root=data_root,
        model_variants_list=model_list,
        batch=batch_size
    )