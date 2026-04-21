import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

class MetaLearner(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_correct = 0; total = 0
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            if is_train: optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            total_correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_correct / total, total_loss / len(loader)

if __name__ == "__main__":
    FEATURE_DIR = Path("./hetero_features")
    
    train_dat = np.load(FEATURE_DIR / "meta_train.npz")
    test_dat  = np.load(FEATURE_DIR / "meta_test.npz")
    val_dat   = np.load(FEATURE_DIR / "meta_val.npz")

    train_ds = FeatureDataset(train_dat['features'], train_dat['labels'])
    test_ds  = FeatureDataset(test_dat['features'],  test_dat['labels'])
    val_ds   = FeatureDataset(val_dat['features'],   val_dat['labels'])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    in_dim = train_ds.features.shape[1]
    num_cls = len(np.unique(train_ds.labels))
    print(f"Meta Model Input Dim: {in_dim}, Classes: {num_cls}")
    
    model = MetaLearner(in_dim, num_cls).cuda()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Training Meta Learner...")
    best_val_acc = 0
    for ep in range(50):
        tr_acc, tr_loss = run_epoch(model, train_loader, opt)
        val_acc, val_loss = run_epoch(model, val_loader, None)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc)
        print(f"Ep {ep+1}: Train Acc={tr_acc:.3f} | Val Acc={val_acc:.3f} (Best: {best_val_acc:.3f})")
    te_acc, te_loss= run_epoch(model, test_loader, None)   
    print(f"Final Heterogeneous Ensemble Accuracy: {te_acc*100:.2f}%")
