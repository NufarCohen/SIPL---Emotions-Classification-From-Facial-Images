import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import timm
from train_base_models_heterogeneous import load_data 

class FeatureExtractor(nn.Module):
    def __init__(self, model_name, num_classes=7):
        super().__init__()
        # num_classes=0 returns the global fetures
        self.net = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.feature_dim = self.net.num_features

    def forward(self, x):
        return self.net(x)

@torch.inference_mode()
def extract_features(loader, models, device):
    all_feats = []
    all_labels = []
    
    print(f"Extracting from {len(loader.dataset)} images...")
    for x, y in loader:
        x = x.to(device)
        
        batch_feats = []
        for model in models:
            f = model(x) # [Batch, Feature_Dim_i]
            batch_feats.append(f)
            
        concat_f = torch.cat(batch_feats, dim=1) # [Batch, Sum_Feature_Dims]
        
        all_feats.append(concat_f.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        
    return np.concatenate(all_feats), np.concatenate(all_labels)

if __name__ == "__main__":
    DATA_ROOT = r""
    MODELS_DIR = Path("./trained_hetero_models")
    FEATURE_DIR = Path("./hetero_features")
    FEATURE_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes, _ = load_data(DATA_ROOT, batch_size=16, seed=42, run_id=1)

    model_configs = ["convnext_xlarge", "tf_efficientnet_b7"]
    loaded_models = []
    
    total_dim = 0
    for name in model_configs:
        path = MODELS_DIR / f"{name}_trained.pt"
        
        model = FeatureExtractor(model_name=name)
        
        state_dict = torch.load(path, map_location='cpu')

        clean_state = {}
        for k, v in state_dict.items():
            if "classifier" not in k and "head" not in k and "fc" not in k:
                clean_state[k] = v
                
        missing, unexpected = model.load_state_dict(clean_state, strict=False)
        
        print(f"Loading {name}...")
        print(f"  Missing keys (should be empty/minimal): {missing}")
        model.to(device).eval()
        loaded_models.append(model)
        total_dim += model.feature_dim
        print(f"Loaded {name}. Dim: {model.feature_dim}")

    print(f"Total Concatenated Feature Dim: {total_dim}")

    train_feats, train_lbls = extract_features(train_loader, loaded_models, device)
    np.savez_compressed(FEATURE_DIR / "meta_train.npz", features=train_feats, labels=train_lbls)

    val_feats, val_lbls = extract_features(val_loader, loaded_models, device)
    np.savez_compressed(FEATURE_DIR / "meta_val.npz", features=val_feats, labels=val_lbls)
    
    test_feats, test_lbls = extract_features(test_loader, loaded_models, device)
    np.savez_compressed(FEATURE_DIR / "meta_test.npz", features=test_feats, labels=test_lbls)
    
    print("Done extracting features.")
