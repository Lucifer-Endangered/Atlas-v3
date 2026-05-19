"""
ATLAS V2 — Constraint Prediction Model + Training
===================================================
MLP classifier: given two entity feature vectors (30-dim combined),
predict the constraint type (None/Mate/Flush/Insert/Angle/Tangent).
"""

import json, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

CONSTRAINT_NAMES = ["None", "Mate", "Flush", "Insert", "Angle", "Tangent"]

# =============================================================================
# MODEL
# =============================================================================

class AtlasV2Model(nn.Module):
    """MLP constraint classifier with symmetric feature processing."""
    
    def __init__(self, entity_dim=15, hidden=256, num_classes=6, dropout=0.3):
        super().__init__()
        self.entity_dim = entity_dim
        
        # Shared entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        
        # Classifier on combined features: [enc1, enc2, |enc1-enc2|, enc1*enc2]
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: [B, 30] = [entity_one(15) || entity_two(15)]
        e1 = x[:, :self.entity_dim]
        e2 = x[:, self.entity_dim:]
        
        h1 = self.entity_encoder(e1)
        h2 = self.entity_encoder(e2)
        
        combined = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=-1)
        return self.classifier(combined)

# =============================================================================
# TRAINING
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train(args):
    device = get_device()
    print(f"🖥️  Device: {device}")
    
    # Load data
    X = np.load(os.path.join(args.data_dir, "X.npy"))
    y = np.load(os.path.join(args.data_dir, "y.npy"))
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    
    print(f"📂 Data: {X.shape[0]} samples, {meta['feature_dim']}-dim features")
    
    # Normalize features (per-column)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    
    # Class weights
    counts = np.bincount(y_train, minlength=meta["num_classes"])
    weights = len(y_train) / (meta["num_classes"] * np.maximum(counts, 1))
    weights = np.clip(weights, 0.5, 15.0)
    print(f"\n📊 Class distribution (train):")
    for i, name in enumerate(CONSTRAINT_NAMES[:meta["num_classes"]]):
        print(f"   {name:12s}: {counts[i]:>6d}  (weight: {weights[i]:.2f})")
    
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Model
    model = AtlasV2Model(
        entity_dim=meta["entity_dim"],
        hidden=args.hidden,
        num_classes=meta["num_classes"],
        dropout=args.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {total_params:,} parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15
    )
    
    os.makedirs(args.output, exist_ok=True)
    
    # Save normalization params for inference
    np.save(os.path.join(args.output, "norm_mean.npy"), mean)
    np.save(os.path.join(args.output, "norm_std.npy"), std)
    
    best_f1 = 0.0
    history = {"train_loss": [], "val_f1": [], "val_acc": []}
    
    print(f"\n{'Epoch':>6} | {'Loss':>10} | {'Val Acc':>8} | {'Val F1':>7}")
    print("=" * 50)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
        avg_loss = total_loss / len(y_train)
        
        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                preds = model(xb).argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_labels.append(yb)
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_acc = (all_preds == all_labels).mean()
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        
        scheduler.step(val_f1)
        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_f1": val_f1, "val_acc": val_acc,
                "hidden": args.hidden, "entity_dim": meta["entity_dim"],
                "num_classes": meta["num_classes"],
            }, os.path.join(args.output, "atlas_v2_best.pt"))
        
        if epoch % 10 == 0 or epoch <= 5 or epoch == args.epochs:
            print(f"  {epoch:>4d}  | {avg_loss:>10.4f} | {val_acc:>7.1%} | {val_f1:>6.3f}")
    
    print(f"\n✅ Best F1: {best_f1:.4f}")
    print(f"\n📋 Final Classification Report:")
    present = sorted(set(all_labels) | set(all_preds))
    names = [CONSTRAINT_NAMES[i] for i in present]
    print(classification_report(all_labels, all_preds, labels=present,
                                target_names=names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds, labels=present))
    
    with open(os.path.join(args.output, "history.json"), "w") as f:
        json.dump(history, f)
    print(f"\n💾 Saved to {args.output}/")


def main():
    p = argparse.ArgumentParser(description="ATLAS V2 — Train constraint predictor")
    p.add_argument("--data-dir", default="./v2_dataset")
    p.add_argument("--output", default="./v2_checkpoints")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=256)
    train(p.parse_args())

if __name__ == "__main__":
    main()
