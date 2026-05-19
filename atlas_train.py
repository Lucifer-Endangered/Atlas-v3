"""
ATLAS Phase 4 — GNN Training Loop
====================================

Trains the AtlasGNN model on the graph dataset built by atlas_dataset.py.

Features:
    - Class-weighted cross-entropy loss (handles No-Constraint class imbalance)
    - Train/validation split with per-epoch metrics
    - Confusion matrix generation + classification report
    - Model checkpoint saving (best validation F1)
    - MPS (Apple Silicon), CUDA, and CPU device support
    - Optional LLM-assisted hyperparameter suggestion via API

USAGE:
    python atlas_train.py --dataset ./dataset/atlas_dataset.pt \
                          --epochs 200 \
                          --lr 0.001 \
                          --hidden 128 \
                          --output ./checkpoints
"""

import os
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score, accuracy_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARNING] scikit-learn not installed. Metrics will be limited.")

from atlas_model import AtlasGNN


# =============================================================================
# CONFIGURATION
# =============================================================================

CONSTRAINT_NAMES = ["NoConstraint", "Mate", "Flush", "Insert", "Angle"]


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def get_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🖥️  Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


# =============================================================================
# CLASS WEIGHT COMPUTATION
# =============================================================================

def compute_class_weights(dataset: list, num_classes: int = 5) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for balanced training.
    Handles extreme imbalance between NoConstraint and actual constraints.
    """
    counts = torch.zeros(num_classes)
    for data in dataset:
        labels = data.target_labels if hasattr(data, 'target_labels') else torch.tensor(data["target_labels"])
        for c in range(num_classes):
            counts[c] += (labels == c).sum().item()

    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))

    # Cap maximum weight to prevent explosion on rare classes
    weights = weights.clamp(max=20.0)

    print(f"\n📊 Class distribution:")
    for i, name in enumerate(CONSTRAINT_NAMES):
        print(f"   {name:15s}: {int(counts[i]):>8d} samples  (weight: {weights[i]:.2f})")

    return weights


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: AtlasGNN,
    dataset: list,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch over all assembly graphs."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data in dataset:
        # Move to device
        if hasattr(data, 'to'):
            data = data.to(device)
            x = data.x
            edge_index = data.edge_index
            target_ei = data.target_edge_index
            labels = data.target_labels
        else:
            x = torch.tensor(data["node_features"], dtype=torch.float, device=device)
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            target_ei = torch.tensor(data["target_edges"], dtype=torch.long, device=device).t()
            labels = torch.tensor(data["target_labels"], dtype=torch.long, device=device)

        if target_ei.shape[1] == 0:
            continue

        optimizer.zero_grad()

        logits = model(x, edge_index, target_ei)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: AtlasGNN,
    dataset: list,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset split."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for data in dataset:
        if hasattr(data, 'to'):
            data = data.to(device)
            x = data.x
            edge_index = data.edge_index
            target_ei = data.target_edge_index
            labels = data.target_labels
        else:
            x = torch.tensor(data["node_features"], dtype=torch.float, device=device)
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            target_ei = torch.tensor(data["target_edges"], dtype=torch.long, device=device).t()
            labels = torch.tensor(data["target_labels"], dtype=torch.long, device=device)

        if target_ei.shape[1] == 0:
            continue

        logits = model(x, edge_index, target_ei)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if not all_preds:
        return 0.0, 0.0, 0.0, np.array([]), np.array([])

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(all_preds)
    accuracy = accuracy_score(all_labels, all_preds) if HAS_SKLEARN else (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if HAS_SKLEARN else 0.0

    return avg_loss, accuracy, f1, all_preds, all_labels


# =============================================================================
# LLM-ASSISTED HYPERPARAMETER SUGGESTION (OPTIONAL)
# =============================================================================

def suggest_hyperparameters_via_llm(
    dataset_stats: dict,
    api_key: str = None,
    model_name: str = "gpt-4o-mini",
) -> dict:
    """
    Optionally use an LLM API to suggest hyperparameters based on dataset stats.
    Falls back to sensible defaults if no API key is provided.
    """
    defaults = {
        "hidden_channels": 128,
        "learning_rate": 0.001,
        "dropout": 0.3,
        "neg_ratio": 5,
        "batch_size": 1,  # Per-graph training
        "epochs": 200,
    }

    if not api_key:
        return defaults

    try:
        import requests

        prompt = f"""Given a Graph Neural Network training task with these dataset statistics:
- Total assembly graphs: {dataset_stats.get('num_graphs', 0)}
- Average nodes per graph: {dataset_stats.get('avg_nodes', 0):.0f}
- Average positive edges: {dataset_stats.get('avg_pos_edges', 0):.0f}
- Average negative edges: {dataset_stats.get('avg_neg_edges', 0):.0f}
- Class imbalance ratio: {dataset_stats.get('imbalance_ratio', 0):.1f}
- Feature dimension: 21
- Number of classes: 5

Suggest optimal hyperparameters as JSON with keys:
hidden_channels, learning_rate, dropout, neg_ratio, epochs

Respond with ONLY the JSON object, no explanation."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 200,
        }

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=15,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            # Parse JSON from response
            import re
            match = re.search(r'\{[^}]+\}', content)
            if match:
                suggested = json.loads(match.group())
                defaults.update(suggested)
                print(f"🤖 LLM suggested hyperparameters: {suggested}")
    except Exception as e:
        print(f"  [LLM] Suggestion failed: {e}. Using defaults.")

    return defaults


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_pipeline(args):
    """Full training pipeline."""
    device = get_device()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\n📂 Loading dataset: {args.dataset}")
    dataset = torch.load(args.dataset, weights_only=False)
    print(f"   {len(dataset)} assembly graphs loaded")

    if len(dataset) == 0:
        print("❌ Empty dataset. Run atlas_dataset.py first.")
        return

    # ── Dataset statistics ────────────────────────────────────────────────────
    stats = {
        "num_graphs": len(dataset),
        "avg_nodes": np.mean([d.num_nodes if hasattr(d, 'num_nodes') else d["num_nodes"] for d in dataset]),
    }

    # ── LLM hyperparameter suggestion ────────────────────────────────────────
    hp = suggest_hyperparameters_via_llm(
        stats,
        api_key=args.llm_api_key if hasattr(args, 'llm_api_key') else None,
    )

    hidden = args.hidden or hp["hidden_channels"]
    lr = args.lr or hp["learning_rate"]
    dropout = args.dropout if args.dropout is not None else hp["dropout"]
    epochs = args.epochs or hp["epochs"]

    # ── Train/Val split (80/20) ───────────────────────────────────────────────
    n = len(dataset)
    split = max(1, int(n * 0.8))
    train_data = dataset[:split]
    val_data = dataset[split:] if split < n else dataset[-1:]

    print(f"   Train: {len(train_data)} graphs, Val: {len(val_data)} graphs")

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_data).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AtlasGNN(
        in_channels=21,
        hidden_channels=hidden,
        num_classes=5,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 AtlasGNN: {total_params:,} parameters")
    print(f"   Hidden: {hidden}, LR: {lr}, Dropout: {dropout}")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=15, verbose=True)

    # ── Training ──────────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    best_f1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>9} | {'Val Acc':>8} | {'Val F1':>7}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_data, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_data, criterion, device)

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            ckpt_path = os.path.join(args.output, "atlas_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "hidden_channels": hidden,
            }, ckpt_path)

        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch <= 5 or epoch == epochs:
            print(f"  {epoch:>4d}  | {train_loss:>11.4f} | {val_loss:>9.4f} | "
                  f"{val_acc:>7.1%} | {val_f1:>6.3f}  ({elapsed:.1f}s)")

    # ── Final Report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"   Checkpoint: {os.path.join(args.output, 'atlas_best.pt')}")

    # Classification report on best model
    if HAS_SKLEARN and len(val_preds) > 0:
        print(f"\n📋 Final Validation Report:")
        present_labels = sorted(set(val_labels.tolist()) | set(val_preds.tolist()))
        target_names = [CONSTRAINT_NAMES[i] for i in present_labels]
        print(classification_report(
            val_labels, val_preds,
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        ))

        cm = confusion_matrix(val_labels, val_preds, labels=present_labels)
        print("Confusion Matrix:")
        print(cm)

    # Save training history
    hist_path = os.path.join(args.output, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"📈 History saved: {hist_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ATLAS Phase 4 — Train GNN")
    parser.add_argument("--dataset", required=True, help="Path to atlas_dataset.pt")
    parser.add_argument("--output", default="./checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="OpenAI API key for LLM hyperparameter suggestions")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini",
                        help="LLM model name for suggestions")
    args = parser.parse_args()

    train_pipeline(args)


if __name__ == "__main__":
    main()
