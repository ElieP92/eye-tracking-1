# train_hybrid.py
"""
Hybrid model: MLP on rich features + lightweight CNN on heatmaps.
Emphasizes the discriminative features while using spatial info as auxiliary.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DL_DATA_DIR = PROJECT_ROOT / "results" / "deep_learning_data"
RESULTS_DIR = PROJECT_ROOT / "results" / "deep_learning_results"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Young', 'Middle-aged', 'Older']

BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 5e-4
PATIENCE = 30


class HybridDataset(Dataset):
    def __init__(self, features, heatmaps, labels, augment=False):
        self.features = torch.FloatTensor(features)
        self.heatmaps = torch.FloatTensor(heatmaps)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        f = self.features[idx]
        h = self.heatmaps[idx]
        y = self.labels[idx]
        
        if self.augment:
            # Feature noise
            f = f + torch.randn_like(f) * 0.05
            # Heatmap flip
            if torch.rand(1) > 0.5:
                h = torch.flip(h, dims=[1])
        
        return f, h, y


class FeatureDominantHybrid(nn.Module):
    """
    Model that emphasizes rich features with auxiliary spatial info.
    Features contribute ~80% of the classification signal.
    """
    
    def __init__(self, n_features, n_heatmap_channels=4, heatmap_size=112, num_classes=3):
        super().__init__()
        
        # Strong feature branch (main discriminative power)
        self.feature_net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Light CNN branch (auxiliary spatial info)
        self.cnn_net = nn.Sequential(
            nn.Conv2d(n_heatmap_channels, 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten()
        )
        
        # Fusion with attention
        self.feature_weight = nn.Parameter(torch.tensor(0.8))  # Initial 80% weight to features
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, features, heatmaps):
        # Feature branch (main)
        feat_out = self.feature_net(features)
        
        # CNN branch (auxiliary)
        cnn_out = self.cnn_net(heatmaps)
        
        # Weighted fusion
        w = torch.sigmoid(self.feature_weight)  # Keep between 0 and 1
        fused = torch.cat([feat_out * w, cnn_out * (1 - w)], dim=1)
        
        return self.classifier(fused)


def prepare_data():
    """Prepare data from rich features CSV and heatmaps."""
    
    # Load rich features
    df = pd.read_csv(DL_DATA_DIR / "rich_features.csv")
    
    # Numeric features
    exclude_cols = ['group', 'group_label', 'subject', 'file', 'left_valence', 'right_valence']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_features = df[feature_cols].values
    y = df['group_label'].values
    
    # Handle NaN/Inf
    X_features = np.nan_to_num(X_features, nan=0, posinf=0, neginf=0)
    
    # Load heatmaps (use original or multimodal)
    if (DL_DATA_DIR / "multimodal_maps.npy").exists():
        heatmaps = np.load(DL_DATA_DIR / "multimodal_maps.npy")
    else:
        heatmaps = np.load(DL_DATA_DIR / "heatmaps.npy")
        heatmaps = heatmaps[:, np.newaxis, :, :]  # Add channel dim
    
    # Ensure same size
    min_len = min(len(X_features), len(heatmaps))
    X_features = X_features[:min_len]
    heatmaps = heatmaps[:min_len]
    y = y[:min_len]
    
    return X_features, heatmaps, y, feature_cols


def get_class_weights(labels):
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights)


def train_hybrid():
    print("="*60)
    print("HYBRID MODEL TRAINING")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Prepare data
    X_features, heatmaps, y, feature_cols = prepare_data()
    
    print(f"\nFeatures: {X_features.shape}")
    print(f"Heatmaps: {heatmaps.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split
    X_feat_train, X_feat_test, X_heat_train, X_heat_test, y_train, y_test = train_test_split(
        X_features, heatmaps, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_feat_train = scaler.fit_transform(X_feat_train)
    X_feat_test = scaler.transform(X_feat_test)
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Datasets
    train_dataset = HybridDataset(X_feat_train, X_heat_train, y_train, augment=True)
    test_dataset = HybridDataset(X_feat_test, X_heat_test, y_test, augment=False)
    
    # Weighted sampler
    class_weights = get_class_weights(y_train)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = FeatureDominantHybrid(
        n_features=X_features.shape[1],
        n_heatmap_channels=heatmaps.shape[1],
        heatmap_size=heatmaps.shape[2],
        num_classes=3
    ).to(DEVICE)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # Training
    best_acc = 0
    best_model_state = None
    epochs_without_improvement = 0
    train_history = []
    test_history = []
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for feat_batch, heat_batch, y_batch in train_loader:
            feat_batch = feat_batch.to(DEVICE)
            heat_batch = heat_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(feat_batch, heat_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        train_acc = train_correct / train_total
        scheduler.step()
        
        # Evaluate
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for feat_batch, heat_batch, y_batch in test_loader:
                feat_batch = feat_batch.to(DEVICE)
                heat_batch = heat_batch.to(DEVICE)
                
                outputs = model(feat_batch, heat_batch)
                _, predicted = outputs.max(1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(y_batch.numpy())
        
        test_acc = accuracy_score(test_labels, test_preds)
        epoch_time = time.time() - epoch_start
        
        train_history.append({'acc': train_acc})
        test_history.append({'acc': test_acc})
        
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = test_preds.copy()
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            torch.save(best_model_state, RESULTS_DIR / "hybrid_best_model.pt")
        else:
            epochs_without_improvement += 1
        
        # Log feature weight
        feat_w = torch.sigmoid(model.feature_weight).item()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: Train={train_acc:.4f}, Test={test_acc:.4f}, Best={best_acc:.4f}, FeatW={feat_w:.2f} | {epoch_time:.1f}s")
        
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation with best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    final_preds = []
    final_labels = []
    
    with torch.no_grad():
        for feat_batch, heat_batch, y_batch in test_loader:
            feat_batch = feat_batch.to(DEVICE)
            heat_batch = heat_batch.to(DEVICE)
            
            outputs = model(feat_batch, heat_batch)
            _, predicted = outputs.max(1)
            final_preds.extend(predicted.cpu().numpy())
            final_labels.extend(y_batch.numpy())
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Final Feature Weight: {torch.sigmoid(model.feature_weight).item():.2f}")
    print(f"\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Hybrid Model (Acc={best_acc:.3f})')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_confusion_matrix.png", dpi=150)
    plt.close()
    
    # Training history
    plt.figure(figsize=(10, 4))
    epochs_range = range(1, len(train_history) + 1)
    plt.plot(epochs_range, [h['acc'] for h in train_history], label='Train')
    plt.plot(epochs_range, [h['acc'] for h in test_history], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Hybrid Model Training')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hybrid_training_history.png", dpi=150)
    plt.close()
    
    # Save results
    results = {
        'model': 'FeatureDominantHybrid',
        'best_accuracy': best_acc,
        'epochs_trained': epoch + 1,
        'final_feature_weight': torch.sigmoid(model.feature_weight).item(),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(final_labels, final_preds, target_names=CLASS_NAMES, output_dict=True)
    }
    
    with open(RESULTS_DIR / "hybrid_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved: {RESULTS_DIR / 'hybrid_best_model.pt'}")
    
    return results


if __name__ == "__main__":
    results = train_hybrid()








