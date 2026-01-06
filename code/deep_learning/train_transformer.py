# train_transformer.py
"""
Transformer/LSTM classifier for scanpath sequence classification.
Captures temporal dynamics of eye movements for age group classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DL_DATA_DIR = PROJECT_ROOT / "results" / "deep_learning_data"
RESULTS_DIR = PROJECT_ROOT / "results" / "deep_learning_results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Young', 'Middle-aged', 'Older']

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
N_FOLDS = 5
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.3
PATIENCE = 30  # Early stopping patience


class ScanpathDataset(Dataset):
    """Dataset for scanpath sequences."""
    
    def __init__(self, sequences, labels, augment=False):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.sequences[idx].clone()
        y = self.labels[idx]
        
        if self.augment:
            # Small spatial noise
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(x[:, :2]) * 0.02
                x[:, :2] = torch.clamp(x[:, :2] + noise, 0, 1)
            # Duration jitter
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(x[:, 2:3]) * 0.1
                x[:, 2:3] = torch.clamp(x[:, 2:3] + noise, 0, 1)
        
        return x, y


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScanpathTransformer(nn.Module):
    """Transformer encoder for scanpath classification."""
    
    def __init__(self, input_dim=4, d_model=D_MODEL, n_heads=N_HEADS, 
                 n_layers=N_LAYERS, num_classes=3, dropout=DROPOUT):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Attention pooling
        self.attention_weight = nn.Linear(d_model, 1)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, 4)
        
        # Create padding mask (where all features are 0)
        if mask is None:
            mask = (x.sum(dim=-1) == 0)  # (batch, seq_len)
        
        # Project input
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Attention pooling
        attn_scores = self.attention_weight(x).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        x = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch, d_model)
        
        # Classify
        return self.classifier(x), attn_weights


class ScanpathLSTM(nn.Module):
    """BiLSTM with attention for scanpath classification."""
    
    def __init__(self, input_dim=4, hidden_dim=64, num_classes=3, dropout=DROPOUT):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, 4)
        
        if mask is None:
            mask = (x.sum(dim=-1) == 0)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        return self.classifier(context), attn_weights


def get_class_weights(labels):
    """Calculate class weights for imbalanced data."""
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs, _ = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_attentions = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs, attn = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_attentions.append(attn.cpu().numpy())
    
    return total_loss / len(loader), np.array(all_preds), np.array(all_labels), np.concatenate(all_attentions)


def train_with_cv(model_type='transformer'):
    """Train sequence model with Leave-Subject-Out cross-validation."""
    print("="*60)
    print(f"{model_type.upper()} TRAINING - SCANPATH CLASSIFICATION")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Load data
    sequences = np.load(DL_DATA_DIR / "sequences.npy")
    labels = np.load(DL_DATA_DIR / "labels.npy")
    subject_ids = np.load(DL_DATA_DIR / "subject_ids.npy")
    
    print(f"\nData shape: {sequences.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Group K-Fold by subject
    unique_subjects = np.unique(subject_ids)
    subject_to_group = {s: i for i, s in enumerate(unique_subjects)}
    groups = np.array([subject_to_group[s] for s in subject_ids])
    
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(unique_subjects)))
    
    all_preds = np.zeros(len(labels))
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(sequences, labels, groups)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        
        # Split data
        X_train, X_test = sequences[train_idx], sequences[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Datasets
        train_dataset = ScanpathDataset(X_train, y_train, augment=True)
        test_dataset = ScanpathDataset(X_test, y_test, augment=False)
        
        # Weighted sampler
        class_weights = get_class_weights(y_train)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model
        if model_type == 'transformer':
            model = ScanpathTransformer(num_classes=3).to(DEVICE)
        else:
            model = ScanpathLSTM(num_classes=3).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        # Training with early stopping and best model saving
        best_acc = 0
        best_model_state = None
        epochs_without_improvement = 0
        epoch_times = []
        
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            test_loss, preds, true, attn = evaluate(model, test_loader, criterion)
            test_acc = accuracy_score(true, preds)
            scheduler.step()
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_preds = preds
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            avg_epoch_time = np.mean(epoch_times)
            remaining = (EPOCHS - epoch - 1) * avg_epoch_time
            remaining_folds = (N_FOLDS - fold - 1) * EPOCHS * avg_epoch_time
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}: Loss={train_loss:.4f}, Acc={test_acc:.4f}, Best={best_acc:.4f} | {epoch_time:.1f}s/ep, ETA: {remaining/60:.1f}min")
            
            # Early stopping
            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Store predictions
        all_preds[test_idx] = best_preds
        fold_results.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': best_acc,
            'epochs_trained': epoch + 1
        })
        
        print(f"  Best Accuracy: {best_acc:.4f} (trained {epoch+1} epochs)")
    
    # Overall results
    overall_acc = accuracy_score(labels, all_preds)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels, all_preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(labels, all_preds)
    
    # Save results
    results = {
        'model': model_type,
        'overall_accuracy': overall_acc,
        'fold_results': fold_results,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    }
    
    with open(RESULTS_DIR / f"{model_type}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens' if model_type == 'transformer' else 'Oranges',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_type.capitalize()} Confusion Matrix (Acc={overall_acc:.3f})')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_type}_confusion_matrix.png", dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return results


def train_simple(model_type='transformer', test_size=0.2):
    """Train sequence model with simple train/test split (no cross-validation)."""
    print("="*60)
    print(f"{model_type.upper()} TRAINING - SIMPLE SPLIT (NO CV)")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Load data
    sequences = np.load(DL_DATA_DIR / "sequences.npy")
    labels = np.load(DL_DATA_DIR / "labels.npy")
    
    print(f"\nData shape: {sequences.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Datasets
    train_dataset = ScanpathDataset(X_train, y_train, augment=True)
    test_dataset = ScanpathDataset(X_test, y_test, augment=False)
    
    # Weighted sampler
    class_weights = get_class_weights(y_train)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    if model_type == 'transformer':
        model = ScanpathTransformer(num_classes=3).to(DEVICE)
    else:
        model = ScanpathLSTM(num_classes=3).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # Training with early stopping
    best_acc = 0
    best_model_state = None
    epochs_without_improvement = 0
    train_history = []
    test_history = []
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, preds, true, attn = evaluate(model, test_loader, criterion)
        test_acc = accuracy_score(true, preds)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        train_history.append({'loss': train_loss, 'acc': train_acc})
        test_history.append({'loss': test_loss, 'acc': test_acc})
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = preds
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            torch.save(best_model_state, RESULTS_DIR / f"{model_type}_best_model.pt")
        else:
            epochs_without_improvement += 1
        
        remaining = (EPOCHS - epoch - 1) * epoch_time
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}: Train={train_acc:.4f}, Test={test_acc:.4f}, Best={best_acc:.4f} | {epoch_time:.1f}s/ep, ETA: {remaining/60:.1f}min")
        
        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    _, final_preds, final_true, _ = evaluate(model, test_loader, criterion)
    final_acc = accuracy_score(final_true, final_preds)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Epochs trained: {epoch+1}")
    print(f"\nClassification Report:")
    print(classification_report(final_true, final_preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(final_true, final_preds)
    
    # Save results
    results = {
        'model': model_type,
        'mode': 'simple_split',
        'test_size': test_size,
        'best_accuracy': best_acc,
        'epochs_trained': epoch + 1,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(final_true, final_preds, target_names=CLASS_NAMES, output_dict=True)
    }
    
    with open(RESULTS_DIR / f"{model_type}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens' if model_type == 'transformer' else 'Oranges',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_type.capitalize()} Confusion Matrix (Acc={best_acc:.3f})')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_type}_confusion_matrix.png", dpi=150)
    plt.close()
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(train_history) + 1)
    
    axes[0].plot(epochs_range, [h['loss'] for h in train_history], label='Train')
    axes[0].plot(epochs_range, [h['loss'] for h in test_history], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_type.capitalize()} Training Loss')
    axes[0].legend()
    
    axes[1].plot(epochs_range, [h['acc'] for h in train_history], label='Train')
    axes[1].plot(epochs_range, [h['acc'] for h in test_history], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_type.capitalize()} Training Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_type}_training_history.png", dpi=150)
    plt.close()
    
    print(f"\nModel saved: {RESULTS_DIR / f'{model_type}_best_model.pt'}")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return results


def visualize_attention(model, sequences, labels, n_samples=5):
    """Visualize attention weights for sample predictions."""
    model.eval()
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    
    for i in range(n_samples):
        idx = np.random.randint(len(sequences))
        x = torch.FloatTensor(sequences[idx:idx+1]).to(DEVICE)
        
        with torch.no_grad():
            output, attn = model(x)
        
        pred = output.argmax(1).item()
        true = labels[idx]
        
        ax = axes[i] if n_samples > 1 else axes
        ax.bar(range(len(attn[0])), attn[0].cpu().numpy())
        ax.set_title(f'Sample {idx}: True={CLASS_NAMES[true]}, Pred={CLASS_NAMES[pred]}')
        ax.set_xlabel('Fixation Index')
        ax.set_ylabel('Attention Weight')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "attention_visualization.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # Train both models
    transformer_results = train_with_cv('transformer')
    lstm_results = train_with_cv('lstm')
    
    # Compare
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"Transformer Accuracy: {transformer_results['overall_accuracy']:.4f}")
    print(f"LSTM Accuracy: {lstm_results['overall_accuracy']:.4f}")

