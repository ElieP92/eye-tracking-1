# train_cnn.py
"""
CNN classifier for heatmap-based age group classification.
Uses ResNet18 adapted for single-channel input with Leave-Subject-Out CV.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
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
LEARNING_RATE = 1e-4
N_FOLDS = 5
PATIENCE = 30  # Early stopping patience


class HeatmapDataset(Dataset):
    """Dataset for heatmap images."""
    
    def __init__(self, heatmaps, labels, augment=False):
        self.heatmaps = torch.FloatTensor(heatmaps).unsqueeze(1)  # Add channel dim
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.heatmaps[idx]
        y = self.labels[idx]
        
        if self.augment:
            # Random spatial jitter
            if torch.rand(1) > 0.5:
                shift = torch.randint(-5, 6, (2,))
                x = torch.roll(x, shifts=shift.tolist(), dims=(1, 2))
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])
        
        return x, y


class HeatmapCNN(nn.Module):
    """ResNet18 adapted for single-channel heatmaps."""
    
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Modify first conv for single channel
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class SimpleCNN(nn.Module):
    """Simple custom CNN for comparison."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


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
        outputs = model(x)
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
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)


def train_with_cv():
    """Train CNN with Leave-Subject-Out cross-validation."""
    print("="*60)
    print("CNN TRAINING - HEATMAP CLASSIFICATION")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Load data
    heatmaps = np.load(DL_DATA_DIR / "heatmaps.npy")
    labels = np.load(DL_DATA_DIR / "labels.npy")
    subject_ids = np.load(DL_DATA_DIR / "subject_ids.npy")
    
    print(f"\nData shape: {heatmaps.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Group K-Fold by subject
    unique_subjects = np.unique(subject_ids)
    subject_to_group = {s: i for i, s in enumerate(unique_subjects)}
    groups = np.array([subject_to_group[s] for s in subject_ids])
    
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(unique_subjects)))
    
    all_preds = np.zeros(len(labels))
    all_probs = np.zeros((len(labels), 3))
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(heatmaps, labels, groups)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        
        # Split data
        X_train, X_test = heatmaps[train_idx], heatmaps[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Datasets
        train_dataset = HeatmapDataset(X_train, y_train, augment=True)
        test_dataset = HeatmapDataset(X_test, y_test, augment=False)
        
        # Weighted sampler for class imbalance
        class_weights = get_class_weights(y_train)
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model
        model = SimpleCNN(num_classes=3).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training with early stopping and best model saving
        best_acc = 0
        best_model_state = None
        epochs_without_improvement = 0
        epoch_times = []
        
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            test_loss, preds, true = evaluate(model, test_loader, criterion)
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
        
        # Load best model for final predictions
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
        'model': 'SimpleCNN',
        'overall_accuracy': overall_acc,
        'fold_results': fold_results,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    }
    
    with open(RESULTS_DIR / "cnn_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'CNN Confusion Matrix (Acc={overall_acc:.3f})')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_confusion_matrix.png", dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return results


def train_simple(test_size=0.2):
    """Train CNN with simple train/test split (no cross-validation)."""
    print("="*60)
    print("CNN TRAINING - SIMPLE SPLIT (NO CV)")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Load data
    heatmaps = np.load(DL_DATA_DIR / "heatmaps.npy")
    labels = np.load(DL_DATA_DIR / "labels.npy")
    
    print(f"\nData shape: {heatmaps.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        heatmaps, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Datasets
    train_dataset = HeatmapDataset(X_train, y_train, augment=True)
    test_dataset = HeatmapDataset(X_test, y_test, augment=False)
    
    # Weighted sampler
    class_weights = get_class_weights(y_train)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = SimpleCNN(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training with early stopping
    best_acc = 0
    best_model_state = None
    epochs_without_improvement = 0
    train_history = []
    test_history = []
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, preds, true = evaluate(model, test_loader, criterion)
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
            # Save model checkpoint
            torch.save(best_model_state, RESULTS_DIR / "cnn_best_model.pt")
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
    _, final_preds, final_true = evaluate(model, test_loader, criterion)
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
        'model': 'SimpleCNN',
        'mode': 'simple_split',
        'test_size': test_size,
        'best_accuracy': best_acc,
        'epochs_trained': epoch + 1,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(final_true, final_preds, target_names=CLASS_NAMES, output_dict=True)
    }
    
    with open(RESULTS_DIR / "cnn_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'CNN Confusion Matrix (Acc={best_acc:.3f})')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_confusion_matrix.png", dpi=150)
    plt.close()
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(train_history) + 1)
    
    axes[0].plot(epochs_range, [h['loss'] for h in train_history], label='Train')
    axes[0].plot(epochs_range, [h['loss'] for h in test_history], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    
    axes[1].plot(epochs_range, [h['acc'] for h in train_history], label='Train')
    axes[1].plot(epochs_range, [h['acc'] for h in test_history], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].axhline(y=best_acc, color='r', linestyle='--', label=f'Best={best_acc:.3f}')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_training_history.png", dpi=150)
    plt.close()
    
    print(f"\nModel saved: {RESULTS_DIR / 'cnn_best_model.pt'}")
    print(f"Results saved to: {RESULTS_DIR}")
    
    return results


def permutation_test(n_permutations=100):
    """Run permutation test to assess significance."""
    print(f"\nRunning permutation test ({n_permutations} permutations)...")
    
    heatmaps = np.load(DL_DATA_DIR / "heatmaps.npy")
    labels = np.load(DL_DATA_DIR / "labels.npy")
    subject_ids = np.load(DL_DATA_DIR / "subject_ids.npy")
    
    # Get baseline accuracy (already computed)
    with open(RESULTS_DIR / "cnn_results.json") as f:
        baseline_acc = json.load(f)['overall_accuracy']
    
    # Permutation accuracies
    perm_accs = []
    
    for i in range(n_permutations):
        # Shuffle labels by subject
        unique_subjects = np.unique(subject_ids)
        shuffled_labels = labels.copy()
        
        for subj in unique_subjects:
            mask = subject_ids == subj
            # Assign random class to entire subject
            shuffled_labels[mask] = np.random.randint(0, 3)
        
        # Quick evaluation with simple model
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=i)
        
        # Flatten heatmaps
        X_flat = heatmaps.reshape(len(heatmaps), -1)
        
        # Simple train/test split
        split = int(0.8 * len(X_flat))
        clf.fit(X_flat[:split], shuffled_labels[:split])
        preds = clf.predict(X_flat[split:])
        acc = accuracy_score(shuffled_labels[split:], preds)
        perm_accs.append(acc)
        
        if (i + 1) % 20 == 0:
            print(f"  Permutation {i+1}/{n_permutations}")
    
    # Calculate p-value
    p_value = (np.sum(np.array(perm_accs) >= baseline_acc) + 1) / (n_permutations + 1)
    
    print(f"\nBaseline accuracy: {baseline_acc:.4f}")
    print(f"Permutation mean: {np.mean(perm_accs):.4f} (+/- {np.std(perm_accs):.4f})")
    print(f"P-value: {p_value:.4f}")
    
    return p_value


if __name__ == "__main__":
    results = train_with_cv()
    # permutation_test()  # Uncomment to run permutation test

