# train_multimodal.py
"""
Multimodal Deep Learning for age classification:
- Channel 1: Spatial heatmap (fixation locations)
- Channel 2: Pupil heatmap (pupil size by location)
- Channel 3: Velocity heatmap (saccade velocity by location)
- Channel 4: Duration heatmap (fixation duration by location)

Plus rich feature vector concatenated with CNN output.
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
from dataclasses import dataclass
from typing import List, Dict
import re

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DATA_ROOT = PROJECT_ROOT.parent
DL_DATA_DIR = PROJECT_ROOT / "results" / "deep_learning_data"
RESULTS_DIR = PROJECT_ROOT / "results" / "deep_learning_results"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Young', 'Middle-aged', 'Older']

# Screen and heatmap config
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
HEATMAP_SIZE = 112  # Smaller for multi-channel

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 20

GROUP_MAPPING = {
    "agé": "age", "age": "age", "âgé": "age",
    "moyen": "moyen", "jeunes": "jeunes", "jeune": "jeunes"
}


@dataclass
class FixationEvent:
    start_time: float
    end_time: float
    duration: float
    x: float
    y: float
    pupil: float


@dataclass
class SaccadeEvent:
    start_time: float
    end_time: float
    duration: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    amplitude: float
    velocity: float


def parse_asc_multimodal(asc_file: Path) -> List[Dict]:
    """Parse ASC file for multimodal features."""
    trials = []
    current_trial_start = None
    current_trial_end = None
    fixations = []
    saccades = []
    
    with open(asc_file, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            
            if 'MSG' in line and 'Image_Start' in line:
                parts = line.split()
                if len(parts) >= 3:
                    if current_trial_start is not None and fixations:
                        trials.append({
                            'start': current_trial_start,
                            'end': current_trial_end or current_trial_start + 5000,
                            'fixations': fixations.copy(),
                            'saccades': saccades.copy()
                        })
                    current_trial_start = float(parts[1])
                    fixations = []
                    saccades = []
            
            if 'MSG' in line and 'pictureTrial_Offset' in line:
                parts = line.split()
                if len(parts) >= 3:
                    current_trial_end = float(parts[1])
                    if current_trial_start is not None and fixations:
                        trials.append({
                            'start': current_trial_start,
                            'end': current_trial_end,
                            'fixations': fixations.copy(),
                            'saccades': saccades.copy()
                        })
                    current_trial_start = None
                    fixations = []
                    saccades = []
            
            if line.startswith('EFIX') and current_trial_start:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        fix = FixationEvent(
                            start_time=float(parts[2]),
                            end_time=float(parts[3]),
                            duration=float(parts[4]),
                            x=float(parts[5]),
                            y=float(parts[6]),
                            pupil=float(parts[7]) if len(parts) > 7 else 0
                        )
                        if fix.start_time >= current_trial_start:
                            fixations.append(fix)
                    except (ValueError, IndexError):
                        continue
            
            if line.startswith('ESACC') and current_trial_start:
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        sac = SaccadeEvent(
                            start_time=float(parts[2]),
                            end_time=float(parts[3]),
                            duration=float(parts[4]),
                            start_x=float(parts[5]) if parts[5] != '.' else 0,
                            start_y=float(parts[6]) if parts[6] != '.' else 0,
                            end_x=float(parts[7]) if parts[7] != '.' else 0,
                            end_y=float(parts[8]) if parts[8] != '.' else 0,
                            amplitude=float(parts[9]) if parts[9] != '.' else 0,
                            velocity=float(parts[10]) if len(parts) > 10 and parts[10] != '.' else 0
                        )
                        if sac.start_time >= current_trial_start:
                            saccades.append(sac)
                    except (ValueError, IndexError):
                        continue
    
    return trials


def generate_multimodal_maps(fixations: List[FixationEvent], 
                             saccades: List[SaccadeEvent],
                             size: int = HEATMAP_SIZE) -> np.ndarray:
    """Generate 4-channel heatmap: spatial, pupil, velocity, duration."""
    
    # 4 channels
    maps = np.zeros((4, size, size), dtype=np.float32)
    
    if not fixations:
        return maps
    
    sigma = size // 25
    
    # Channel 0: Spatial (fixation count weighted by duration)
    # Channel 1: Pupil size
    # Channel 2: Velocity (from saccades ending at each location)
    # Channel 3: Duration
    
    for fix in fixations:
        x = int(fix.x / SCREEN_WIDTH * size)
        y = int(fix.y / SCREEN_HEIGHT * size)
        x = max(0, min(size - 1, x))
        y = max(0, min(size - 1, y))
        
        # Add gaussian kernel
        for dx in range(-sigma, sigma + 1):
            for dy in range(-sigma, sigma + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    dist_sq = dx * dx + dy * dy
                    gauss = np.exp(-dist_sq / (2 * sigma * sigma))
                    
                    # Channel 0: Spatial (duration-weighted)
                    maps[0, ny, nx] += gauss * (fix.duration / 100)
                    
                    # Channel 1: Pupil
                    if fix.pupil > 0:
                        maps[1, ny, nx] += gauss * (fix.pupil / 1000)
                    
                    # Channel 3: Duration
                    maps[3, ny, nx] += gauss * (fix.duration / 500)
    
    # Channel 2: Saccade velocity at landing positions
    for sac in saccades:
        if sac.velocity > 0 and sac.end_x > 0:
            x = int(sac.end_x / SCREEN_WIDTH * size)
            y = int(sac.end_y / SCREEN_HEIGHT * size)
            x = max(0, min(size - 1, x))
            y = max(0, min(size - 1, y))
            
            for dx in range(-sigma, sigma + 1):
                for dy in range(-sigma, sigma + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        dist_sq = dx * dx + dy * dy
                        gauss = np.exp(-dist_sq / (2 * sigma * sigma))
                        maps[2, ny, nx] += gauss * (sac.velocity / 500)
    
    # Normalize each channel
    for c in range(4):
        if maps[c].max() > 0:
            maps[c] = maps[c] / maps[c].max()
    
    return maps


def compute_trial_features(fixations: List[FixationEvent], 
                           saccades: List[SaccadeEvent]) -> np.ndarray:
    """Compute rich features for the trial."""
    features = []
    
    n_fix = len(fixations)
    n_sac = len(saccades)
    
    # Basic counts
    features.append(n_fix)
    features.append(n_sac)
    
    if n_fix > 0:
        # Pupil features
        pupil_sizes = [f.pupil for f in fixations if f.pupil > 0]
        features.append(np.mean(pupil_sizes) if pupil_sizes else 0)
        features.append(np.std(pupil_sizes) if len(pupil_sizes) > 1 else 0)
        features.append(np.max(pupil_sizes) - np.min(pupil_sizes) if pupil_sizes else 0)
        
        # Fixation features
        fix_dur = [f.duration for f in fixations]
        features.append(np.mean(fix_dur))
        features.append(np.std(fix_dur))
        
        # Spatial features
        fix_x = [f.x for f in fixations]
        fix_y = [f.y for f in fixations]
        features.append(np.std(fix_x))
        features.append(np.std(fix_y))
        
        # AOI features (left vs right)
        n_left = sum(1 for f in fixations if f.x < SCREEN_WIDTH/2 - 50)
        n_right = sum(1 for f in fixations if f.x > SCREEN_WIDTH/2 + 50)
        features.append(n_left / (n_right + 1))
    else:
        features.extend([0] * 9)
    
    if n_sac > 0:
        velocities = [s.velocity for s in saccades if s.velocity > 0]
        features.append(np.mean(velocities) if velocities else 0)
        features.append(np.max(velocities) if velocities else 0)
        
        amplitudes = [s.amplitude for s in saccades if s.amplitude > 0]
        features.append(np.mean(amplitudes) if amplitudes else 0)
    else:
        features.extend([0] * 3)
    
    return np.array(features, dtype=np.float32)


def extract_multimodal_data():
    """Extract multimodal data from all subjects."""
    print("="*60)
    print("EXTRACTING MULTIMODAL DATA")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    all_maps = []
    all_features = []
    all_labels = []
    all_subjects = []
    
    for group_folder in ['jeunes', 'moyen', 'agé', 'age']:
        group_path = RAW_DATA_ROOT / group_folder
        if not group_path.exists():
            continue
        
        group_name = GROUP_MAPPING.get(group_folder, group_folder)
        group_label = {'jeunes': 0, 'moyen': 1, 'age': 2}.get(group_name, -1)
        
        if group_label == -1:
            continue
        
        print(f"\nProcessing {group_name}...")
        
        for subject_dir in sorted(group_path.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            subject_id = f"{group_name}_{subject_dir.name}"
            
            for asc_file in subject_dir.glob("*.asc"):
                print(f"  {asc_file.name}", end=" ")
                
                trials = parse_asc_multimodal(asc_file)
                valid_count = 0
                
                for trial in trials:
                    if len(trial['fixations']) < 3:
                        continue
                    
                    # Generate multimodal maps
                    maps = generate_multimodal_maps(
                        trial['fixations'], 
                        trial['saccades']
                    )
                    
                    # Compute features
                    features = compute_trial_features(
                        trial['fixations'],
                        trial['saccades']
                    )
                    
                    all_maps.append(maps)
                    all_features.append(features)
                    all_labels.append(group_label)
                    all_subjects.append(subject_id)
                    valid_count += 1
                
                print(f"-> {valid_count} trials")
    
    # Convert to arrays
    maps = np.array(all_maps, dtype=np.float32)
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    subjects = np.array(all_subjects)
    
    print(f"\n{'='*60}")
    print(f"Maps shape: {maps.shape}")  # (N, 4, H, W)
    print(f"Features shape: {features.shape}")  # (N, F)
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Save
    np.save(DL_DATA_DIR / "multimodal_maps.npy", maps)
    np.save(DL_DATA_DIR / "multimodal_features.npy", features)
    np.save(DL_DATA_DIR / "multimodal_labels.npy", labels)
    np.save(DL_DATA_DIR / "multimodal_subjects.npy", subjects)
    
    return maps, features, labels, subjects


class MultimodalDataset(Dataset):
    """Dataset for multimodal data."""
    
    def __init__(self, maps, features, labels, augment=False):
        self.maps = torch.FloatTensor(maps)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        m = self.maps[idx]
        f = self.features[idx]
        y = self.labels[idx]
        
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                m = torch.flip(m, dims=[2])
            # Small noise
            m = m + torch.randn_like(m) * 0.02
        
        return m, f, y


class MultimodalCNN(nn.Module):
    """Multimodal CNN with feature fusion."""
    
    def __init__(self, n_channels=4, n_features=14, num_classes=3):
        super().__init__()
        
        # CNN branch for maps
        self.conv_branch = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
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
            
            nn.Flatten()
        )
        
        # Feature branch
        self.feature_branch = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Fusion and classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, maps, features):
        # CNN branch
        cnn_out = self.conv_branch(maps)
        
        # Feature branch
        feat_out = self.feature_branch(features)
        
        # Fusion
        fused = torch.cat([cnn_out, feat_out], dim=1)
        
        return self.classifier(fused)


def get_class_weights(labels):
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights)


def train_multimodal():
    """Train multimodal model."""
    print("\n" + "="*60)
    print("MULTIMODAL CNN TRAINING")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Load data
    maps = np.load(DL_DATA_DIR / "multimodal_maps.npy")
    features = np.load(DL_DATA_DIR / "multimodal_features.npy")
    labels = np.load(DL_DATA_DIR / "multimodal_labels.npy")
    
    print(f"\nMaps: {maps.shape}, Features: {features.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Handle NaN/Inf in features
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    
    # Split
    X_maps_train, X_maps_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        maps, features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_feat_train = scaler.fit_transform(X_feat_train)
    X_feat_test = scaler.transform(X_feat_test)
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Datasets
    train_dataset = MultimodalDataset(X_maps_train, X_feat_train, y_train, augment=True)
    test_dataset = MultimodalDataset(X_maps_test, X_feat_test, y_test, augment=False)
    
    # Weighted sampler
    class_weights = get_class_weights(y_train)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = MultimodalCNN(
        n_channels=maps.shape[1],
        n_features=features.shape[1],
        num_classes=3
    ).to(DEVICE)
    
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
        
        for maps_batch, feat_batch, y_batch in train_loader:
            maps_batch = maps_batch.to(DEVICE)
            feat_batch = feat_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(maps_batch, feat_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        train_acc = train_correct / train_total
        scheduler.step()
        
        # Evaluate
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for maps_batch, feat_batch, y_batch in test_loader:
                maps_batch = maps_batch.to(DEVICE)
                feat_batch = feat_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                outputs = model(maps_batch, feat_batch)
                loss = criterion(outputs, y_batch)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        test_acc = accuracy_score(all_labels, all_preds)
        epoch_time = time.time() - epoch_start
        
        train_history.append({'loss': train_loss / len(train_loader), 'acc': train_acc})
        test_history.append({'loss': test_loss / len(test_loader), 'acc': test_acc})
        
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, RESULTS_DIR / "multimodal_best_model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: Train={train_acc:.4f}, Test={test_acc:.4f}, Best={best_acc:.4f} | {epoch_time:.1f}s")
        
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for maps_batch, feat_batch, y_batch in test_loader:
            maps_batch = maps_batch.to(DEVICE)
            feat_batch = feat_batch.to(DEVICE)
            
            outputs = model(maps_batch, feat_batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    final_acc = accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Multimodal CNN (Acc={best_acc:.3f})')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "multimodal_confusion_matrix.png", dpi=150)
    plt.close()
    
    # Training history
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
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "multimodal_training_history.png", dpi=150)
    plt.close()
    
    # Save results
    results = {
        'model': 'MultimodalCNN',
        'best_accuracy': best_acc,
        'epochs_trained': epoch + 1,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    }
    
    with open(RESULTS_DIR / "multimodal_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved: {RESULTS_DIR / 'multimodal_best_model.pt'}")
    
    return results


if __name__ == "__main__":
    # Extract data if not exists
    if not (DL_DATA_DIR / "multimodal_maps.npy").exists():
        extract_multimodal_data()
    
    # Train
    results = train_multimodal()








