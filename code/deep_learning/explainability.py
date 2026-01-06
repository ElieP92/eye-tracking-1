# explainability.py
"""
Explainability tools for deep learning models:
- Grad-CAM for CNN heatmaps
- Attention maps for Transformer/LSTM
- Feature importance comparison
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import models
from train_cnn import SimpleCNN, HeatmapDataset
from train_transformer import ScanpathTransformer, ScanpathDataset

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DL_DATA_DIR = PROJECT_ROOT / "results" / "deep_learning_data"
RESULTS_DIR = PROJECT_ROOT / "results" / "deep_learning_results"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Young', 'Middle-aged', 'Older']


class GradCAM:
    """Grad-CAM implementation for CNN visualization."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        """Generate Grad-CAM for input x."""
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(1)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def visualize_gradcam_samples(n_samples=9):
    """Generate Grad-CAM visualizations for sample heatmaps."""
    print("Generating Grad-CAM visualizations...")
    
    # Load data
    heatmaps = np.load(DL_DATA_DIR / "heatmaps.npy")
    labels = np.load(DL_DATA_DIR / "labels.npy")
    
    # Load model
    model = SimpleCNN(num_classes=3).to(DEVICE)
    
    # Train briefly on all data for visualization
    from torch.utils.data import DataLoader
    dataset = HeatmapDataset(heatmaps, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
    # Setup Grad-CAM
    target_layer = model.features[-4]  # Last conv layer before pooling
    gradcam = GradCAM(model, target_layer)
    
    # Visualize samples from each class
    fig, axes = plt.subplots(3, n_samples // 3, figsize=(15, 12))
    
    for class_idx in range(3):
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]
        
        for j in range(n_samples // 3):
            idx = class_indices[j * (len(class_indices) // (n_samples // 3))]
            
            x = torch.FloatTensor(heatmaps[idx:idx+1]).unsqueeze(1).to(DEVICE)
            cam = gradcam(x)
            
            ax = axes[class_idx, j]
            
            # Overlay
            ax.imshow(heatmaps[idx], cmap='hot')
            ax.imshow(cam, cmap='jet', alpha=0.4)
            ax.set_title(f'{CLASS_NAMES[class_idx]}')
            ax.axis('off')
    
    plt.suptitle('Grad-CAM: Important Regions for Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gradcam_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {RESULTS_DIR / 'gradcam_visualization.png'}")


def visualize_attention_patterns(n_samples=6):
    """Visualize transformer attention patterns for different classes."""
    print("Generating attention visualizations...")
    
    # Load data
    sequences = np.load(DL_DATA_DIR / "sequences.npy")
    labels = np.load(DL_DATA_DIR / "labels.npy")
    
    # Load model
    model = ScanpathTransformer(num_classes=3).to(DEVICE)
    
    # Train briefly for visualization
    from torch.utils.data import DataLoader
    dataset = ScanpathDataset(sequences, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(20):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    
    # Visualize
    fig, axes = plt.subplots(3, n_samples // 3, figsize=(15, 10))
    
    for class_idx in range(3):
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]
        
        for j in range(n_samples // 3):
            idx = class_indices[j * (len(class_indices) // (n_samples // 3))]
            
            x = torch.FloatTensor(sequences[idx:idx+1]).to(DEVICE)
            
            with torch.no_grad():
                output, attn = model(x)
            
            pred = output.argmax(1).item()
            attn_np = attn[0].cpu().numpy()
            
            ax = axes[class_idx, j]
            
            # Find valid (non-padded) length
            valid_len = (sequences[idx].sum(axis=1) != 0).sum()
            
            ax.bar(range(valid_len), attn_np[:valid_len], color=plt.cm.Set1(class_idx))
            ax.set_title(f'True: {CLASS_NAMES[class_idx]}, Pred: {CLASS_NAMES[pred]}')
            ax.set_xlabel('Fixation')
            ax.set_ylabel('Attention')
    
    plt.suptitle('Transformer Attention: Temporal Focus by Age Group', fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "attention_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {RESULTS_DIR / 'attention_patterns.png'}")


def analyze_class_differences():
    """Analyze spatial and temporal differences between age groups."""
    print("Analyzing class differences...")
    
    heatmaps = np.load(DL_DATA_DIR / "heatmaps.npy")
    sequences = np.load(DL_DATA_DIR / "sequences.npy")
    labels = np.load(DL_DATA_DIR / "labels.npy")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Average heatmaps per class
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = labels == i
        avg_heatmap = heatmaps[class_mask].mean(axis=0)
        
        ax = axes[0, i]
        im = ax.imshow(avg_heatmap, cmap='hot')
        ax.set_title(f'Average Heatmap: {class_name}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Scanpath statistics per class
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = labels == i
        class_seqs = sequences[class_mask]
        
        # Average fixation position over time
        valid_counts = (class_seqs.sum(axis=2) != 0).sum(axis=0)
        avg_x = []
        avg_y = []
        avg_dur = []
        
        for t in range(class_seqs.shape[1]):
            valid_mask = class_seqs[:, t, :].sum(axis=1) != 0
            if valid_mask.sum() > 0:
                avg_x.append(class_seqs[valid_mask, t, 0].mean())
                avg_y.append(class_seqs[valid_mask, t, 1].mean())
                avg_dur.append(class_seqs[valid_mask, t, 2].mean())
            else:
                break
        
        ax = axes[1, i]
        ax.plot(avg_x, label='X position', color='blue')
        ax.plot(avg_y, label='Y position', color='red')
        ax.plot(np.array(avg_dur) * 2, label='Duration (scaled)', color='green', linestyle='--')
        ax.set_title(f'Scanpath Pattern: {class_name}')
        ax.set_xlabel('Fixation Index')
        ax.set_ylabel('Normalized Value')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 30)
    
    plt.suptitle('Age Group Differences in Eye Movement Patterns', fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_differences.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {RESULTS_DIR / 'class_differences.png'}")
    
    # Statistics summary
    summary = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = labels == i
        class_seqs = sequences[class_mask]
        
        # Number of fixations per trial
        n_fix = (class_seqs.sum(axis=2) != 0).sum(axis=1)
        
        # Average duration
        durations = class_seqs[:, :, 2]
        valid_dur = durations[durations > 0]
        
        # Spatial spread
        x_coords = class_seqs[:, :, 0]
        y_coords = class_seqs[:, :, 1]
        valid_x = x_coords[x_coords > 0]
        valid_y = y_coords[y_coords > 0]
        
        summary[class_name] = {
            'n_samples': int(class_mask.sum()),
            'avg_n_fixations': float(n_fix.mean()),
            'std_n_fixations': float(n_fix.std()),
            'avg_duration': float(valid_dur.mean()) if len(valid_dur) > 0 else 0,
            'x_spread': float(valid_x.std()) if len(valid_x) > 0 else 0,
            'y_spread': float(valid_y.std()) if len(valid_y) > 0 else 0,
        }
        
        print(f"\n{class_name}:")
        print(f"  Samples: {summary[class_name]['n_samples']}")
        print(f"  Avg fixations: {summary[class_name]['avg_n_fixations']:.1f} +/- {summary[class_name]['std_n_fixations']:.1f}")
        print(f"  Avg duration: {summary[class_name]['avg_duration']:.3f}")
        print(f"  Spatial spread: X={summary[class_name]['x_spread']:.3f}, Y={summary[class_name]['y_spread']:.3f}")
    
    with open(RESULTS_DIR / "class_analysis.json", 'w') as f:
        json.dump(summary, f, indent=2)


def generate_report():
    """Generate comprehensive explainability report."""
    print("\n" + "="*60)
    print("GENERATING EXPLAINABILITY REPORT")
    print("="*60)
    
    # Run all visualizations
    try:
        visualize_gradcam_samples()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
    
    try:
        visualize_attention_patterns()
    except Exception as e:
        print(f"Attention error: {e}")
    
    analyze_class_differences()
    
    # Create markdown report
    report = """# Deep Learning Explainability Report

## 1. Grad-CAM Visualizations (CNN)

Grad-CAM highlights the regions of the heatmap that most influence the CNN's classification decision.

![Grad-CAM](gradcam_visualization.png)

**Interpretation**: Brighter regions in the overlay indicate areas the model focuses on for age group classification.

## 2. Attention Patterns (Transformer)

The attention weights show which fixations the Transformer model considers most important.

![Attention Patterns](attention_patterns.png)

**Interpretation**: Higher attention weights indicate temporally important fixations for classification.

## 3. Class Differences Analysis

Average eye movement patterns across age groups:

![Class Differences](class_differences.png)

### Key Observations:

- **Spatial Distribution**: Average heatmaps reveal different exploration patterns
- **Temporal Dynamics**: Scanpath trajectories show age-related differences in:
  - Initial fixation patterns (early vs late attention)
  - Fixation duration profiles
  - Spatial exploration extent

## 4. Feature Summary

See `class_analysis.json` for detailed statistics per age group.

---
*Generated automatically by explainability.py*
"""
    
    with open(RESULTS_DIR / "explainability_report.md", 'w') as f:
        f.write(report)
    
    print(f"\nReport saved: {RESULTS_DIR / 'explainability_report.md'}")


if __name__ == "__main__":
    generate_report()

