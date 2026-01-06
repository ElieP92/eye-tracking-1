# extract_fixations.py
"""
Extract all fixations with coordinates from .asc files for deep learning.
Generates heatmaps and scanpath sequences per trial.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DATA_ROOT = PROJECT_ROOT.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DL_DATA_DIR = RESULTS_DIR / "deep_learning_data"
DL_DATA_DIR.mkdir(exist_ok=True)

# Screen dimensions
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768

# Heatmap size
HEATMAP_SIZE = 224

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


def parse_asc_fixations(asc_file: Path) -> Tuple[List[FixationEvent], List[Dict]]:
    """Parse all fixations and trial markers from ASC file."""
    fixations = []
    trials = []
    current_trial_start = None
    
    with open(asc_file, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            
            # Trial markers
            if 'MSG' in line and 'Image_Start' in line:
                parts = line.split()
                if len(parts) >= 3:
                    current_trial_start = float(parts[1])
                    trials.append({
                        'start_time': current_trial_start,
                        'fixations': []
                    })
            
            if 'MSG' in line and 'pictureTrial_Offset' in line:
                parts = line.split()
                if len(parts) >= 3 and trials:
                    trials[-1]['end_time'] = float(parts[1])
            
            # Fixation events: EFIX R/L start end dur x y pupil
            if line.startswith('EFIX'):
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
                        fixations.append(fix)
                        
                        # Add to current trial if within bounds
                        if trials and current_trial_start:
                            if fix.start_time >= current_trial_start:
                                if 'end_time' not in trials[-1] or fix.start_time < trials[-1]['end_time']:
                                    trials[-1]['fixations'].append(fix)
                    except (ValueError, IndexError):
                        continue
    
    return fixations, trials


def generate_heatmap(fixations: List[FixationEvent], size: int = HEATMAP_SIZE) -> np.ndarray:
    """Generate gaussian heatmap from fixations."""
    heatmap = np.zeros((size, size), dtype=np.float32)
    
    if not fixations:
        return heatmap
    
    # Gaussian kernel size
    sigma = size // 20
    
    for fix in fixations:
        # Normalize coordinates to heatmap size
        x = int(fix.x / SCREEN_WIDTH * size)
        y = int(fix.y / SCREEN_HEIGHT * size)
        
        # Clip to bounds
        x = max(0, min(size - 1, x))
        y = max(0, min(size - 1, y))
        
        # Add gaussian with amplitude proportional to duration
        amplitude = fix.duration / 100  # Scale duration
        
        # Create gaussian kernel
        for dx in range(-sigma * 2, sigma * 2 + 1):
            for dy in range(-sigma * 2, sigma * 2 + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    dist_sq = dx * dx + dy * dy
                    gauss = amplitude * np.exp(-dist_sq / (2 * sigma * sigma))
                    heatmap[ny, nx] += gauss
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


def create_scanpath_sequence(fixations: List[FixationEvent], max_length: int = 50) -> np.ndarray:
    """Create scanpath sequence [x_norm, y_norm, duration_norm, is_saccade]."""
    sequence = np.zeros((max_length, 4), dtype=np.float32)
    
    for i, fix in enumerate(fixations[:max_length]):
        # Normalize coordinates
        x_norm = fix.x / SCREEN_WIDTH
        y_norm = fix.y / SCREEN_HEIGHT
        dur_norm = min(fix.duration / 500, 1.0)  # Cap at 500ms
        is_saccade = 0  # This is a fixation
        
        sequence[i] = [x_norm, y_norm, dur_norm, is_saccade]
    
    return sequence


def process_all_data():
    """Process all participants and generate DL-ready data."""
    print("="*60)
    print("EXTRACTING FIXATIONS FOR DEEP LEARNING")
    print("="*60)
    
    all_heatmaps = []
    all_sequences = []
    all_labels = []
    all_metadata = []
    
    # Process each group
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
            
            subject_id = subject_dir.name
            
            for asc_file in subject_dir.glob("*.asc"):
                print(f"  {asc_file.name}", end=" ")
                
                # Parse fixations
                all_fix, trials = parse_asc_fixations(asc_file)
                
                trial_count = 0
                for trial in trials:
                    if 'fixations' not in trial or len(trial['fixations']) < 3:
                        continue
                    
                    # Generate heatmap
                    heatmap = generate_heatmap(trial['fixations'])
                    
                    # Generate sequence
                    sequence = create_scanpath_sequence(trial['fixations'])
                    
                    all_heatmaps.append(heatmap)
                    all_sequences.append(sequence)
                    all_labels.append(group_label)
                    all_metadata.append({
                        'group': group_name,
                        'subject': subject_id,
                        'file': asc_file.name,
                        'n_fixations': len(trial['fixations'])
                    })
                    
                    trial_count += 1
                
                print(f"-> {trial_count} trials")
    
    # Convert to numpy arrays
    heatmaps = np.array(all_heatmaps, dtype=np.float32)
    sequences = np.array(all_sequences, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    
    print(f"\n{'='*60}")
    print(f"Total samples: {len(labels)}")
    print(f"Heatmaps shape: {heatmaps.shape}")
    print(f"Sequences shape: {sequences.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Save data
    np.save(DL_DATA_DIR / "heatmaps.npy", heatmaps)
    np.save(DL_DATA_DIR / "sequences.npy", sequences)
    np.save(DL_DATA_DIR / "labels.npy", labels)
    
    with open(DL_DATA_DIR / "metadata.json", 'w') as f:
        json.dump(all_metadata, f)
    
    # Save subject IDs for GroupKFold
    subjects = [m['subject'] for m in all_metadata]
    groups = [m['group'] for m in all_metadata]
    subject_ids = [f"{g}_{s}" for g, s in zip(groups, subjects)]
    np.save(DL_DATA_DIR / "subject_ids.npy", np.array(subject_ids))
    
    print(f"\nData saved to: {DL_DATA_DIR}")
    print("="*60)
    
    return heatmaps, sequences, labels


if __name__ == "__main__":
    process_all_data()

