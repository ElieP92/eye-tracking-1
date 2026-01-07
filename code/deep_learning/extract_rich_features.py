# extract_rich_features.py
"""
Extract rich, discriminative features for age group classification.
Based on cognitive aging literature:
- Pupil dynamics (cognitive load, emotional reactivity)
- Saccade velocity (presbykinesis)
- Temporal dynamics (early vs late processing)
- Valence × AOI interaction (positivity bias in aging)
- Intra-individual variability
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from scipy import stats

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
SCREEN_CENTER_X = SCREEN_WIDTH / 2

# AOI definitions
AOI_LEFT = (0, SCREEN_CENTER_X - 50)  # Left image
AOI_RIGHT = (SCREEN_CENTER_X + 50, SCREEN_WIDTH)  # Right image

# Temporal windows
WINDOW_EARLY = (0, 2500)  # 0-2.5s
WINDOW_LATE = (2500, 5000)  # 2.5-5s

GROUP_MAPPING = {
    "agé": "age", "age": "age", "âgé": "age",
    "moyen": "moyen", "jeunes": "jeunes", "jeune": "jeunes"
}

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

@dataclass
class FixationEvent:
    start_time: float
    end_time: float
    duration: float
    x: float
    y: float
    pupil: float

@dataclass
class TrialData:
    trial_start: float
    trial_end: float
    fixations: List[FixationEvent]
    saccades: List[SaccadeEvent]
    left_image: str
    right_image: str
    

def parse_asc_detailed(asc_file: Path) -> List[TrialData]:
    """Parse ASC file with full detail extraction."""
    trials = []
    current_trial = None
    fixations = []
    saccades = []
    pupil_samples = []
    left_img = ""
    right_img = ""
    
    with open(asc_file, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            
            # Trial markers
            if 'MSG' in line and 'Image_Start' in line:
                parts = line.split()
                if len(parts) >= 3:
                    trial_start = float(parts[1])
                    if current_trial is not None:
                        # Save previous trial
                        trials.append(TrialData(
                            trial_start=current_trial,
                            trial_end=trial_start,
                            fixations=fixations.copy(),
                            saccades=saccades.copy(),
                            left_image=left_img,
                            right_image=right_img
                        ))
                    current_trial = trial_start
                    fixations = []
                    saccades = []
            
            if 'MSG' in line and 'pictureTrial_Offset' in line:
                parts = line.split()
                if len(parts) >= 3 and current_trial is not None:
                    trial_end = float(parts[1])
                    trials.append(TrialData(
                        trial_start=current_trial,
                        trial_end=trial_end,
                        fixations=fixations.copy(),
                        saccades=saccades.copy(),
                        left_image=left_img,
                        right_image=right_img
                    ))
                    current_trial = None
                    fixations = []
                    saccades = []
            
            # Image names
            if 'MSG' in line and 'StimulusGauche' in line:
                match = re.search(r"'StimulusGauche\s+(\S+)'", line)
                if match:
                    left_img = match.group(1)
            
            if 'MSG' in line and 'StimulusDroit' in line:
                match = re.search(r"'StimulusDroit\s+(\S+)'", line)
                if match:
                    right_img = match.group(1)
            
            # Fixation: EFIX L start end dur x y pupil
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
                        if current_trial is not None:
                            if fix.start_time >= current_trial:
                                fixations.append(fix)
                    except (ValueError, IndexError):
                        continue
            
            # Saccade: ESACC L start end dur start_x start_y end_x end_y amp vel
            if line.startswith('ESACC'):
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        sac = SaccadeEvent(
                            start_time=float(parts[2]),
                            end_time=float(parts[3]),
                            duration=float(parts[4]),
                            start_x=float(parts[5]),
                            start_y=float(parts[6]),
                            end_x=float(parts[7]),
                            end_y=float(parts[8]),
                            amplitude=float(parts[9]) if parts[9] != '.' else 0,
                            velocity=float(parts[10]) if len(parts) > 10 and parts[10] != '.' else 0
                        )
                        if current_trial is not None:
                            if sac.start_time >= current_trial:
                                saccades.append(sac)
                    except (ValueError, IndexError):
                        continue
    
    return trials


def get_valence_from_filename(filename: str) -> str:
    """Extract valence from image filename."""
    filename = filename.lower()
    if 'neg' in filename:
        return 'negative'
    elif 'pos' in filename:
        return 'positive'
    elif 'neu' in filename:
        return 'neutral'
    return 'unknown'


def compute_trial_features(trial: TrialData) -> Dict:
    """Compute rich features for a single trial."""
    features = {}
    
    # Basic counts
    n_fix = len(trial.fixations)
    n_sac = len(trial.saccades)
    features['n_fixations'] = n_fix
    features['n_saccades'] = n_sac
    
    if n_fix == 0:
        return None  # Invalid trial
    
    # === PUPIL DYNAMICS ===
    pupil_sizes = [f.pupil for f in trial.fixations if f.pupil > 0]
    if pupil_sizes:
        features['pupil_mean'] = np.mean(pupil_sizes)
        features['pupil_std'] = np.std(pupil_sizes)
        features['pupil_max'] = np.max(pupil_sizes)
        features['pupil_min'] = np.min(pupil_sizes)
        features['pupil_range'] = features['pupil_max'] - features['pupil_min']
        # Pupil change over time (reactivity)
        if len(pupil_sizes) > 1:
            features['pupil_slope'] = np.polyfit(range(len(pupil_sizes)), pupil_sizes, 1)[0]
        else:
            features['pupil_slope'] = 0
    else:
        features['pupil_mean'] = 0
        features['pupil_std'] = 0
        features['pupil_max'] = 0
        features['pupil_min'] = 0
        features['pupil_range'] = 0
        features['pupil_slope'] = 0
    
    # === SACCADE DYNAMICS ===
    if n_sac > 0:
        velocities = [s.velocity for s in trial.saccades if s.velocity > 0]
        amplitudes = [s.amplitude for s in trial.saccades if s.amplitude > 0]
        sac_durations = [s.duration for s in trial.saccades]
        
        features['saccade_velocity_mean'] = np.mean(velocities) if velocities else 0
        features['saccade_velocity_max'] = np.max(velocities) if velocities else 0
        features['saccade_velocity_std'] = np.std(velocities) if velocities else 0
        features['saccade_amplitude_mean'] = np.mean(amplitudes) if amplitudes else 0
        features['saccade_duration_mean'] = np.mean(sac_durations)
    else:
        features['saccade_velocity_mean'] = 0
        features['saccade_velocity_max'] = 0
        features['saccade_velocity_std'] = 0
        features['saccade_amplitude_mean'] = 0
        features['saccade_duration_mean'] = 0
    
    # === FIXATION DYNAMICS ===
    fix_durations = [f.duration for f in trial.fixations]
    features['fixation_duration_mean'] = np.mean(fix_durations)
    features['fixation_duration_std'] = np.std(fix_durations)
    features['fixation_duration_max'] = np.max(fix_durations)
    features['fixation_duration_min'] = np.min(fix_durations)
    
    # === SPATIAL DISTRIBUTION ===
    fix_x = [f.x for f in trial.fixations]
    fix_y = [f.y for f in trial.fixations]
    features['spatial_spread_x'] = np.std(fix_x)
    features['spatial_spread_y'] = np.std(fix_y)
    
    # === AOI ANALYSIS ===
    fix_left = [f for f in trial.fixations if AOI_LEFT[0] <= f.x <= AOI_LEFT[1]]
    fix_right = [f for f in trial.fixations if AOI_RIGHT[0] <= f.x <= AOI_RIGHT[1]]
    
    features['n_fix_left'] = len(fix_left)
    features['n_fix_right'] = len(fix_right)
    features['ratio_left_right'] = len(fix_left) / (len(fix_right) + 1)  # Avoid div by 0
    
    total_dur_left = sum(f.duration for f in fix_left)
    total_dur_right = sum(f.duration for f in fix_right)
    features['dwell_time_left'] = total_dur_left
    features['dwell_time_right'] = total_dur_right
    features['dwell_ratio_left_right'] = total_dur_left / (total_dur_right + 1)
    
    # === TEMPORAL DYNAMICS (Early vs Late) ===
    trial_duration = trial.trial_end - trial.trial_start
    mid_point = trial.trial_start + trial_duration / 2
    
    fix_early = [f for f in trial.fixations if f.start_time < mid_point]
    fix_late = [f for f in trial.fixations if f.start_time >= mid_point]
    
    features['n_fix_early'] = len(fix_early)
    features['n_fix_late'] = len(fix_late)
    features['ratio_early_late'] = len(fix_early) / (len(fix_late) + 1)
    
    # First fixation latency (time to first fixation on image)
    first_image_fix = None
    for f in trial.fixations:
        if f.x < AOI_LEFT[1] or f.x > AOI_RIGHT[0]:
            first_image_fix = f
            break
    if first_image_fix:
        features['first_fix_latency'] = first_image_fix.start_time - trial.trial_start
    else:
        features['first_fix_latency'] = 0
    
    # === VALENCE INTERACTION ===
    left_valence = get_valence_from_filename(trial.left_image)
    right_valence = get_valence_from_filename(trial.right_image)
    
    # Dwell time on emotional vs neutral
    if left_valence != 'neutral':
        features['dwell_emotional'] = total_dur_left
        features['dwell_neutral'] = total_dur_right
    elif right_valence != 'neutral':
        features['dwell_emotional'] = total_dur_right
        features['dwell_neutral'] = total_dur_left
    else:
        features['dwell_emotional'] = 0
        features['dwell_neutral'] = total_dur_left + total_dur_right
    
    features['emotional_bias'] = features['dwell_emotional'] - features['dwell_neutral']
    
    # === TRANSITION PATTERNS ===
    transitions_lr = 0  # Left to Right
    transitions_rl = 0  # Right to Left
    
    prev_aoi = None
    for f in trial.fixations:
        if f.x <= AOI_LEFT[1]:
            curr_aoi = 'left'
        elif f.x >= AOI_RIGHT[0]:
            curr_aoi = 'right'
        else:
            curr_aoi = 'center'
        
        if prev_aoi == 'left' and curr_aoi == 'right':
            transitions_lr += 1
        elif prev_aoi == 'right' and curr_aoi == 'left':
            transitions_rl += 1
        
        prev_aoi = curr_aoi
    
    features['transitions_lr'] = transitions_lr
    features['transitions_rl'] = transitions_rl
    features['total_transitions'] = transitions_lr + transitions_rl
    
    # === VARIABILITY INDICES ===
    features['cv_fixation_duration'] = features['fixation_duration_std'] / (features['fixation_duration_mean'] + 1)
    
    # Encode valence info
    features['left_valence'] = left_valence
    features['right_valence'] = right_valence
    
    return features


def process_all_subjects():
    """Process all subjects and extract rich features."""
    print("="*60)
    print("EXTRACTING RICH FEATURES FOR ML/DL")
    print("="*60)
    
    all_features = []
    
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
                
                trials = parse_asc_detailed(asc_file)
                valid_count = 0
                
                for trial in trials:
                    features = compute_trial_features(trial)
                    if features is None:
                        continue
                    
                    # Add metadata
                    features['group'] = group_name
                    features['group_label'] = group_label
                    features['subject'] = subject_id
                    features['file'] = asc_file.name
                    
                    all_features.append(features)
                    valid_count += 1
                
                print(f"-> {valid_count} trials")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    print(f"\n{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"\nClass distribution:")
    print(df['group'].value_counts())
    
    # Save
    df.to_csv(DL_DATA_DIR / "rich_features.csv", index=False)
    print(f"\nSaved to: {DL_DATA_DIR / 'rich_features.csv'}")
    
    # Create numeric feature matrix
    numeric_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64'] and c not in ['group_label']]
    X = df[numeric_cols].values
    y = df['group_label'].values
    
    np.save(DL_DATA_DIR / "rich_features_X.npy", X)
    np.save(DL_DATA_DIR / "rich_features_y.npy", y)
    np.save(DL_DATA_DIR / "rich_features_columns.npy", np.array(numeric_cols))
    
    # Subject IDs for GroupKFold
    subject_ids = df.apply(lambda r: f"{r['group']}_{r['subject']}", axis=1).values
    np.save(DL_DATA_DIR / "rich_features_subjects.npy", subject_ids)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Feature statistics by group
    print(f"\n{'='*60}")
    print("FEATURE MEANS BY GROUP")
    print("="*60)
    
    key_features = [
        'pupil_mean', 'pupil_slope', 'pupil_range',
        'saccade_velocity_mean', 'saccade_velocity_max',
        'fixation_duration_mean', 'fixation_duration_std',
        'ratio_left_right', 'emotional_bias',
        'first_fix_latency', 'total_transitions'
    ]
    
    for feat in key_features:
        if feat in df.columns:
            means = df.groupby('group')[feat].mean()
            stds = df.groupby('group')[feat].std()
            print(f"\n{feat}:")
            for g in ['jeunes', 'moyen', 'age']:
                if g in means.index:
                    print(f"  {g}: {means[g]:.2f} +/- {stds[g]:.2f}")
    
    return df


if __name__ == "__main__":
    df = process_all_subjects()








