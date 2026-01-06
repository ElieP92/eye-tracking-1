# analyze_emotional_design.py
"""
Analyze discriminative features by emotional design:
- Pair types (Positive-Neutral, Negative-Neutral, Positive-Negative)
- Image valence being viewed (positive, negative, neutral)
- Arousal level (HA vs LA)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DL_DATA_DIR = PROJECT_ROOT / "results" / "deep_learning_data"
RESULTS_DIR = PROJECT_ROOT / "results" / "deep_learning_results"

def get_pair_type(left_img: str, right_img: str) -> str:
    """Determine pair type from image names."""
    left = left_img.lower() if isinstance(left_img, str) else ""
    right = right_img.lower() if isinstance(right_img, str) else ""
    
    has_pos = 'pos' in left or 'pos' in right
    has_neg = 'neg' in left or 'neg' in right
    has_neu = 'neu' in left or 'neu' in right
    
    if has_pos and has_neg:
        return 'Pos-Neg'
    elif has_pos and has_neu:
        return 'Pos-Neu'
    elif has_neg and has_neu:
        return 'Neg-Neu'
    else:
        return 'Other'


def get_viewed_valence(fixations_left: int, fixations_right: int, 
                       left_valence: str, right_valence: str) -> str:
    """Determine which valence was predominantly viewed."""
    if fixations_left > fixations_right:
        return left_valence
    elif fixations_right > fixations_left:
        return right_valence
    else:
        return 'equal'


def analyze_by_emotional_conditions():
    """Analyze features by emotional design conditions."""
    print("="*70)
    print("ANALYSIS BY EMOTIONAL DESIGN")
    print("="*70)
    
    df = pd.read_csv(DL_DATA_DIR / "rich_features.csv")
    
    # Add pair type
    df['pair_type'] = df.apply(
        lambda r: get_pair_type(r.get('left_valence', ''), r.get('right_valence', '')), 
        axis=1
    )
    
    # Add predominantly viewed valence
    df['viewed_valence'] = df.apply(
        lambda r: get_viewed_valence(
            r.get('n_fix_left', 0), r.get('n_fix_right', 0),
            r.get('left_valence', ''), r.get('right_valence', '')
        ), axis=1
    )
    
    # Key features to analyze
    key_features = [
        'pupil_mean', 'pupil_range', 'pupil_slope',
        'saccade_velocity_mean', 'saccade_velocity_max',
        'fixation_duration_mean', 'emotional_bias',
        'first_fix_latency', 'total_transitions',
        'dwell_time_left', 'dwell_time_right'
    ]
    
    # =========================================
    # 1. ANALYSIS BY PAIR TYPE
    # =========================================
    print("\n" + "="*70)
    print("1. FEATURES BY PAIR TYPE")
    print("="*70)
    
    pair_types = ['Pos-Neu', 'Neg-Neu', 'Pos-Neg']
    groups = ['jeunes', 'moyen', 'age']
    
    results_by_pair = {}
    
    for pair in pair_types:
        print(f"\n--- {pair} ---")
        df_pair = df[df['pair_type'] == pair]
        
        if len(df_pair) == 0:
            continue
        
        results_by_pair[pair] = {}
        
        for feat in ['pupil_mean', 'emotional_bias', 'first_fix_latency']:
            if feat not in df_pair.columns:
                continue
            
            group_data = [df_pair[df_pair['group'] == g][feat].dropna() for g in groups]
            
            if any(len(g) < 10 for g in group_data):
                continue
            
            f_stat, p_val = stats.f_oneway(*group_data)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            
            means = {g: df_pair[df_pair['group'] == g][feat].mean() for g in groups}
            
            print(f"  {feat}: F={f_stat:.2f}, p={p_val:.4f} {sig}")
            print(f"    Jeunes: {means['jeunes']:.2f}, Moyen: {means['moyen']:.2f}, Âgés: {means['age']:.2f}")
            
            results_by_pair[pair][feat] = {
                'f_stat': f_stat, 'p_val': p_val, 'means': means
            }
    
    # =========================================
    # 2. EMOTIONAL BIAS BY GROUP AND PAIR TYPE
    # =========================================
    print("\n" + "="*70)
    print("2. EMOTIONAL ATTENTION BIAS BY GROUP")
    print("="*70)
    
    # Calculate preference for emotional vs neutral
    for pair in pair_types:
        df_pair = df[df['pair_type'] == pair]
        
        if len(df_pair) == 0:
            continue
        
        print(f"\n--- {pair} ---")
        print("Dwell time on emotional image (ms):")
        
        for g in groups:
            df_g = df_pair[df_pair['group'] == g]
            emotional_dwell = df_g['dwell_emotional'].mean()
            neutral_dwell = df_g['dwell_neutral'].mean()
            bias = emotional_dwell - neutral_dwell
            
            direction = "→ emotional" if bias > 0 else "→ neutral"
            print(f"  {g:8s}: Emo={emotional_dwell:.0f}, Neu={neutral_dwell:.0f}, Bias={bias:+.0f} {direction}")
    
    # =========================================
    # 3. PUPILLARY RESPONSE BY VALENCE VIEWED
    # =========================================
    print("\n" + "="*70)
    print("3. PUPILLARY RESPONSE BY VIEWED VALENCE")
    print("="*70)
    
    valences = ['positive', 'negative', 'neutral']
    
    pupil_by_valence = defaultdict(lambda: defaultdict(list))
    
    for _, row in df.iterrows():
        viewed = row.get('viewed_valence', '')
        if viewed in valences:
            pupil_by_valence[row['group']][viewed].append(row['pupil_mean'])
    
    print("\nMean pupil size when predominantly viewing each valence:")
    print("-" * 60)
    print(f"{'Group':10s} {'Positive':>12s} {'Negative':>12s} {'Neutral':>12s}")
    print("-" * 60)
    
    for g in groups:
        vals = []
        for v in valences:
            data = pupil_by_valence[g][v]
            vals.append(np.mean(data) if data else 0)
        print(f"{g:10s} {vals[0]:>12.1f} {vals[1]:>12.1f} {vals[2]:>12.1f}")
    
    # =========================================
    # 4. FIRST SACCADE DIRECTION BY PAIR TYPE
    # =========================================
    print("\n" + "="*70)
    print("4. FIRST FIXATION PREFERENCE BY PAIR TYPE")
    print("="*70)
    
    for pair in pair_types:
        df_pair = df[df['pair_type'] == pair]
        
        if len(df_pair) == 0:
            continue
        
        print(f"\n--- {pair} ---")
        print("First fixation toward left image (%):")
        
        for g in groups:
            df_g = df_pair[df_pair['group'] == g]
            # Use ratio_left_right as proxy
            pct_left = (df_g['n_fix_left'] > df_g['n_fix_right']).mean() * 100
            print(f"  {g:8s}: {pct_left:.1f}%")
    
    # =========================================
    # 5. STATISTICAL TESTS: Group × Pair Type Interaction
    # =========================================
    print("\n" + "="*70)
    print("5. GROUP × PAIR TYPE INTERACTION (2-way ANOVA)")
    print("="*70)
    
    # For each key feature, test Group × PairType interaction
    from scipy.stats import f_oneway
    
    for feat in ['pupil_mean', 'emotional_bias', 'first_fix_latency']:
        if feat not in df.columns:
            continue
        
        print(f"\n{feat}:")
        
        # Simple approach: compare group effect within each pair type
        for pair in pair_types:
            df_pair = df[df['pair_type'] == pair]
            group_data = [df_pair[df_pair['group'] == g][feat].dropna() for g in groups]
            
            if any(len(g) < 10 for g in group_data):
                continue
            
            f_stat, p_val = stats.f_oneway(*group_data)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"  {pair}: F={f_stat:.2f}, p={p_val:.4f} {sig}")
    
    # =========================================
    # 6. VISUALIZATION
    # =========================================
    print("\n" + "="*70)
    print("6. GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Figure 1: Pupil size by group and pair type
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, pair in enumerate(pair_types):
        ax = axes[i]
        df_pair = df[df['pair_type'] == pair]
        
        data = [df_pair[df_pair['group'] == g]['pupil_mean'].dropna() for g in groups]
        
        bp = ax.boxplot(data, tick_labels=['Young', 'Middle', 'Older'])
        ax.set_title(f'{pair} Pairs')
        ax.set_ylabel('Pupil Size')
    
    plt.suptitle('Pupil Size by Age Group and Pair Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "emotional_pupil_by_pair.png", dpi=150)
    plt.close()
    
    # Figure 2: Emotional bias by group and pair type
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bias_data = []
    for g in groups:
        for pair in pair_types:
            df_sub = df[(df['group'] == g) & (df['pair_type'] == pair)]
            bias = df_sub['emotional_bias'].mean()
            bias_data.append({'Group': g, 'Pair Type': pair, 'Emotional Bias': bias})
    
    bias_df = pd.DataFrame(bias_data)
    
    x = np.arange(len(pair_types))
    width = 0.25
    
    for i, g in enumerate(groups):
        values = bias_df[bias_df['Group'] == g]['Emotional Bias'].values
        ax.bar(x + i*width, values, width, label=g.capitalize())
    
    ax.set_xlabel('Pair Type')
    ax.set_ylabel('Emotional Bias (ms)')
    ax.set_title('Emotional Attention Bias by Age Group')
    ax.set_xticks(x + width)
    ax.set_xticklabels(pair_types)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "emotional_bias_by_group.png", dpi=150)
    plt.close()
    
    # Figure 3: First fixation latency
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, pair in enumerate(pair_types):
        ax = axes[i]
        df_pair = df[df['pair_type'] == pair]
        
        data = [df_pair[df_pair['group'] == g]['first_fix_latency'].dropna() for g in groups]
        
        bp = ax.boxplot(data, tick_labels=['Young', 'Middle', 'Older'])
        ax.set_title(f'{pair} Pairs')
        ax.set_ylabel('First Fix Latency (ms)')
    
    plt.suptitle('First Fixation Latency by Age Group and Pair Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "emotional_latency_by_pair.png", dpi=150)
    plt.close()
    
    print("Figures saved!")
    
    return df


if __name__ == "__main__":
    df = analyze_by_emotional_conditions()






