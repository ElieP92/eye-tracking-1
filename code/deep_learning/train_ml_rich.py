# train_ml_rich.py
"""
Traditional ML classifiers on rich features.
Tests: Random Forest, Gradient Boosting, SVM, MLP
With feature importance analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DL_DATA_DIR = PROJECT_ROOT / "results" / "deep_learning_data"
RESULTS_DIR = PROJECT_ROOT / "results" / "deep_learning_results"
RESULTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ['Young', 'Middle-aged', 'Older']


def load_rich_features():
    """Load rich features from CSV."""
    df = pd.read_csv(DL_DATA_DIR / "rich_features.csv")
    
    # Numeric features only
    exclude_cols = ['group', 'group_label', 'subject', 'file', 'left_valence', 'right_valence']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['group_label'].values
    subjects = df.apply(lambda r: f"{r['group']}_{r['subject']}", axis=1).values
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X, y, subjects, feature_cols


def train_and_evaluate():
    """Train multiple classifiers and compare."""
    print("="*60)
    print("ML CLASSIFIERS ON RICH FEATURES")
    print("="*60)
    
    X, y, subjects, feature_names = load_rich_features()
    
    print(f"\nData shape: {X.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection (top 20 features)
    print("\n--- Feature Selection (ANOVA F-value) ---")
    selector = SelectKBest(f_classif, k=min(20, len(feature_names)))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [f for f, s in zip(feature_names, selected_mask) if s]
    print(f"Selected features ({len(selected_features)}):")
    
    # Feature scores
    scores = selector.scores_
    feature_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    for f, s in feature_scores[:15]:
        print(f"  {f}: {s:.2f}")
    
    # Classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', C=1, gamma='scale',
            class_weight='balanced', random_state=42
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            early_stopping=True, random_state=42
        )
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")
        
        # Train on selected features
        clf.fit(X_train_selected, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_selected)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        
        results[name] = {
            'accuracy': acc,
            'predictions': y_pred.tolist(),
            'true': y_test.tolist()
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{name} (Acc={acc:.3f})')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"ml_{name.lower().replace(' ', '_')}_cm.png", dpi=150)
        plt.close()
    
    # Cross-validation on best model
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-fold)")
    print("="*60)
    
    best_clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    cv_scores = cross_val_score(best_clf, X_train_selected, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    # Feature importance from Random Forest
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*60)
    
    rf = classifiers['Random Forest']
    importances = rf.feature_importances_
    
    # Map to selected features
    selected_importances = [(f, imp) for f, imp, sel in 
                           zip(feature_names, 
                               [importances[i] if i < len(importances) else 0 
                                for i in range(len(feature_names))],
                               selected_mask) if sel]
    
    # Sort by importance
    # Get actual importances for selected features
    selected_indices = np.where(selected_mask)[0]
    feat_imp = [(selected_features[i], importances[i]) for i in range(len(selected_features))]
    feat_imp.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop features:")
    for f, imp in feat_imp[:10]:
        print(f"  {f}: {imp:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    features = [f for f, _ in feat_imp[:15]]
    imps = [i for _, i in feat_imp[:15]]
    
    plt.barh(range(len(features)), imps)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ml_feature_importance.png", dpi=150)
    plt.close()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"  {name}: {res['accuracy']:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model: {best_model[0]} ({best_model[1]['accuracy']:.4f})")
    
    # Save results
    with open(RESULTS_DIR / "ml_rich_results.json", 'w') as f:
        json.dump({k: {'accuracy': v['accuracy']} for k, v in results.items()}, f, indent=2)
    
    return results


def analyze_discriminative_features():
    """Detailed analysis of which features discriminate groups."""
    print("\n" + "="*60)
    print("DISCRIMINATIVE FEATURE ANALYSIS")
    print("="*60)
    
    df = pd.read_csv(DL_DATA_DIR / "rich_features.csv")
    
    # Key features to analyze
    key_features = [
        'pupil_mean', 'pupil_slope', 'pupil_range',
        'saccade_velocity_mean', 'saccade_velocity_max',
        'fixation_duration_mean', 'fixation_duration_std',
        'cv_fixation_duration', 'spatial_spread_x',
        'emotional_bias', 'first_fix_latency',
        'total_transitions', 'ratio_early_late'
    ]
    
    # Statistical tests
    from scipy import stats
    
    print("\nANOVA F-test (Young vs Middle vs Older):")
    print("-" * 50)
    
    significant_features = []
    
    for feat in key_features:
        if feat not in df.columns:
            continue
        
        groups = [df[df['group'] == g][feat].dropna() for g in ['jeunes', 'moyen', 'age']]
        
        if any(len(g) == 0 for g in groups):
            continue
        
        f_stat, p_val = stats.f_oneway(*groups)
        
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        if p_val < 0.05:
            significant_features.append((feat, f_stat, p_val))
        
        print(f"{feat:30s}: F={f_stat:8.2f}, p={p_val:.4f} {sig}")
    
    print("\nSignificant features for age group discrimination:")
    for feat, f, p in sorted(significant_features, key=lambda x: x[1], reverse=True):
        print(f"  {feat}: F={f:.2f}, p={p:.4f}")
    
    # Visualize top discriminative features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    top_features = [f for f, _, _ in sorted(significant_features, key=lambda x: x[1], reverse=True)[:6]]
    
    for i, feat in enumerate(top_features):
        if i >= len(axes):
            break
        ax = axes[i]
        
        data = [df[df['group'] == g][feat].dropna() for g in ['jeunes', 'moyen', 'age']]
        
        bp = ax.boxplot(data, labels=['Young', 'Middle', 'Older'])
        ax.set_title(feat)
        ax.set_ylabel('Value')
    
    plt.suptitle('Top Discriminative Features by Age Group', fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ml_discriminative_features.png", dpi=150)
    plt.close()
    
    return significant_features


if __name__ == "__main__":
    # First extract features if not done
    if not (DL_DATA_DIR / "rich_features.csv").exists():
        print("Extracting features first...")
        from extract_rich_features import process_all_subjects
        process_all_subjects()
    
    # Train and evaluate
    results = train_and_evaluate()
    
    # Analyze discriminative features
    analyze_discriminative_features()






