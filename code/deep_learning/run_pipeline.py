# run_pipeline.py
"""
Main script to run the complete deep learning pipeline:
1. Extract fixation data and generate heatmaps/sequences
2. Train CNN on heatmaps
3. Train Transformer/LSTM on scanpaths
4. Generate explainability visualizations
5. Create comprehensive report
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def run_simple_pipeline():
    """Run pipeline with simple train/test split (no cross-validation)."""
    print("="*70)
    print(" DEEP LEARNING PIPELINE - SIMPLE SPLIT (NO CV)")
    print("="*70)
    
    # Step 1: Extract data
    print("\n" + "="*70)
    print(" STEP 1: DATA EXTRACTION")
    print("="*70)
    
    from extract_fixations import process_all_data
    heatmaps, sequences, labels = process_all_data()
    
    if len(labels) == 0:
        print("ERROR: No data extracted. Check raw data paths.")
        return
    
    # Step 2: Train CNN
    print("\n" + "="*70)
    print(" STEP 2: CNN TRAINING (HEATMAPS)")
    print("="*70)
    
    from train_cnn import train_simple as train_cnn
    cnn_results = train_cnn()
    
    # Step 3: Train Transformer and LSTM
    print("\n" + "="*70)
    print(" STEP 3: SEQUENCE MODELS (SCANPATHS)")
    print("="*70)
    
    from train_transformer import train_simple as train_seq
    transformer_results = train_seq('transformer')
    lstm_results = train_seq('lstm')
    
    # Step 4: Explainability
    print("\n" + "="*70)
    print(" STEP 4: EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    from explainability import generate_report
    generate_report()
    
    # Final Summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)
    print(f"\nModel Performance (Simple 80/20 Split):")
    print(f"  CNN (Heatmaps):     {cnn_results['best_accuracy']:.1%}")
    print(f"  Transformer:        {transformer_results['best_accuracy']:.1%}")
    print(f"  BiLSTM:             {lstm_results['best_accuracy']:.1%}")
    
    print(f"\nBest model: ", end="")
    best = max([
        ('CNN', cnn_results['best_accuracy']),
        ('Transformer', transformer_results['best_accuracy']),
        ('BiLSTM', lstm_results['best_accuracy'])
    ], key=lambda x: x[1])
    print(f"{best[0]} ({best[1]:.1%})")
    
    print(f"\nResults directory: {SCRIPT_DIR.parent.parent / 'results' / 'deep_learning_results'}")
    print("="*70)


def run_cv_pipeline():
    """Run pipeline with Leave-Subject-Out cross-validation."""
    print("="*70)
    print(" DEEP LEARNING PIPELINE - CROSS-VALIDATION")
    print("="*70)
    
    # Step 1: Extract data
    print("\n" + "="*70)
    print(" STEP 1: DATA EXTRACTION")
    print("="*70)
    
    from extract_fixations import process_all_data
    heatmaps, sequences, labels = process_all_data()
    
    if len(labels) == 0:
        print("ERROR: No data extracted. Check raw data paths.")
        return
    
    # Step 2: Train CNN
    print("\n" + "="*70)
    print(" STEP 2: CNN TRAINING (HEATMAPS)")
    print("="*70)
    
    from train_cnn import train_with_cv as train_cnn
    cnn_results = train_cnn()
    
    # Step 3: Train Transformer and LSTM
    print("\n" + "="*70)
    print(" STEP 3: SEQUENCE MODELS (SCANPATHS)")
    print("="*70)
    
    from train_transformer import train_with_cv as train_seq
    transformer_results = train_seq('transformer')
    lstm_results = train_seq('lstm')
    
    # Step 4: Explainability
    print("\n" + "="*70)
    print(" STEP 4: EXPLAINABILITY ANALYSIS")
    print("="*70)
    
    from explainability import generate_report
    generate_report()
    
    # Final Summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)
    print(f"\nModel Performance (Leave-Subject-Out CV):")
    print(f"  CNN (Heatmaps):     {cnn_results['overall_accuracy']:.1%}")
    print(f"  Transformer:        {transformer_results['overall_accuracy']:.1%}")
    print(f"  BiLSTM:             {lstm_results['overall_accuracy']:.1%}")
    
    print(f"\nBest model: ", end="")
    best = max([
        ('CNN', cnn_results['overall_accuracy']),
        ('Transformer', transformer_results['overall_accuracy']),
        ('BiLSTM', lstm_results['overall_accuracy'])
    ], key=lambda x: x[1])
    print(f"{best[0]} ({best[1]:.1%})")
    
    print(f"\nResults directory: {SCRIPT_DIR.parent.parent / 'results' / 'deep_learning_results'}")
    print("="*70)


def run_quick_test():
    """Quick test with reduced epochs."""
    print("Running quick test mode...")
    
    import train_cnn
    import train_transformer
    
    train_cnn.EPOCHS = 10
    train_cnn.PATIENCE = 5
    train_transformer.EPOCHS = 10
    train_transformer.PATIENCE = 5
    
    run_simple_pipeline()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Learning Pipeline for Eye-Tracking')
    parser.add_argument('--cv', action='store_true', help='Use cross-validation (default: simple split)')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (10 epochs)')
    parser.add_argument('--extract-only', action='store_true', help='Only extract data')
    parser.add_argument('--cnn-only', action='store_true', help='Only train CNN')
    parser.add_argument('--transformer-only', action='store_true', help='Only train Transformer')
    
    args = parser.parse_args()
    
    if args.extract_only:
        from extract_fixations import process_all_data
        process_all_data()
    elif args.cnn_only:
        from extract_fixations import process_all_data
        process_all_data()
        from train_cnn import train_simple
        train_simple()
    elif args.transformer_only:
        from extract_fixations import process_all_data
        process_all_data()
        from train_transformer import train_simple
        train_simple('transformer')
    elif args.quick:
        run_quick_test()
    elif args.cv:
        run_cv_pipeline()
    else:
        # Default: simple split (no CV)
        run_simple_pipeline()
