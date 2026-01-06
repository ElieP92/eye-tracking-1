# Eye-Tracking Analysis - Emotional Attention & Age Classification

Eye-tracking data analysis pipeline for studying visual attention toward emotional stimuli across the lifespan, with ML/DL age group classification.

## ⚠️ Data Disclaimer

**The data included in this repository is NOT the original dataset from the published research article.** Instead, it consists of:
- A **subset** of the original participant data
- Values that have been **slightly modified** for privacy and demonstration purposes
- Data from **pilot/pretest sessions** used during experiment development

This repository is intended for **educational and methodological demonstration purposes only**. The results obtained with this dataset should not be interpreted as replicating the original study findings.

## Research Question

**Can oculomotor patterns during emotional image viewing discriminate between age groups, and what features are most informative?**

This project investigates age-related differences in visual attention toward emotional stimuli (negative, positive, neutral) using eye-tracking. Beyond traditional inferential statistics, machine learning approaches are used to classify participants into age groups based on their eye movement patterns.

## Key Results

### Classification Performance

| Model | Accuracy | Key Insight |
|-------|----------|-------------|
| **Hybrid DL** | **72.6%** | Optimal balance: 64% features + 36% spatial |
| Gradient Boosting | 69.6% | Best classical ML approach |
| Random Forest | 66.3% | Good for feature importance |

### Most Discriminative Features

| Feature | F-statistic | Interpretation |
|---------|-------------|----------------|
| **Pupil size (mean)** | 888*** | Age-related pupil differences |
| **Pupil range** | 549*** | Reduced dynamic range with age |
| **Saccade velocity** | 56*** | Slower eye movements in older adults |
| **First fixation latency** | 46*** | Processing speed differences |
| **Emotional bias** | 12*** | Shift from negativity to positivity bias |

### Main Findings: Age × Emotion Interactions

1. **Positivity effect in older adults**: Confirmed bias toward positive images in older group
2. **Negativity bias in young adults**: Preferential attention to threatening stimuli
3. **Pupil dynamics**: Largest age differences in pupil response amplitude
4. **Temporal evolution**: Group × Time window interactions in fixation patterns

## Project Structure

```
projet_occulo/
├── code/
│   ├── preprocess_occulo.py      # Raw data parsing & quality control
│   ├── analyse_stats.R           # ANOVA statistical analysis (R)
│   └── deep_learning/
│       ├── extract_rich_features.py  # Hand-crafted feature extraction
│       ├── train_hybrid.py           # Best model (72.6%)
│       ├── train_ml_rich.py          # Classical ML classifiers
│       ├── analyze_emotional_design.py
│       └── explainability.py
├── results/
│   └── deep_learning_results/    # Figures, metrics, reports
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

```bash
python code/preprocess_occulo.py
```

Quality criteria:
- First saccade latency: 80-500 ms
- First saccade amplitude: 40-300 pixels
- First saccade duration: < 100 ms
- Blinks: < 5 per trial

### 2. Statistical Analysis (R)

```bash
Rscript code/analyse_stats.R
```

Mixed ANOVAs: Group (between) × Valence × Arousal × Time Window (within)

### 3. Machine Learning Pipeline

```bash
cd code/deep_learning
python run_pipeline.py
```

## Experimental Design

- **Between-subjects**: Age group (young 18-30, middle-aged 40-55, older 65+)
- **Within-subjects**: 
  - Valence (negative, positive, neutral)
  - Arousal (high/low)
  - Time window (0-2.5s, 2.5-5s)
- **Pair types**: Pos-Neu, Neg-Neu, Pos-Neg
- **Trial duration**: 5 seconds free viewing

## Dependencies

### Python
- Python 3.10+
- pandas, numpy, scipy
- torch, torchvision
- scikit-learn
- matplotlib, seaborn

### R
- R 4.0+
- tidyverse, afex, emmeans
- rstatix, ggpubr

## License

Academic use only.
