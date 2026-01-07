# Example Data

This folder contains sample files to test the preprocessing pipeline.

## Files

- `0011.asc` - EyeLink eye-tracking data (ASCII export from .edf)
- `HAconfig1_V4_withOculo-01-32.txt` - E-Prime behavioral log file

## Usage

To test the preprocessing pipeline with this example data, you can modify the paths in `preprocess_occulo.py` or copy these files to the expected folder structure:

```
data/
└── jeunes/
    └── 1/
        ├── 0011.asc
        └── HAconfig1_V4_withOculo-01-32.txt
```

Then run:
```bash
python code/preprocess_occulo.py
```

## File Format

### .asc file (EyeLink)
- Contains fixation events (EFIX), saccade events (ESACC), blink events (EBLINK)
- Message markers for trial structure (Image_Start, pictureTrial_Offset)
- Stimulus information embedded in messages

### .txt file (E-Prime)
- Contains trial-level behavioral data
- Stimulus conditions (valence, arousal)
- Timing information

