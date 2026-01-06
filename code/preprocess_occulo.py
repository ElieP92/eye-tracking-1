# preprocess_occulo.py
"""
Eye-tracking data preprocessing for projet_occulo.

Methodology based on thesis:
- First saccade latency: 80-500 ms
- First saccade amplitude: 40-300 pixels
- First saccade duration: < 100 ms
- Blinks: < 5 per trial (5 seconds)

Usage:
    python preprocess_occulo.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_ROOT = PROJECT_ROOT.parent  # C:\Users\eliep\Desktop\occulo\

OUT_DIR = PROJECT_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_SESSION_DIR = OUT_DIR / "per_session"
PER_SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Group name mapping
GROUP_MAPPING = {
    "agé": "age",
    "age": "age",
    "âgé": "age",
    "moyen": "moyen",
    "jeunes": "jeunes",
    "jeune": "jeunes"
}

# Screen parameters (from thesis)
SCREEN_WIDTH_PX = 1024
SCREEN_HEIGHT_PX = 768
SCREEN_CENTER_X = SCREEN_WIDTH_PX / 2  # 512
SCREEN_CENTER_Y = SCREEN_HEIGHT_PX / 2  # 384

# Image parameters (from thesis)
# Each image: 640x480 pixels, 61 pixels gap between images
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
GAP_BETWEEN_IMAGES = 61

# Image positions on screen
# Total width of 2 images + gap = 640 + 61 + 640 = 1341
# But screen is 1024, so images might be smaller or centered differently
# Using AOIs based on screen center
IMAGE_LEFT_CENTER_X = SCREEN_CENTER_X - GAP_BETWEEN_IMAGES/2 - IMAGE_WIDTH/2  # ~161
IMAGE_RIGHT_CENTER_X = SCREEN_CENTER_X + GAP_BETWEEN_IMAGES/2 + IMAGE_WIDTH/2  # ~862

# =============================================================================
# QUALITY CRITERIA (from thesis)
# =============================================================================

# For first saccade rate
MIN_SAC1_LAT_MS = 80      # minimum latency
MAX_SAC1_LAT_MS = 500     # maximum latency
MIN_SAC1_AMP_PX = 40      # minimum amplitude in PIXELS (inner edge of image - center)
MAX_SAC1_AMP_PX = 300     # maximum amplitude in PIXELS (outer edge of image - center)
MAX_SAC1_DUR_MS = 100     # maximum duration

# Threshold to consider a saccade as "toward an image" (not micro-saccade)
# A saccade toward an image must start from center and have significant amplitude
MIN_SACCADE_TO_IMAGE_AMP = 20  # at least 20px to be considered

# For fixations
MAX_BLINKS_PER_TRIAL = 5  # exclude if >= 5 blinks

# AOI (Areas of Interest) - left/right images
# Central zone (fixation cross): about 100px around center
FIXATION_ZONE_RADIUS = 100
AOI_CENTER = {
    "x_min": SCREEN_CENTER_X - FIXATION_ZONE_RADIUS,
    "x_max": SCREEN_CENTER_X + FIXATION_ZONE_RADIUS,
    "y_min": SCREEN_CENTER_Y - FIXATION_ZONE_RADIUS,
    "y_max": SCREEN_CENTER_Y + FIXATION_ZONE_RADIUS
}
AOI_LEFT = {"x_min": 0, "x_max": SCREEN_CENTER_X - GAP_BETWEEN_IMAGES/2, "y_min": 0, "y_max": SCREEN_HEIGHT_PX}
AOI_RIGHT = {"x_min": SCREEN_CENTER_X + GAP_BETWEEN_IMAGES/2, "x_max": SCREEN_WIDTH_PX, "y_min": 0, "y_max": SCREEN_HEIGHT_PX}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Fixation:
    eye: str
    start_time: int
    end_time: int
    duration: int
    x: float
    y: float
    pupil: float = 0.0


@dataclass
class Saccade:
    eye: str
    start_time: int
    end_time: int
    duration: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    amplitude_deg: float  # in degrees (from file)
    amplitude_px: float   # in pixels (calculated)
    velocity: float


@dataclass
class Blink:
    eye: str
    start_time: int
    end_time: int
    duration: int


@dataclass 
class Message:
    time: int
    text: str


# =============================================================================
# EYELINK .ASC PARSER
# =============================================================================

class EyeLinkASCParser:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.fixations: List[Fixation] = []
        self.saccades: List[Saccade] = []
        self.blinks: List[Blink] = []
        self.messages: List[Message] = []
        self.metadata: Dict = {}
        
    def parse(self) -> bool:
        try:
            with open(self.filepath, 'r', encoding='latin-1', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {self.filepath}: {e}")
            return False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if line.startswith("**"):
                self._parse_metadata(line)
            elif parts[0] == "MSG":
                msg = self._parse_message(parts)
                if msg:
                    self.messages.append(msg)
            elif parts[0] == "EFIX":
                fix = self._parse_efix(parts)
                if fix:
                    self.fixations.append(fix)
            elif parts[0] == "ESACC":
                sac = self._parse_esacc(parts)
                if sac:
                    self.saccades.append(sac)
            elif parts[0] == "EBLINK":
                blink = self._parse_eblink(parts)
                if blink:
                    self.blinks.append(blink)
                    
        return True
    
    def _parse_metadata(self, line: str):
        line = line.lstrip("*").strip()
        if ":" in line:
            key, _, value = line.partition(":")
            self.metadata[key.strip()] = value.strip()
    
    def _parse_message(self, parts: List[str]) -> Optional[Message]:
        if len(parts) < 2:
            return None
        try:
            time = int(parts[1])
            text = " ".join(parts[2:]) if len(parts) > 2 else ""
            return Message(time=time, text=text)
        except ValueError:
            return None
    
    def _parse_efix(self, parts: List[str]) -> Optional[Fixation]:
        # EFIX L start end dur x y pupil
        if len(parts) < 7:
            return None
        try:
            return Fixation(
                eye=parts[1],
                start_time=int(parts[2]),
                end_time=int(parts[3]),
                duration=int(parts[4]),
                x=float(parts[5]),
                y=float(parts[6]),
                pupil=float(parts[7]) if len(parts) > 7 else 0.0
            )
        except (ValueError, IndexError):
            return None
    
    def _parse_esacc(self, parts: List[str]) -> Optional[Saccade]:
        # ESACC L start end dur sx sy ex ey amp vel
        if len(parts) < 10:
            return None
        try:
            start_x = float(parts[5])
            start_y = float(parts[6])
            end_x = float(parts[7])
            end_y = float(parts[8])
            
            # Calculate amplitude in PIXELS
            amplitude_px = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            return Saccade(
                eye=parts[1],
                start_time=int(parts[2]),
                end_time=int(parts[3]),
                duration=int(parts[4]),
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                amplitude_deg=float(parts[9]),
                amplitude_px=amplitude_px,
                velocity=float(parts[10]) if len(parts) > 10 else 0.0
            )
        except (ValueError, IndexError):
            return None
    
    def _parse_eblink(self, parts: List[str]) -> Optional[Blink]:
        if len(parts) < 4:
            return None
        try:
            start = int(parts[2])
            end = int(parts[3])
            return Blink(
                eye=parts[1],
                start_time=start,
                end_time=end,
                duration=int(parts[4]) if len(parts) > 4 else end - start
            )
        except (ValueError, IndexError):
            return None


# =============================================================================
# E-PRIME .TXT PARSER
# =============================================================================

class EPrimeParser:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.header: Dict = {}
        self.trials: List[Dict] = []
        
    def parse(self) -> bool:
        content = None
        for encoding in ['utf-16', 'utf-16-le', 'utf-8', 'latin-1']:
            try:
                with open(self.filepath, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                if 'LogFrame' in content or 'Header' in content:
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            return False
        
        # Parse header
        header_match = re.search(r'[\t ]*\*\*\*\s*Header Start\s*\*\*\*(.*?)[\t ]*\*\*\*\s*Header End\s*\*\*\*', 
                                  content, re.DOTALL | re.IGNORECASE)
        if header_match:
            self._parse_header(header_match.group(1))
        
        # Parse LogFrames
        frames = re.findall(r'[\t ]*\*\*\*\s*LogFrame Start\s*\*\*\*(.*?)[\t ]*\*\*\*\s*LogFrame End\s*\*\*\*',
                           content, re.DOTALL | re.IGNORECASE)
        
        for i, frame in enumerate(frames):
            trial_data = self._parse_logframe(frame)
            trial_data['TrialNum'] = i + 1
            self.trials.append(trial_data)
            
        return True
    
    def _parse_header(self, header_text: str):
        for line in header_text.strip().split('\n'):
            line = line.strip()
            if ':' in line:
                key, _, value = line.partition(':')
                self.header[key.strip()] = value.strip()
    
    def _parse_logframe(self, frame_text: str) -> Dict:
        data = {}
        for line in frame_text.strip().split('\n'):
            line = line.strip()
            if ':' in line:
                key, _, value = line.partition(':')
                key = key.strip()
                value = value.strip()
                try:
                    if '.' in value:
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value
        return data


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_valence(stimulus_name: str) -> str:
    """Extract valence from stimulus name."""
    name_lower = stimulus_name.lower()
    if 'neg' in name_lower:
        return 'neg'
    elif 'pos' in name_lower:
        return 'pos'
    elif 'neu' in name_lower:
        return 'neu'
    return 'unknown'


def get_saccade_direction(saccade: Saccade) -> str:
    """Determine saccade direction based on final position."""
    # Direction based on where the saccade ends (in which AOI)
    if point_in_aoi(saccade.end_x, saccade.end_y, AOI_LEFT):
        return 'left'
    elif point_in_aoi(saccade.end_x, saccade.end_y, AOI_RIGHT):
        return 'right'
    return 'center'


def is_saccade_to_image(saccade: Saccade) -> bool:
    """
    Check if a saccade is a true saccade toward an image.
    
    Criteria:
    - Starts from central zone (near fixation cross)
    - Has significant horizontal amplitude (not a micro-saccade)
    - Ends in an image zone (left or right)
    """
    # Check that saccade starts from center (or nearby)
    starts_from_center = (
        abs(saccade.start_x - SCREEN_CENTER_X) < FIXATION_ZONE_RADIUS * 1.5 and
        abs(saccade.start_y - SCREEN_CENTER_Y) < FIXATION_ZONE_RADIUS * 1.5
    )
    
    # Check that saccade has significant horizontal movement
    horizontal_movement = abs(saccade.end_x - saccade.start_x)
    has_significant_movement = horizontal_movement >= MIN_SACCADE_TO_IMAGE_AMP
    
    # Check that saccade ends in an image zone
    ends_in_image = (
        point_in_aoi(saccade.end_x, saccade.end_y, AOI_LEFT) or
        point_in_aoi(saccade.end_x, saccade.end_y, AOI_RIGHT)
    )
    
    return starts_from_center and has_significant_movement and ends_in_image


def find_first_saccade_to_image(saccades: List[Saccade], trial_start: int) -> Optional[Saccade]:
    """
    Find the first saccade that goes toward an image.
    
    Ignores micro-saccades and saccades that don't go toward images.
    """
    for sac in saccades:
        if sac.start_time < trial_start:
            continue
        if is_saccade_to_image(sac):
            return sac
    return None


def point_in_aoi(x: float, y: float, aoi: Dict) -> bool:
    return (aoi['x_min'] <= x <= aoi['x_max'] and 
            aoi['y_min'] <= y <= aoi['y_max'])


# =============================================================================
# TRIAL METRICS CALCULATION
# =============================================================================

def calculate_trial_metrics(
    trial_num: int,
    start_time: int,
    end_time: int,
    fixations: List[Fixation],
    saccades: List[Saccade],
    blinks: List[Blink],
    stim_info: Dict,
    eprime_data: Optional[Dict]
) -> Dict:
    """Calculate metrics for a trial."""
    
    metrics = {
        'trial_num': trial_num,
        'trial_start': start_time,
        'trial_end': end_time,
        'trial_duration': end_time - start_time,
        
        # Counts
        'n_fixations': len(fixations),
        'n_saccades': len(saccades),
        'n_blinks': len(blinks),
        
        # First saccade
        'sac1_latency': np.nan,
        'sac1_amplitude_px': np.nan,
        'sac1_amplitude_deg': np.nan,
        'sac1_duration': np.nan,
        'sac1_direction': '',
        'sac1_velocity': np.nan,
        'sac1_start_x': np.nan,
        'sac1_start_y': np.nan,
        'sac1_end_x': np.nan,
        'sac1_end_y': np.nan,
        
        # Saccades 2-5
        'sac2_direction': '', 'sac2_latency': np.nan, 'sac2_amplitude': np.nan, 'sac2_duration': np.nan,
        'sac3_direction': '', 'sac3_latency': np.nan, 'sac3_amplitude': np.nan, 'sac3_duration': np.nan,
        'sac4_direction': '', 'sac4_latency': np.nan, 'sac4_amplitude': np.nan, 'sac4_duration': np.nan,
        'sac5_direction': '', 'sac5_latency': np.nan, 'sac5_amplitude': np.nan, 'sac5_duration': np.nan,
        
        # Fixations by hemispace
        'n_fix_left': 0,
        'n_fix_right': 0,
        'dur_fix_left': 0,
        'dur_fix_right': 0,
        'first_fix_excluded': False,  # To exclude 1st fixation (left bias)
        
        # Saccades by image
        'n_saccades_left': 0,
        'n_saccades_right': 0,
        
        # Visits per image (number of times gaze returns to image)
        'n_visites_left': 0,
        'n_visites_right': 0,
        
        # Expansion (sum of distances: fixations - image center)
        'expansion_left': 0.0,
        'expansion_right': 0.0,
        
        # Uncounted saccade (first saccade stays at center)
        'sac_non_comptee': 0,
        
        # Pupil
        'mean_pupil': np.nan,
        
        # Stimuli
        'stim_gauche': stim_info.get('stim_gauche', ''),
        'stim_droit': stim_info.get('stim_droit', ''),
        'val_gauche': stim_info.get('val_gauche', ''),
        'val_droit': stim_info.get('val_droit', ''),
        
        # E-Prime conditions
        'arousal': '',
        'procedure': '',
        'pair_type': '',
        
        # Result
        'first_saccade_valence': '',
        'first_saccade_hemispace': '',
    }
    
    # Filter trial saccades
    trial_saccades = [s for s in saccades if s.start_time >= start_time]
    
    # First saccade (according to thesis methodology: take the actual first saccade)
    # Micro-saccades will be flagged as invalid by quality criteria
    first_sac = trial_saccades[0] if trial_saccades else None
    
    if first_sac:
        metrics['sac1_latency'] = first_sac.start_time - start_time
        metrics['sac1_amplitude_px'] = first_sac.amplitude_px
        metrics['sac1_amplitude_deg'] = first_sac.amplitude_deg
        metrics['sac1_duration'] = first_sac.duration
        metrics['sac1_velocity'] = first_sac.velocity
        metrics['sac1_start_x'] = first_sac.start_x
        metrics['sac1_start_y'] = first_sac.start_y
        metrics['sac1_end_x'] = first_sac.end_x
        metrics['sac1_end_y'] = first_sac.end_y
        
        # Direction based on final position
        direction = get_saccade_direction(first_sac)
        metrics['sac1_direction'] = direction
        metrics['first_saccade_hemispace'] = direction
        
        # First saccade valence
        if direction == 'left':
            metrics['first_saccade_valence'] = metrics['val_gauche']
        elif direction == 'right':
            metrics['first_saccade_valence'] = metrics['val_droit']
        
        # Uncounted saccade if stays at center
        if direction == 'center':
            metrics['sac_non_comptee'] = 1
    
    # Saccades 2-5
    for i, sac in enumerate(trial_saccades[1:5], start=2):
        direction = get_saccade_direction(sac)
        # Convert 'left'/'right'/'center' to 'L'/'R'/'C'
        dir_code = direction[0].upper() if direction else ''
        metrics[f'sac{i}_direction'] = dir_code
        metrics[f'sac{i}_latency'] = sac.start_time - start_time
        metrics[f'sac{i}_amplitude'] = sac.amplitude_px
        metrics[f'sac{i}_duration'] = sac.duration
    
    # Count saccades by image
    for sac in trial_saccades:
        direction = get_saccade_direction(sac)
        if direction == 'left':
            metrics['n_saccades_left'] += 1
        elif direction == 'right':
            metrics['n_saccades_right'] += 1
    
    # Image centers for expansion calculation
    # Left image: approximate center based on AOI
    image_left_center_x = (AOI_LEFT['x_min'] + AOI_LEFT['x_max']) / 2
    image_left_center_y = SCREEN_CENTER_Y
    # Right image: approximate center based on AOI
    image_right_center_x = (AOI_RIGHT['x_min'] + AOI_RIGHT['x_max']) / 2
    image_right_center_y = SCREEN_CENTER_Y
    
    # Fixation analysis (exclude first one for left bias)
    pupil_values = []
    prev_location = 'center'  # To count visits
    
    for i, fix in enumerate(fixations):
        # Exclude first fixation (orientation bias)
        if i == 0:
            metrics['first_fix_excluded'] = True
            # Determine initial position for visits
            if point_in_aoi(fix.x, fix.y, AOI_LEFT):
                prev_location = 'left'
            elif point_in_aoi(fix.x, fix.y, AOI_RIGHT):
                prev_location = 'right'
            else:
                prev_location = 'center'
            continue
        
        current_location = 'center'
        if point_in_aoi(fix.x, fix.y, AOI_LEFT):
            current_location = 'left'
            metrics['n_fix_left'] += 1
            metrics['dur_fix_left'] += fix.duration
            # Expansion: distance to left image center
            dist = math.sqrt((fix.x - image_left_center_x)**2 + (fix.y - image_left_center_y)**2)
            metrics['expansion_left'] += dist
            
        elif point_in_aoi(fix.x, fix.y, AOI_RIGHT):
            current_location = 'right'
            metrics['n_fix_right'] += 1
            metrics['dur_fix_right'] += fix.duration
            # Expansion: distance to right image center
            dist = math.sqrt((fix.x - image_right_center_x)**2 + (fix.y - image_right_center_y)**2)
            metrics['expansion_right'] += dist
        
        # Count visits (transition to an image from elsewhere)
        if current_location == 'left' and prev_location != 'left':
            metrics['n_visites_left'] += 1
        elif current_location == 'right' and prev_location != 'right':
            metrics['n_visites_right'] += 1
        
        prev_location = current_location
            
        if fix.pupil > 0:
            pupil_values.append(fix.pupil)
    
    if pupil_values:
        metrics['mean_pupil'] = np.mean(pupil_values)
    
    # E-Prime data
    if eprime_data:
        metrics['arousal'] = str(eprime_data.get('BasArousal', eprime_data.get('Arousal', '')))
        metrics['procedure'] = str(eprime_data.get('Procedure', ''))
        
        # Extract valences from E-Prime if not already present
        for key, val in eprime_data.items():
            if isinstance(val, str):
                if 'Neg' in key and 'Num' in key:
                    if not metrics['val_gauche']:
                        metrics['stim_gauche'] = val
                        metrics['val_gauche'] = 'neg'
                    elif not metrics['val_droit']:
                        metrics['stim_droit'] = val
                        metrics['val_droit'] = 'neg'
                elif 'Pos' in key and 'Num' in key:
                    if not metrics['val_gauche']:
                        metrics['stim_gauche'] = val
                        metrics['val_gauche'] = 'pos'
                    elif not metrics['val_droit']:
                        metrics['stim_droit'] = val
                        metrics['val_droit'] = 'pos'
                elif 'Neu' in key and 'Num' in key:
                    if not metrics['val_droit']:
                        metrics['stim_droit'] = val
                        metrics['val_droit'] = 'neu'
                    elif not metrics['val_gauche']:
                        metrics['stim_gauche'] = val
                        metrics['val_gauche'] = 'neu'
    
    # Pair type
    vals = {metrics['val_gauche'], metrics['val_droit']} - {'', 'unknown'}
    if vals == {'neg', 'neu'}:
        metrics['pair_type'] = 'neg_neu'
    elif vals == {'pos', 'neu'}:
        metrics['pair_type'] = 'pos_neu'
    elif vals == {'neg', 'pos'}:
        metrics['pair_type'] = 'neg_pos'
    
    return metrics


def apply_quality_flags(metrics: Dict) -> Dict:
    """
    Apply quality criteria according to thesis methodology.
    
    For first saccade:
    - Latency: 80-500 ms
    - Amplitude: 40-300 pixels
    - Duration: < 100 ms
    
    For fixations:
    - Blinks: < 5 per trial
    """
    metrics = metrics.copy()
    
    lat = metrics.get('sac1_latency', np.nan)
    amp_px = metrics.get('sac1_amplitude_px', np.nan)
    dur = metrics.get('sac1_duration', np.nan)
    n_blinks = metrics.get('n_blinks', 0)
    
    # No saccade toward image found
    metrics['flag_no_saccade'] = pd.isna(lat)
    
    # Individual flags for first saccade (only if saccade found)
    if pd.notna(lat):
        metrics['flag_sac1_lat_low'] = lat < MIN_SAC1_LAT_MS
        metrics['flag_sac1_lat_high'] = lat > MAX_SAC1_LAT_MS
        metrics['flag_sac1_lat'] = metrics['flag_sac1_lat_low'] or metrics['flag_sac1_lat_high']
    else:
        metrics['flag_sac1_lat_low'] = False
        metrics['flag_sac1_lat_high'] = False
        metrics['flag_sac1_lat'] = False  # No flag if no saccade
    
    if pd.notna(amp_px):
        metrics['flag_sac1_amp_low'] = amp_px < MIN_SAC1_AMP_PX
        metrics['flag_sac1_amp_high'] = amp_px > MAX_SAC1_AMP_PX
        metrics['flag_sac1_amp'] = metrics['flag_sac1_amp_low'] or metrics['flag_sac1_amp_high']
    else:
        metrics['flag_sac1_amp_low'] = False
        metrics['flag_sac1_amp_high'] = False
        metrics['flag_sac1_amp'] = False
    
    if pd.notna(dur):
        metrics['flag_sac1_dur'] = dur > MAX_SAC1_DUR_MS
    else:
        metrics['flag_sac1_dur'] = False
    
    # Blinks flag (for fixation analysis)
    metrics['flag_blinks'] = n_blinks >= MAX_BLINKS_PER_TRIAL
    
    # Valid trial for first saccade analysis
    # Excluded if: no saccade OR latency out of bounds OR amplitude out of bounds OR duration too long
    metrics['valid_for_saccade'] = (
        not metrics['flag_no_saccade'] and
        not metrics['flag_sac1_lat'] and
        not metrics['flag_sac1_amp'] and
        not metrics['flag_sac1_dur']
    )
    
    # Valid trial for fixation analysis (blinks only)
    metrics['valid_for_fixation'] = not metrics['flag_blinks']
    
    # Combined
    metrics['good_trial'] = metrics['valid_for_saccade']
    
    return metrics


# =============================================================================
# SESSION PROCESSING
# =============================================================================

def find_matching_files(session_folder: Path) -> List[Tuple[Path, Path]]:
    """Find pairs of .asc and .txt files.
    
    Typical format:
    - ASC: 1131.asc (subject 11, session 31) or 10431.asc (subject 104, session 31)
    - TXT: HAconfig1_V4_withOculo-11-31.txt or LAconfig1_V4_withOculo-104-31.txt
    
    The last digit(s) of the ASC correspond to the session code in the TXT.
    """
    asc_files = list(session_folder.glob("*.asc"))
    txt_files = [f for f in session_folder.glob("*.txt") if not f.name.startswith("~$")]
    
    pairs = []
    used_txt = set()
    
    for asc in asc_files:
        asc_id = asc.stem  # e.g., "1131", "10431"
        matching_txt = None
        
        # Extract session code from ASC file
        # Session code is usually the last 1-2 characters (31, 42, 1, 2)
        # Try last 2 first, then last 1
        asc_session_codes = []
        if len(asc_id) >= 2:
            asc_session_codes.append(asc_id[-2:])  # e.g., "31" from "1131"
            asc_session_codes.append(asc_id[-1])   # e.g., "1" from "1131"
        
        # Find txt file with same session code
        for txt in txt_files:
            if txt in used_txt:
                continue
            
            txt_name = txt.stem.upper()
            
            # Look for pattern -XX-YY or -XX-Y at end (subject-session)
            match = re.search(r'-(\d+)-(\d+)(?:\D|$)', txt.stem)
            if match:
                txt_session_code = match.group(2)  # Session code (31, 42, 1, 2)
                
                # Check if session code matches
                for asc_code in asc_session_codes:
                    if txt_session_code == asc_code:
                        matching_txt = txt
                        break
                
                if matching_txt:
                    break
        
        # Fallback: check if complete ID is in filename
        if matching_txt is None:
            for txt in txt_files:
                if txt in used_txt:
                    continue
                if asc_id in txt.stem:
                    matching_txt = txt
                    break
        
        if matching_txt:
            used_txt.add(matching_txt)
        
        pairs.append((asc, matching_txt))
    
    return pairs


def process_session(group: str, subject: str, session_folder: Path) -> Optional[pd.DataFrame]:
    """Process a complete session."""
    print(f"\n{'='*60}")
    print(f"Group: {group}, Subject: {subject}")
    print(f"{'='*60}")
    
    file_pairs = find_matching_files(session_folder)
    if not file_pairs:
        print(f"  No .asc files found")
        return None
    
    all_trials_data = []
    
    for session_num, (asc_file, txt_file) in enumerate(file_pairs, 1):
        print(f"\n  Session {session_num}: {asc_file.name}")
        
        # Parse EyeLink
        parser = EyeLinkASCParser(asc_file)
        if not parser.parse():
            continue
        
        print(f"    Fixations: {len(parser.fixations)}, Saccades: {len(parser.saccades)}, Blinks: {len(parser.blinks)}")
        
        # Parse E-Prime
        eprime = None
        if txt_file and txt_file.exists():
            eprime = EPrimeParser(txt_file)
            if eprime.parse():
                print(f"    E-Prime trials: {len(eprime.trials)}")
            else:
                eprime = None
        
        # Trial structure in EyeLink files:
        # 1. 'StimulusGauche/Droit ...' - stimulus names
        # 2. Croix_Start... - fixation cross start
        # 3. Image_Start... - IMAGE START (reference point for latency)
        # 4. pictureTrial_Offset... - image end
        
        # Collect information for each trial
        current_stim_info = {'stim_gauche': '', 'stim_droit': '', 'val_gauche': '', 'val_droit': ''}
        trial_starts = []  # Image_Start times
        trial_ends = []    # pictureTrial_Offset times
        trial_info = []
        
        for msg in parser.messages:
            # Collect stimulus names
            if "'Stimulus" in msg.text or "Stimulus" in msg.text:
                stim_match = re.search(r"['\"]?Stimulus(Gauche|Droit|Left|Right)?\s*(.+?)['\"]?\s*$", 
                                       msg.text, re.IGNORECASE)
                if stim_match:
                    side = (stim_match.group(1) or "").lower()
                    stim_name = stim_match.group(2).strip().strip("'\"")
                    valence = extract_valence(stim_name)
                    
                    if 'gauche' in side or 'left' in side:
                        current_stim_info['stim_gauche'] = stim_name
                        current_stim_info['val_gauche'] = valence
                    elif 'droit' in side or 'right' in side:
                        current_stim_info['stim_droit'] = stim_name
                        current_stim_info['val_droit'] = valence
                    else:
                        if not current_stim_info['stim_gauche']:
                            current_stim_info['stim_gauche'] = stim_name
                            current_stim_info['val_gauche'] = valence
                        else:
                            current_stim_info['stim_droit'] = stim_name
                            current_stim_info['val_droit'] = valence
            
            # Detect Image_Start - this is the true trial start for latency calculation
            elif 'Image_Start' in msg.text:
                trial_starts.append(msg.time)
                trial_info.append(current_stim_info.copy())
                # Reset for next trial
                current_stim_info = {'stim_gauche': '', 'stim_droit': '', 'val_gauche': '', 'val_droit': ''}
            
            # Detect trial end
            elif 'pictureTrial_Offset' in msg.text or 'Trial_Offset' in msg.text:
                trial_ends.append(msg.time)
        
        # If no Image_Start markers, try with Stimulus messages (old system)
        if not trial_starts:
            stim_times = []
            for msg in parser.messages:
                if "'Stimulus" in msg.text or "Stimulus" in msg.text:
                    stim_times.append(msg.time)
            
            if stim_times:
                # Group nearby markers (< 100ms)
                i = 0
                while i < len(stim_times):
                    start = stim_times[i]
                    j = i + 1
                    while j < len(stim_times) and stim_times[j] - start < 100:
                        j += 1
                    trial_starts.append(start)
                    trial_info.append({})
                    i = j
        
        # Fallback: gaps in fixations
        if not trial_starts and parser.fixations:
            trial_starts.append(parser.fixations[0].start_time)
            trial_info.append({})
            for i in range(1, len(parser.fixations)):
                gap = parser.fixations[i].start_time - parser.fixations[i-1].end_time
                if gap > 800:
                    trial_starts.append(parser.fixations[i].start_time)
                    trial_info.append({})
        
        print(f"    Detected trials: {len(trial_starts)}")
        
        # Create trials
        actual_trial_num = 0
        for idx, start_time in enumerate(trial_starts):
            # Trial end (use pictureTrial_Offset if available, otherwise +5000ms)
            if idx < len(trial_ends):
                end_time = trial_ends[idx]
            elif idx + 1 < len(trial_starts):
                end_time = trial_starts[idx + 1]
            else:
                end_time = start_time + 5000
            
            # Ignore too short trials (< 3000ms since presentation = 5s)
            if end_time - start_time < 3000:
                continue
            
            actual_trial_num += 1
            
            # Collect events for this trial
            trial_fixations = [f for f in parser.fixations if start_time <= f.start_time < end_time]
            trial_saccades = [s for s in parser.saccades if start_time <= s.start_time < end_time]
            trial_blinks = [b for b in parser.blinks if start_time <= b.start_time < end_time]
            
            # Stimulus info
            stim_info = trial_info[idx] if idx < len(trial_info) else {}
            
            # E-Prime data
            eprime_data = None
            if eprime and actual_trial_num <= len(eprime.trials):
                eprime_data = eprime.trials[actual_trial_num - 1]
            
            # Calculate metrics
            metrics = calculate_trial_metrics(
                trial_num=actual_trial_num,
                start_time=start_time,
                end_time=end_time,
                fixations=trial_fixations,
                saccades=trial_saccades,
                blinks=trial_blinks,
                stim_info=stim_info,
                eprime_data=eprime_data
            )
            
            # Apply flags
            metrics = apply_quality_flags(metrics)
            
            # Metadata
            metrics['group'] = group
            metrics['subject'] = subject
            metrics['session'] = session_num
            metrics['asc_file'] = asc_file.name
            metrics['txt_file'] = txt_file.name if txt_file else ''
            
            all_trials_data.append(metrics)
    
    if not all_trials_data:
        return None
    
    df = pd.DataFrame(all_trials_data)
    
    # Detailed statistics
    n_total = len(df)
    n_no_sac = df['flag_no_saccade'].sum()
    n_lat_excl = df['flag_sac1_lat'].sum()
    n_amp_excl = df['flag_sac1_amp'].sum()
    n_dur_excl = df['flag_sac1_dur'].sum()
    n_valid_sac = df['valid_for_saccade'].sum()
    n_valid_fix = df['valid_for_fixation'].sum()
    
    pct_excl_sac = (1 - n_valid_sac / n_total) * 100
    pct_excl_fix = (1 - n_valid_fix / n_total) * 100
    
    print(f"\n  Summary:")
    print(f"    Total trials: {n_total}")
    print(f"    No saccade toward image: {n_no_sac} ({n_no_sac/n_total*100:.1f}%)")
    print(f"    Excluded latency: {n_lat_excl} ({n_lat_excl/n_total*100:.1f}%)")
    print(f"    Excluded amplitude: {n_amp_excl} ({n_amp_excl/n_total*100:.1f}%)")
    print(f"    Excluded duration: {n_dur_excl} ({n_dur_excl/n_total*100:.1f}%)")
    print(f"    -> Valid saccades: {n_valid_sac} ({100-pct_excl_sac:.1f}%) - Total excluded: {pct_excl_sac:.1f}%")
    print(f"    -> Valid fixations: {n_valid_fix} ({100-pct_excl_fix:.1f}%) - Excluded blinks: {pct_excl_fix:.1f}%")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("EYE-TRACKING DATA PREPROCESSING")
    print("Criteria: lat 80-500ms, amp 40-300px, dur <100ms, blinks <5")
    print("="*70)
    
    sessions_found = []
    
    for grp_folder in RAW_DATA_ROOT.iterdir():
        if not grp_folder.is_dir():
            continue
        
        grp_name = grp_folder.name.lower()
        grp_normalized = GROUP_MAPPING.get(grp_name, grp_name)
        
        if grp_normalized not in ['age', 'moyen', 'jeunes']:
            continue
        
        for subj_folder in grp_folder.iterdir():
            if not subj_folder.is_dir():
                continue
            if list(subj_folder.glob("*.asc")):
                sessions_found.append((grp_normalized, subj_folder.name, subj_folder))
    
    if not sessions_found:
        print("No sessions found!")
        return
    
    print(f"\n{len(sessions_found)} participants found")
    
    all_data = []
    summaries = []
    
    for group, subject, folder in sessions_found:
        df = process_session(group, subject, folder)
        
        if df is not None and len(df) > 0:
            # Save CSV with ; for Excel
            out_name = f"preproc_{group}_{subject}.csv"
            out_path = PER_SESSION_DIR / out_name
            df.to_csv(out_path, index=False, sep=';', encoding='utf-8-sig')
            
            all_data.append(df)
            
            n_valid = int(df['valid_for_saccade'].sum())
            pct_excl = (1 - n_valid / len(df)) * 100
            summaries.append({
                'group': group,
                'subject': subject,
                'n_trials': len(df),
                'n_valid_saccade': n_valid,
                'pct_excluded_saccade': round(pct_excl, 1),
                'n_valid_fixation': int(df['valid_for_fixation'].sum()),
                'pct_excluded_blinks': round((1 - df['valid_for_fixation'].mean()) * 100, 1)
            })
    
    if not all_data:
        print("\nNo preprocessed data.")
        return
    
    # Global file
    global_df = pd.concat(all_data, ignore_index=True)
    global_df.to_csv(OUT_DIR / "global_preproc.csv", index=False, sep=';', encoding='utf-8-sig')
    
    # Summary
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "sessions_summary.csv", index=False, sep=';', encoding='utf-8-sig')
    
    print(f"\n{'='*70}")
    print("SUMMARY BY GROUP")
    print(f"{'='*70}")
    
    for grp in ['jeunes', 'moyen', 'age']:
        grp_data = summary_df[summary_df['group'] == grp]
        if len(grp_data) == 0:
            continue
        
        print(f"\n{grp.upper()}:")
        print(f"  Participants: {len(grp_data)}")
        print(f"  Total trials: {grp_data['n_trials'].sum()}")
        print(f"  % excluded (saccades): {grp_data['pct_excluded_saccade'].mean():.1f}%")
        print(f"  % excluded (blinks): {grp_data['pct_excluded_blinks'].mean():.1f}%")
    
    print(f"\n{'='*70}")
    print(f"Global file: {OUT_DIR / 'global_preproc.csv'}")
    print(f"Total trials: {len(global_df)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
