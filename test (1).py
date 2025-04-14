import threading
import flask
from flask import Flask, request, jsonify, render_template
import cv2
# *** ADDED BACK ***
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import math
import base64
import numpy as np
import io
from PIL import Image
from typing import Dict, Optional, Any, Tuple, List
import torch
import torchvision.transforms.functional as F # Keep for DeepLab preprocess
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation # Keep SegFormer
import traceback
import os
import time
from flask_cors import CORS
import cv2
import numpy as np
from scipy import ndimage
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# *** UPDATED *** Reflecting hybrid approach
DEBUG_DIR = "debug_images_v8_hybrid_y"
RAW_IMAGE_DIR = "raw_measure_images"
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR); print(f"Created debug directory: {DEBUG_DIR}")
# *** NEW *** Create raw image directory
if not os.path.exists(RAW_IMAGE_DIR):
    os.makedirs(RAW_IMAGE_DIR); print(f"Created raw image directory: {RAW_IMAGE_DIR}")
# --- Model Setup (DeepLabV3, SegFormer, MediaPipe) ---

# --- DeepLabV3 Model Setup (for Masking) ---
DEEPLAB_MODEL_LOADED = False
# ... (Keep DeepLabV3 loading code as before) ...
deeplab_model = None
deeplab_preprocess = None
DEEPLAB_PERSON_CLASS_INDEX = -1
print("Loading DeepLabV3 model (for masking)...")
start_time = time.time()
try:
    deeplab_weights = DeepLabV3_ResNet101_Weights.DEFAULT
    deeplab_model = deeplabv3_resnet101(weights=deeplab_weights)
    deeplab_model.eval()
    deeplab_model.to(DEVICE)
    deeplab_preprocess = deeplab_weights.transforms()
    deeplab_class_names = deeplab_weights.meta["categories"]
    try:
        DEEPLAB_PERSON_CLASS_INDEX = deeplab_class_names.index('person')
        print(f"DeepLab 'person' class index: {DEEPLAB_PERSON_CLASS_INDEX}")
        DEEPLAB_MODEL_LOADED = True
    except ValueError: print("ERROR: 'person' class not found in DeepLab categories.")
except Exception as e: print(f"FATAL ERROR: Could not load DeepLabV3 model: {e}")
print(f"DeepLabV3 loaded: {DEEPLAB_MODEL_LOADED} ({time.time() - start_time:.2f}s)")

# --- SegFormer Model Setup (for Bounding Box) ---
SEGFORMER_MODEL_LOADED = False
# ... (Keep SegFormer loading code as before) ...
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
segformer_processor = None
segformer_model = None
SEGFORMER_PERSON_CLASS_ID = -1
print(f"Loading SegFormer model ({SEGFORMER_MODEL_NAME}) for bounding box...")
start_time = time.time()
try:
    segformer_processor = AutoImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME)
    segformer_model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_NAME).to(DEVICE)
    segformer_model.eval()
    id2label = segformer_model.config.id2label
    found_person = False
    for id_val_str, label in id2label.items():
        try:
            id_val = int(id_val_str)
            if label.lower() == 'person': SEGFORMER_PERSON_CLASS_ID = id_val; found_person = True; break
        except ValueError: continue
    if not found_person: print(f"Warning: SegFormer 'person' ID not verified. Assuming {SEGFORMER_PERSON_CLASS_ID}.")
    if SEGFORMER_PERSON_CLASS_ID == -1: SEGFORMER_PERSON_CLASS_ID = 12
    SEGFORMER_MODEL_LOADED = True
except Exception as e: print(f"FATAL ERROR: Could not load SegFormer model: {e}")
print(f"SegFormer loaded: {SEGFORMER_MODEL_LOADED} ({time.time() - start_time:.2f}s)")

# --- MediaPipe Pose Setup (for Landmark Relative Y) ---
# *** ADDED BACK ***
MP_POSE_LOADED = False
# ... (Keep MediaPipe Pose loading code as before) ...
mp_pose = mp.solutions.pose
pose_processor = None
print("Loading MediaPipe Pose model...")
start_time = time.time()
try:
    pose_processor = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    MP_POSE_LOADED = True
except Exception as e: print(f"FATAL ERROR: Could not load MediaPipe Pose model: {e}")
print(f"MediaPipe Pose loaded: {MP_POSE_LOADED} ({time.time() - start_time:.2f}s)")


# --- Body Part Localization Config ---
# <<<--- REMOVED BODY_PART_RELATIVE_Y --- >>>
# *** ADDED *** Offset ratios for Chest/Waist relative to Shoulder/Hip landmarks
# These might need tuning based on visual results
CHEST_Y_OFFSET_RATIO = 0.15 # % of shoulder-hip distance below shoulder landmark
WAIST_Y_OFFSET_RATIO = 0.15 # % of shoulder-hip distance above hip landmark

# --- Calibration Data (Ground Truth) ---
# ... (Keep calibration_guide as before) ...
calibration_guide = {
    'shoulder_width': 40.64, 'chest_circ': 96.52, 'hip_width': 38.10,
    'waist_circ': 92.0, 'thigh_circ': 56.0,
}

# --- Global State (Multi-Factor) ---
# ... (Keep calibration_factors and is_calibrated as before) ...
calibration_factors: Dict[str, Optional[float]] = {
    'shoulder_width': None, 'hip_width': None, 'chest_circ': None,
    'waist_circ': None, 'thigh_circ': None,
}
is_calibrated: bool = False

COUNTER_FILE = os.path.join(os.path.dirname(__file__), "request_counter.txt") # Store alongside script
request_counter = 0
counter_lock = threading.Lock() # To prevent race conditions if using threads/multiple workers



# --- Constants ---
SIZE_MAPPING = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}
SIZE_REVERSE_MAPPING = {v: k for k, v in SIZE_MAPPING.items()}
SIZES = ["XS", "S", "M", "L", "XL", "XXL"]

# --- Thresholds (Upper bounds, exclusive, in CM) ---
# Derived from analyzing various regional size charts (Men's/Unisex leaning)
# These are REPRESENTATIVE AVERAGES and subject to brand/regional variation.
# Format: [XS_max, S_max, M_max, L_max, XL_max] - anything >= XL_max is XXL

THRESHOLDS = {
    'asian': {
        # Generally smaller fit for the same label
        'chest':    [86, 92, 98, 104, 110], # Approx Chest Circumference
        'waist':    [72, 78, 84, 90, 96],   # Approx Waist Circumference
        'hip_circ': [88, 94, 100, 106, 112], # Approx Hip Circumference
        'shoulder': [40, 42, 44, 46, 48]    # Approx Shoulder Width
    },
    'european': {
        # Moderate fit
        'chest':    [90, 96, 102, 108, 114],
        'waist':    [76, 82, 88, 94, 100],
        'hip_circ': [92, 98, 104, 110, 116],
        'shoulder': [41, 43, 45, 47, 49]
    },
    'western': {
        # Generally larger/looser fit for the same label (often US-based)
        'chest':    [94, 100, 106, 112, 118],
        'waist':    [80, 86, 92, 98, 104],
        'hip_circ': [96, 102, 108, 114, 120],
        'shoulder': [42, 44, 46, 48, 50]
    }
}

# --- Helper Functions ---

def _get_size_from_measurement(
    measurement_cm: Optional[float],
    thresholds: List[float]
) -> Optional[str]:
    """Determines the size label based on a single measurement and its thresholds."""
    if measurement_cm is None or measurement_cm <= 0:
        return None

    for i, threshold in enumerate(thresholds):
        if measurement_cm < threshold:
            return SIZES[i] # XS, S, M, L, XL

    # If measurement is greater than or equal to the last threshold
    return SIZES[-1] # XXL


def _calculate_regional_size_comprehensive(
    measurements: Dict[str, Optional[float]],
    regional_thresholds: Dict[str, List[float]]
) -> Optional[str]:
    """
    Calculates the recommended size for a specific region based on comprehensive measurements.

    Considers Chest Circ, Waist Circ, Hip Circ, and Shoulder Width.
    Returns the largest size required by any of these measurements.
    Ignores Hip Width and Thigh Circumference for primary size determination.
    """
    chest_size_str = _get_size_from_measurement(
        measurements.get('Chest Circumference'), regional_thresholds['chest']
    )
    waist_size_str = _get_size_from_measurement(
        measurements.get('Waist Circumference'), regional_thresholds['waist']
    )
    hip_circ_size_str = _get_size_from_measurement(
        measurements.get('Hip Circumference'), regional_thresholds['hip_circ']
    )
    shoulder_size_str = _get_size_from_measurement(
        measurements.get('Shoulder Width'), regional_thresholds['shoulder']
    )

    determined_sizes = []
    if chest_size_str:
        determined_sizes.append(chest_size_str)
    if waist_size_str:
        determined_sizes.append(waist_size_str)
    if hip_circ_size_str:
        determined_sizes.append(hip_circ_size_str)
    if shoulder_size_str:
        determined_sizes.append(shoulder_size_str)

    if not determined_sizes:
        return "Insufficient Data" # Or None, if preferred

    # Convert sizes to numeric values to find the maximum
    numeric_sizes = [SIZE_MAPPING[size] for size in determined_sizes]
    max_size_value = max(numeric_sizes)

    # Convert the maximum numeric value back to the size label
    return SIZE_REVERSE_MAPPING[max_size_value]


# --- Main Function ---

def get_regional_clothing_sizes_enhanced(
    results_cm: Dict[str, Optional[float]]
) -> Dict[str, Optional[str]]:
    """
    Calculates recommended clothing sizes (XS-XXL) for Asian, European, and
    Western regions based on multiple body measurements.

    Uses Chest Circumference, Waist Circumference, Hip Circumference, and
    Shoulder Width (in cm) as primary determinants. It recommends the largest
    size required by any of these measurements for each region.

    Assumes Men's or Unisex sizing patterns due to the use of shoulder width
    and typical chart structures. Regional thresholds are representative
    averages and actual brand sizing will vary.

    Measurements like Hip Width and Thigh Circumference are not used for this
    primary size determination but could be relevant for assessing fit style
    or specific garment types (e.g., pants).

    Args:
        results_cm: A dictionary containing body measurements in centimeters.
                     Expected keys: 'Shoulder Width', 'Chest Circumference',
                     'Waist Circumference', 'Hip Circumference'. Other keys
                     like 'Hip Width', 'Thigh Circumference' are currently ignored.
                     Values should be positive numbers (int or float) or None.

    Returns:
        A dictionary containing the recommended size (e.g., "S", "M", "L", "XL")
        or "Insufficient Data" for each region: 'asian', 'european', 'western'.

    References for Threshold Derivation Logic:
    - General Regional Differences: Comparison across international retailers
      (e.g., UNIQLO, ASOS, Zalando, Gap)
    - Asian Fit Baseline: UNIQLO Men's Size Charts (e.g., https://www.uniqlo.com/ca/en/size/409212.html - specific product charts vary)
    - European/Western Comparison: ASOS Men's Size Guide (https://www.asos.com/discover/size-charts/men/)
    - EU Sizing Concepts: EN 13402 standard general principles (https://en.wikipedia.org/wiki/EN_13402)
    - US/Western Baseline: Charts from major US retailers (e.g., Gap Men's: https://www.gap.com/browse/info.do?cid=10051)
    - Aggregator Examples (use cautiously): https://www.blitzresults.com/en/us-sizes/
    *Note: Links are examples and may change. Thresholds represent a synthesis.*
    """
    required_keys = [
        'Shoulder Width', 'Chest Circumference', 'Waist Circumference', 'Hip Circumference'
    ]
    validated_measurements: Dict[str, Optional[float]] = {}

    # Validate and clean inputs
    for key in results_cm:
        value = results_cm[key]
        if value is not None:
            if not isinstance(value, (int, float)):
                print(f"Warning: Invalid type for {key}. Expected number or None, got {type(value)}. Treating as None.")
                validated_measurements[key] = None
            elif value < 0:
                print(f"Warning: Negative value for {key}. Measurement cannot be negative. Treating as None.")
                validated_measurements[key] = None
            else:
                 validated_measurements[key] = float(value) # Ensure float for comparisons
        else:
            validated_measurements[key] = None

    # Ensure all potentially used keys exist, even if None
    for key in required_keys + ['Thigh Circumference']:
         if key not in validated_measurements:
              validated_measurements[key] = None


    recommended_sizes = {
        'asian': _calculate_regional_size_comprehensive(
            validated_measurements, THRESHOLDS['asian']
        ),
        'european': _calculate_regional_size_comprehensive(
            validated_measurements, THRESHOLDS['european']
        ),
        'western': _calculate_regional_size_comprehensive(
            validated_measurements, THRESHOLDS['western']
        )
    }

    return recommended_sizes




def load_or_initialize_counter():
    """Loads counter from file or initializes to 0 if file doesn't exist/invalid."""
    global request_counter
    try:
        if os.path.exists(COUNTER_FILE):
            with open(COUNTER_FILE, 'r') as f:
                content = f.read().strip()
                request_counter = int(content)
                print(f"Loaded request counter: {request_counter}")
        else:
            request_counter = 0 # Start from 0 if file doesn't exist
            print("Counter file not found, initializing counter to 0.")
            # Optionally write the initial value
            save_counter(request_counter)
    except (ValueError, IOError) as e:
        print(f"Error loading counter file '{COUNTER_FILE}': {e}. Resetting counter to 0.")
        request_counter = 0
    except Exception as e:
        print(f"Unexpected error loading counter: {e}. Resetting counter to 0.")
        request_counter = 0

def save_counter(count):
    """Saves the current counter value to the file."""
    try:
        # Write operation should be atomic enough for this simple case,
        # but lock ensures safety if multiple threads/workers exist.
        with counter_lock:
            with open(COUNTER_FILE, 'w') as f:
                f.write(str(count))
    except IOError as e:
        print(f"Error saving counter file '{COUNTER_FILE}': {e}")
    except Exception as e:
        print(f"Unexpected error saving counter: {e}")

# --- Helper Functions ---

# base64_to_image, generate_human_mask_deeplab, calculate_mask_width_at_y
# remain unchanged. Include them here.
# ... (Paste functions here) ...
def base64_to_image(image_base64: str) -> Optional[np.ndarray]:
    """Convert base64 string to OpenCV image (BGR)."""
    try:
        image_bytes = base64.b64decode(image_base64)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e: print(f"Error converting base64 to image: {e}"); return None

def generate_human_mask_deeplab(image_np: np.ndarray, debug_filename_prefix: Optional[str] = None) -> Optional[np.ndarray]:
    """Generates DeepLab mask and optional overlay."""
    if not DEEPLAB_MODEL_LOADED: return None # Simplified check
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        original_width, original_height = image_pil.size
        input_tensor = deeplab_preprocess(image_pil)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad(): output = deeplab_model(input_batch)['out'][0]
        output_predictions = output.argmax(0).cpu().numpy()
        binary_mask_np = np.where(output_predictions == DEEPLAB_PERSON_CLASS_INDEX, 255, 0).astype(np.uint8)
        binary_mask_resized = cv2.resize(binary_mask_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        if debug_filename_prefix:
            try:
                colored_mask_viz = np.zeros_like(image_np); colored_mask_viz[binary_mask_resized == 255] = (0, 255, 0)
                overlay_img = cv2.addWeighted(image_np, 1, colored_mask_viz, 0.4, 0)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_{debug_filename_prefix}_deeplab_overlay.jpg"), overlay_img)
            except Exception as e_overlay: print(f"Warning: Failed DeepLab overlay save: {e_overlay}")
        return binary_mask_resized
    except Exception as e: print(f"Error during DeepLab mask generation: {e}"); return None

def calculate_mask_width_at_y(mask: np.ndarray, y: int, band_height: int = 5) -> Optional[float]:
    """Calculates average mask width in a horizontal band."""
    if mask is None: return None
    if y < 0 or y >= mask.shape[0]: return None
    if band_height <= 0: band_height = 1
    height = mask.shape[0]; half_band = band_height // 2
    start_row = max(0, y - half_band); end_row = min(height, y + half_band + (band_height % 2))
    row_widths = []
    for current_y in range(start_row, end_row):
        row = mask[current_y, :]; white_pixels = np.where(row == 255)[0]
        if len(white_pixels) > 0:
            width = float(np.max(white_pixels) - np.min(white_pixels) + 1)
            if width > 0: row_widths.append(width)
    if not row_widths: return 0.0
    return np.mean(row_widths)

# --- UPDATED Y-Coordinate Function (Hybrid Approach) ---
def get_body_part_y_coordinates_hybrid(image_np: np.ndarray, debug_filename_prefix: Optional[str] = None) -> Optional[Dict[str, int]]:
    """
    Hybrid approach: Uses SegFormer for bounding box, MediaPipe for relative landmarks.
    Calculates absolute Y coordinates for measurement based on landmark positions within the box.
    Requires RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE from MediaPipe.
    """
    if not SEGFORMER_MODEL_LOADED or not MP_POSE_LOADED:
        print("Error: SegFormer or MediaPipe model not loaded for hybrid localization.")
        return None

    image_height, image_width, _ = image_np.shape
    estimated_y_coords = {}

    # --- 1. Get Bounding Box from SegFormer ---
    try:
        image_pil_seg = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        inputs = segformer_processor(images=image_pil_seg, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): outputs = segformer_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(image_height, image_width), mode="bilinear", align_corners=False)
        seg_map = logits.argmax(dim=1)[0].cpu().numpy()
        person_mask = (seg_map == SEGFORMER_PERSON_CLASS_ID).astype(np.uint8)
        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: print("SegFormer found no contours."); return None
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) < 500: print("SegFormer contour too small."); return None
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(main_contour)
        if bbox_h <= 0: print("SegFormer bounding box height is zero."); return None
        print(f"Hybrid: SegFormer BBox: y={bbox_y}, h={bbox_h}")
    except Exception as e_seg:
        print(f"Error during SegFormer bounding box detection: {e_seg}"); return None

    # --- 2. Get Landmarks from MediaPipe ---
    try:
        image_rgb_mp = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_rgb_mp.flags.writeable = False
        results = pose_processor.process(image_rgb_mp)
        image_rgb_mp.flags.writeable = True

        if not results.pose_landmarks: print("MediaPipe detected no landmarks."); return None
        landmarks = results.pose_landmarks.landmark
        keypoints_abs_y = {} # Store absolute Y pixel coordinates
        min_visibility = 0.3

        required_lm_enums = {
            'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'hip': mp_pose.PoseLandmark.RIGHT_HIP,
            'knee': mp_pose.PoseLandmark.RIGHT_KNEE
        }
        for name, lm_enum in required_lm_enums.items():
            lm = landmarks[lm_enum.value]
            if lm.visibility >= min_visibility:
                keypoints_abs_y[name] = int(lm.y * image_height)
            else:
                 print(f"Warning: MediaPipe landmark '{lm_enum.name}' low visibility ({lm.visibility:.2f}).")
                 # Don't store if visibility is too low, check later

        # Check if essential landmarks were found
        if not all(k in keypoints_abs_y for k in ['shoulder', 'hip', 'knee']):
             print(f"Error: Missing essential high-visibility MediaPipe landmarks. Found: {list(keypoints_abs_y.keys())}")
             return None

        y_shoulder_abs = keypoints_abs_y['shoulder']
        y_hip_abs = keypoints_abs_y['hip']
        y_knee_abs = keypoints_abs_y['knee']
        print(f"Hybrid: MediaPipe Abs Y: Shoulder={y_shoulder_abs}, Hip={y_hip_abs}, Knee={y_knee_abs}")

    except Exception as e_mp:
        print(f"Error during MediaPipe landmark detection: {e_mp}"); return None

    # --- 3. Calculate Relative Y positions within BBox ---
    rel_y = {}
    try:
        # Calculate relative position only if landmark is within bbox vertically
        if bbox_y <= y_shoulder_abs < bbox_y + bbox_h:
            rel_y['shoulder'] = (y_shoulder_abs - bbox_y) / bbox_h
        else: print("Warning: Shoulder landmark outside SegFormer bbox Y range."); rel_y['shoulder'] = 0.1 # Fallback

        if bbox_y <= y_hip_abs < bbox_y + bbox_h:
             rel_y['hip'] = (y_hip_abs - bbox_y) / bbox_h
        else: print("Warning: Hip landmark outside SegFormer bbox Y range."); rel_y['hip'] = 0.6 # Fallback

        # Knee relative pos isn't directly used for a measurement line, but needed for thigh calc
        rel_y_knee = -1 # Default invalid
        if bbox_y <= y_knee_abs < bbox_y + bbox_h:
            rel_y_knee = (y_knee_abs - bbox_y) / bbox_h
        else: print("Warning: Knee landmark outside SegFormer bbox Y range.")

        # Check relative order
        if rel_y.get('shoulder', 1.0) >= rel_y.get('hip', 0.0):
             print("Warning: Relative Shoulder Y not above Relative Hip Y. Using fallbacks.")
             # Provide safe fallbacks if order is wrong
             rel_y['shoulder'] = 0.15
             rel_y['hip'] = 0.55

        relative_shoulder_hip_dist = rel_y['hip'] - rel_y['shoulder']
        if relative_shoulder_hip_dist <= 0: # Avoid division by zero or negative offset
             print("Warning: Relative shoulder-hip distance invalid. Using fixed offsets.")
             relative_shoulder_hip_dist = 0.4 # Example typical relative distance

        # Calculate target relative Ys based on landmark relatives and offsets
        rel_y['chest'] = rel_y['shoulder'] + CHEST_Y_OFFSET_RATIO * relative_shoulder_hip_dist
        rel_y['waist'] = rel_y['hip'] - WAIST_Y_OFFSET_RATIO * relative_shoulder_hip_dist

        # Thigh: Midpoint between relative hip and relative knee
        # Use absolute knee Y if relative was invalid, but only if knee is below hip
        if rel_y_knee >= 0 and rel_y_knee > rel_y['hip']: # Knee found and below hip
             rel_y['thigh'] = rel_y['hip'] + (rel_y_knee - rel_y['hip']) / 2
        elif y_knee_abs > y_hip_abs: # Absolute knee below absolute hip, use absolute midpoint's relative pos
             abs_y_thigh_mid = (y_hip_abs + y_knee_abs) / 2
             rel_y['thigh'] = (abs_y_thigh_mid - bbox_y) / bbox_h
        else: # Fallback if knee data is unusable
             print("Warning: Knee position invalid for thigh calculation. Using offset from hip.")
             rel_y['thigh'] = rel_y['hip'] + 0.15 # Offset below hip landmark relative pos

        # Ensure calculated relative Ys are within [0, 1] range
        for k in ['chest', 'waist', 'thigh']:
            rel_y[k] = max(0.0, min(rel_y[k], 1.0))

        print(f"Hybrid: Calculated Relative Ys: {rel_y}")

    except ZeroDivisionError:
        print("Error: Division by zero calculating relative Y (bbox_h likely zero).")
        return None
    except KeyError as e_key:
         print(f"Error: Missing relative key during calculation: {e_key}")
         return None

    # --- 4. Calculate Final Absolute Y Coordinates ---
    final_y_coords = {}
    # Use the calculated relative Ys
    final_y_coords['y_shoulder'] = int(bbox_y + bbox_h * rel_y['shoulder'])
    final_y_coords['y_chest'] = int(bbox_y + bbox_h * rel_y['chest'])
    final_y_coords['y_waist'] = int(bbox_y + bbox_h * rel_y['waist'])
    final_y_coords['y_hip'] = int(bbox_y + bbox_h * rel_y['hip'])
    final_y_coords['y_thigh'] = int(bbox_y + bbox_h * rel_y['thigh'])

    # Clamp final absolute coords to image bounds
    for key, y_val in final_y_coords.items():
        final_y_coords[key] = max(0, min(y_val, image_height - 1))

    # --- 5. Debug Drawing ---
    if debug_filename_prefix:
        try:
            debug_img_hybrid = image_np.copy()
            # Draw SegFormer BBox
            cv2.rectangle(debug_img_hybrid, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 255), 1) # Thin Yellow BBox
            # Draw MediaPipe Landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    debug_img_hybrid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,100,0), thickness=1, circle_radius=1), # Blueish landmarks
                    mp_drawing.DrawingSpec(color=(200,200,200), thickness=1)) # Light connections
            # Draw Final Measurement Lines (based on hybrid calculation)
            line_colors = {'y_shoulder': (0, 255, 0),'y_chest':(0, 255, 0),'y_waist':(0, 255, 255),'y_hip':(0, 0, 255),'y_thigh':(255, 0, 255)}
            for key, y_val in final_y_coords.items():
                label = key.replace('y_', '').capitalize()
                color = line_colors.get(key, (255, 255, 255))
                # Draw line across the bounding box width for clarity
                cv2.line(debug_img_hybrid, (bbox_x, y_val), (bbox_x + bbox_w, y_val), color, 1)
                cv2.putText(debug_img_hybrid, f"{label} (y={y_val})", (bbox_x + bbox_w + 5, y_val + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            debug_path = os.path.join(DEBUG_DIR, f"debug_{debug_filename_prefix}_hybrid_localization.jpg")
            cv2.imwrite(debug_path, debug_img_hybrid)
        except Exception as e_dbg:
            print(f"Warning: Failed to save hybrid debug image: {e_dbg}")

    return final_y_coords


# --- UPDATED Measurement Conversion ---
# calculate_measurements_cm_multi_factor remains the same as V5
# ... (Paste calculate_measurements_cm_multi_factor here) ...
def calculate_measurements_cm_multi_factor(
    pixel_measurements: Dict[str, Optional[float]], # Use Optional since measurements might fail
    factors: Dict[str, float]
    ) -> Dict[str, Optional[float]]:
    """Converts pixel measurements to cm using specific factors and estimates circumferences."""
    required_factors = ['shoulder_width', 'hip_width', 'chest_circ', 'waist_circ', 'thigh_circ']
    if not all(factors.get(f) is not None and factors[f] > 0 for f in required_factors):
        print(f"Error: Missing/invalid factors: {factors}. Req: {required_factors}")
        return {key: None for key in ['Shoulder Width', 'Chest Circumference', 'Hip Width', 'Hip Circumference', 'Waist Circumference', 'Thigh Circumference']}

    results_cm = {'Shoulder Width': None, 'Chest Circumference': None,'Hip Circumference': None, 'Waist Circumference': None, 'Thigh Circumference': None,}

    try:
        sw_px = pixel_measurements.get('shoulder_width_px')
        if sw_px is not None and sw_px > 0: results_cm['Shoulder Width'] = round(sw_px / factors['shoulder_width'], 2)

        hw_px = pixel_measurements.get('hip_width_px')
        if hw_px is not None and hw_px > 0: results_cm['Hip Width'] = round(hw_px / factors['hip_width'], 2)

        ct_px = pixel_measurements.get('chest_thickness_px')
        if sw_px is not None and ct_px is not None and sw_px > 0 and ct_px > 0:
             chest_circ_px = math.pi * (sw_px + ct_px) / 2
             results_cm['Chest Circumference'] = round(chest_circ_px / factors['chest_circ'], 2)
        else: print("Warn: Skip Chest Circ.")

        ww_px = pixel_measurements.get('waist_width_px')
        wt_px = pixel_measurements.get('waist_thickness_px')
        if ww_px is not None and wt_px is not None and ww_px > 0 and wt_px > 0:
             waist_circ_px = math.pi * (ww_px + wt_px) / 2
             results_cm['Waist Circumference'] = round(waist_circ_px / factors['waist_circ'], 2)
        else: print("Warn: Skip Waist Circ.")

        thw_px = pixel_measurements.get('thigh_width_px')
        tht_px = pixel_measurements.get('thigh_thickness_px')
        if thw_px is not None and tht_px is not None and thw_px > 0 and tht_px > 0:
             thigh_circ_px = math.pi * (thw_px + tht_px) / 2
             results_cm['Thigh Circumference'] = round(thigh_circ_px / factors['thigh_circ'], 2)
        else: print("Warn: Skip Thigh Circ.")

        ht_px = pixel_measurements.get('hip_thickness_px')
        hip_width_cm = results_cm.get('Hip Width')
        hip_thickness_cm = None
        if ht_px is not None and ht_px > 0: hip_thickness_cm = ht_px / factors['hip_width'] # APPROX

        if hip_width_cm is not None and hip_thickness_cm is not None:
            hip_circ_cm = math.pi * (hip_width_cm + hip_thickness_cm) / 2
            results_cm['Hip Circumference'] = round(hip_circ_cm, 2)
        else: print("Warn: Skip Hip Circ.")

    except KeyError as e: print(f"Error: Missing key in conversion: {e}")
    except ZeroDivisionError: print("Error: Factor is zero.")
    except Exception as e: print(f"Error in conversion: {e}")
    return results_cm







# --- Flask App ---
app = Flask(__name__)
CORS(app)

# --- UPDATED Calibration Route ---
@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Calibrates using FRONT/SIDE images. Hybrid Y coords. Per-part factors.
    """
    global calibration_factors, is_calibrated
    data = request.get_json()
    if not data or 'image_calibrate_front' not in data or 'image_calibrate_side' not in data: return jsonify({"error": "Missing calibration image data"}), 400
    # *** UPDATED *** Check all 3 models needed
    if not DEEPLAB_MODEL_LOADED or not SEGFORMER_MODEL_LOADED or not MP_POSE_LOADED:
         return jsonify({"error": "Models not loaded. Cannot calibrate."}), 500

    try:
        image_cal_front_np = base64_to_image(data['image_calibrate_front'])
        image_cal_side_np = base64_to_image(data['image_calibrate_side'])
        image_cal_front_np = ndimage.rotate(image_cal_front_np, -90)
        image_cal_side_np = ndimage.rotate(image_cal_side_np, -90)
        if image_cal_front_np is None or image_cal_side_np is None: return jsonify({"error": "Error decoding calibration images"}), 400

        height_cal_front, width_cal_front, _ = image_cal_front_np.shape
        height_cal_side, width_cal_side, _ = image_cal_side_np.shape

        # 1. Mask both images
        print("Calibration: Generating DeepLab masks...")
        cal_front_mask = generate_human_mask_deeplab(image_cal_front_np, debug_filename_prefix="calibrate_front")
        cal_side_mask = generate_human_mask_deeplab(image_cal_side_np, debug_filename_prefix="calibrate_side")
        if cal_front_mask is None or cal_side_mask is None: return jsonify({"error": "Failed DeepLab mask generation"}), 500
        # 2. Find y-coordinates using HYBRID method on FRONT view
        print("Calibration: Estimating part locations (Hybrid)...")
        # *** UPDATED *** Call hybrid function
        y_coords = get_body_part_y_coordinates_hybrid(image_cal_front_np, debug_filename_prefix="calibrate_front")
        required_y_keys = ['y_shoulder', 'y_chest', 'y_waist', 'y_hip', 'y_thigh']
        if y_coords is None or not all(k in y_coords for k in required_y_keys):
            return jsonify({"error": f"Failed to estimate required Y locations (Hybrid). Found: {y_coords}"}), 500
        print(f"Calibration: Estimated Y Coords (Hybrid): {y_coords}")
        y_shoulder, y_chest, y_waist, y_hip, y_thigh = (y_coords[k] for k in required_y_keys)

        # 3. Validate Y-coordinates against BOTH cal image bounds
        valid_y_cal_front = all(0 <= y_coords[k] < height_cal_front for k in ['y_shoulder', 'y_hip', 'y_waist', 'y_thigh'])
        valid_y_cal_side = all(0 <= y_coords[k] < height_cal_side for k in ['y_chest', 'y_waist', 'y_thigh'])
        if not valid_y_cal_front: return jsonify({"error": f"Y coords out of bounds for FRONT cal ({height_cal_front})."}), 400
        if not valid_y_cal_side: return jsonify({"error": f"Y coords out of bounds for SIDE cal ({height_cal_side})."}), 400

        # 4. Measure relevant pixel dimensions from CALIBRATION masks
        print("Calibration: Measuring pixel dimensions...")
        cal_band_height = 7
        cal_pixel_measurements = {}
        px_errors = []
        # ... (Pixel measurement logic is the same, uses the new y_coords) ...
        cal_pixel_measurements['shoulder_width_px'] = calculate_mask_width_at_y(cal_front_mask, y_shoulder, cal_band_height)
        cal_pixel_measurements['hip_width_px'] = calculate_mask_width_at_y(cal_front_mask, y_hip, cal_band_height)
        cal_pixel_measurements['waist_width_px'] = calculate_mask_width_at_y(cal_front_mask, y_waist, cal_band_height)
        cal_pixel_measurements['thigh_width_px'] = calculate_mask_width_at_y(cal_front_mask, y_thigh, cal_band_height)
        cal_pixel_measurements['chest_thickness_px'] = calculate_mask_width_at_y(cal_side_mask, y_chest, cal_band_height)
        cal_pixel_measurements['waist_thickness_px'] = calculate_mask_width_at_y(cal_side_mask, y_waist, cal_band_height)
        cal_pixel_measurements['thigh_thickness_px'] = calculate_mask_width_at_y(cal_side_mask, y_thigh, cal_band_height)

        for key, value in cal_pixel_measurements.items():
             if value is None or value <= 0: px_errors.append(f"Invalid px for {key}: {value}")
        if px_errors: return jsonify({"error": "Calibration measurement error(s): " + "; ".join(px_errors)}), 400
        print(f"Calibration: Measured Px Dimensions: {cal_pixel_measurements}")

        # Estimate Circumferences in PIXELS
        estimated_circ_px = {}
        # ... (Circumference estimation logic is the same) ...
        estimated_circ_px['chest'] = math.pi * (cal_pixel_measurements['shoulder_width_px'] + cal_pixel_measurements['chest_thickness_px']) / 2
        estimated_circ_px['waist'] = math.pi * (cal_pixel_measurements['waist_width_px'] + cal_pixel_measurements['waist_thickness_px']) / 2
        estimated_circ_px['thigh'] = math.pi * (cal_pixel_measurements['thigh_width_px'] + cal_pixel_measurements['thigh_thickness_px']) / 2
        print(f"Calibration: Estimated Circ Px: {estimated_circ_px}")


        # 5. Calculate and Store Per-Part Factors
        print("Calibration: Calculating factors...")
        factors_calculated = {}
        calculation_errors = []
        # ... (Factor calculation logic is the same) ...
        for key in calibration_guide.keys():
            known_cm = calibration_guide.get(key)
            measured_px = None
            if key == 'shoulder_width': measured_px = cal_pixel_measurements['shoulder_width_px']
            elif key == 'hip_width': measured_px = cal_pixel_measurements['hip_width_px']
            elif key == 'chest_circ': measured_px = estimated_circ_px.get('chest')
            elif key == 'waist_circ': measured_px = estimated_circ_px.get('waist')
            elif key == 'thigh_circ': measured_px = estimated_circ_px.get('thigh')
            else: continue

            if known_cm is None or known_cm <= 0: calculation_errors.append(f"Invalid ref cm for '{key}'."); continue
            if measured_px is None or measured_px <= 0: calculation_errors.append(f"Invalid px for '{key}'."); continue
            try: factors_calculated[key] = measured_px / known_cm
            except Exception as e_factor: calculation_errors.append(f"Factor error '{key}': {e_factor}")

        # ... (Factor validation is the same) ...
        if len(calculation_errors) > 0 or not all(k in factors_calculated for k in calibration_guide.keys()):
             error_message = "Calibration factor calc errors: " + "; ".join(calculation_errors)
             calibration_factors = {key: None for key in calibration_factors}; is_calibrated = False
             return jsonify({"error": error_message}), 500

        # --- Success ---
        calibration_factors = {key: None for key in calibration_factors}; calibration_factors.update(factors_calculated)
        is_calibrated = True
        print(f"Calibration successful. Factors: {calibration_factors}")

        # Return results
        return jsonify({
            "status": "calibrated", "calculated_factors": calibration_factors,
            "reference_values_cm": calibration_guide, "detected_values_px": cal_pixel_measurements,
            "estimated_circ_px": estimated_circ_px, "estimated_y_coordinates": y_coords # From Hybrid
        })

    except Exception as e:
        print(f"Error during calibration: {e}"); traceback.print_exc()
        calibration_factors = {key: None for key in calibration_factors}; is_calibrated = False
        return jsonify({"error": f"Calibration unexpected error: {e}"}), 500


# --- UPDATED Measurement Route ---
@app.route('/measure', methods=['POST'])
def measure():
    """ Measures using front/side images. Hybrid Y coords. Per-part factors."""
    global calibration_factors, is_calibrated, request_counter # Include counter
    data = request.get_json()

    # --- Request Number Handling (Before other checks) ---
    current_request_no = -1 # Default invalid value
    try:
        with counter_lock: # Acquire lock before accessing/modifying counter
            request_counter += 1
            current_request_no = request_counter
        # Save the *incremented* counter immediately
        save_counter(current_request_no)
        print(f"Processing measure request #{current_request_no}")
    except Exception as e_counter:
         # Log error but try to proceed without saving images if counter fails
         print(f"CRITICAL ERROR updating/saving request counter: {e_counter}")
         # Maybe return an error? For now, just log it.
         # return jsonify({"error": "Internal server error processing request counter"}), 500

    # --- Existing Input and Calibration Checks ---
    if not is_calibrated: return jsonify({"error": "System not calibrated."}), 400
    if not all(calibration_factors.get(f) is not None for f in calibration_factors): return jsonify({"error": "Calibration incomplete."}), 400
    if not data or 'image_front' not in data or 'image_side' not in data: return jsonify({"error": "Missing image data"}), 400
    if not DEEPLAB_MODEL_LOADED or not SEGFORMER_MODEL_LOADED or not MP_POSE_LOADED: return jsonify({"error": "Models not loaded."}), 500

    try:
        # --- Decode Images ---
        image_front_base64 = data['image_front'] # Keep base64 for saving if needed
        image_side_base64 = data['image_side']
        image_front_np = base64_to_image(image_front_base64)
        image_side_np = base64_to_image(image_side_base64)
        image_front_np = ndimage.rotate(image_front_np, -90)
        image_side_np = ndimage.rotate(image_side_np, -90)
        if image_front_np is None or image_side_np is None: return jsonify({"error": "Error decoding images"}), 400

        # *** NEW: Save Raw Input Images ***
        if current_request_no != -1: # Only save if counter was successfully updated
            try:
                front_filename = os.path.join(RAW_IMAGE_DIR, f"front_{current_request_no}.jpg")
                side_filename = os.path.join(RAW_IMAGE_DIR, f"side_{current_request_no}.jpg")
                # Save the numpy arrays (already decoded)
                cv2.imwrite(front_filename, image_front_np)
                cv2.imwrite(side_filename, image_side_np)
                print(f"Saved raw images: {os.path.basename(front_filename)}, {os.path.basename(side_filename)}")
            except Exception as e_save:
                print(f"Warning: Failed to save raw input images for request #{current_request_no}: {e_save}")
        # --- End Save Raw Images ---

        height_front, width_front, _ = image_front_np.shape; height_side, width_side, _ = image_side_np.shape

        # --- Masking, Localization, Validation (Keep existing logic) ---
        print("Measure: Generating DeepLab masks...")
        front_mask = generate_human_mask_deeplab(image_front_np, "measure_front"); side_mask = generate_human_mask_deeplab(image_side_np, "measure_side")
        if front_mask is None or side_mask is None: return jsonify({"error": "Failed DeepLab masks"}), 500

        print("Measure: Estimating part locations (Hybrid)...")
        y_coords = get_body_part_y_coordinates_hybrid(image_front_np, debug_filename_prefix="measure_front")
        required_y_keys = ['y_shoulder', 'y_chest', 'y_waist', 'y_hip', 'y_thigh']
        if y_coords is None or not all(k in y_coords for k in required_y_keys): return jsonify({"error": f"Failed Y coordinates (Hybrid). Found: {y_coords}"}), 500
        print(f"Measure: Estimated Y Coords (Hybrid): {y_coords}")
        y_shoulder, y_chest, y_waist, y_hip, y_thigh = (y_coords[k] for k in required_y_keys)

        valid_y_front = all(0 <= y_coords[k] < height_front for k in required_y_keys)
        valid_y_side = all(0 <= y_coords[k] < height_side for k in required_y_keys)
        if not valid_y_front: return jsonify({"error": f"Y coords out of bounds for FRONT image ({height_front})."}), 400
        if not valid_y_side: return jsonify({"error": f"Y coords out of bounds for SIDE image ({height_side})."}), 400

        # --- Measure Pixel Dimensions (Keep existing logic) ---
        print("Measure: Calculating pixel dimensions...")
        measure_band = 5; measurements_px = {}; px_errors = []
        # ... (Pixel measurement logic is the same) ...
        measurements_px['shoulder_width_px'] = calculate_mask_width_at_y(front_mask, y_shoulder, measure_band)
        measurements_px['hip_width_px'] = calculate_mask_width_at_y(front_mask, y_hip, measure_band)
        measurements_px['waist_width_px'] = calculate_mask_width_at_y(front_mask, y_waist, measure_band)
        measurements_px['thigh_width_px'] = calculate_mask_width_at_y(front_mask, y_thigh, measure_band)
        measurements_px['chest_thickness_px'] = calculate_mask_width_at_y(side_mask, y_chest, measure_band)
        measurements_px['hip_thickness_px'] = calculate_mask_width_at_y(side_mask, y_hip, measure_band)
        measurements_px['waist_thickness_px'] = calculate_mask_width_at_y(side_mask, y_waist, measure_band)
        measurements_px['thigh_thickness_px'] = calculate_mask_width_at_y(side_mask, y_thigh, measure_band)

        for key, value in measurements_px.items(): # Check for errors
             if value is None or value <= 0: px_errors.append(f"Invalid px for {key}: {value}")
        if px_errors: print(f"Warning: Invalid pixel measurements: {'; '.join(px_errors)}")
        print(f"Measure: Pixel Measurements: {measurements_px}")

        # --- Convert to CM (Keep existing logic) ---
        print("Measure: Converting to CM...")
        cm_measurements = calculate_measurements_cm_multi_factor(measurements_px, calibration_factors)
        
        # --- DEBUG Drawing (Keep existing logic) ---
        # ... (Code to draw lines on masks and save debug images) ...
        debug_front_mask_lines = cv2.cvtColor(front_mask, cv2.COLOR_GRAY2BGR)
        debug_side_mask_lines = cv2.cvtColor(side_mask, cv2.COLOR_GRAY2BGR)
        def draw_measurement_line(img, y, label, value_px, color, width): # Keep helper
             if value_px is not None:
                 cv2.line(img, (0, y), (width, y), color, 1)
                 cv2.putText(img, f"{label}: {value_px:.1f}px", (10, y + (-5 if label.endswith('W') else 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # Draw lines using current y_coords
        draw_measurement_line(debug_front_mask_lines, y_shoulder, "SW", measurements_px.get('shoulder_width_px'), (0, 255, 0), width_front)
        draw_measurement_line(debug_front_mask_lines, y_waist, "WW", measurements_px.get('waist_width_px'), (0, 255, 255), width_front)
        draw_measurement_line(debug_front_mask_lines, y_hip, "HW", measurements_px.get('hip_width_px'), (0, 0, 255), width_front)
        draw_measurement_line(debug_front_mask_lines, y_thigh, "ThW", measurements_px.get('thigh_width_px'), (255, 0, 255), width_front)
        draw_measurement_line(debug_front_mask_lines, y_chest, "CT", measurements_px.get('chest_thickness_px'), (0, 255, 0), width_side)
        draw_measurement_line(debug_side_mask_lines, y_chest, "CT", measurements_px.get('chest_thickness_px'), (0, 255, 0), width_side)
        draw_measurement_line(debug_side_mask_lines, y_waist, "WT", measurements_px.get('waist_thickness_px'), (0, 255, 255), width_side)
        draw_measurement_line(debug_side_mask_lines, y_hip, "HT", measurements_px.get('hip_thickness_px'), (0, 0, 255), width_side)
        draw_measurement_line(debug_side_mask_lines, y_thigh, "ThT", measurements_px.get('thigh_thickness_px'), (255, 0, 255), width_side)
        # Save debug images
        cv2.imwrite(os.path.join(DEBUG_DIR, "debug_measure_front_mask_levels_hybrid.jpg"), debug_front_mask_lines)
        cv2.imwrite(os.path.join(DEBUG_DIR, "debug_measure_side_mask_levels_hybrid.jpg"), debug_side_mask_lines)
    
        sizing = get_regional_clothing_sizes_enhanced(cm_measurements)
        print("lala AHAHHAHAAHAH: ",type(cm_measurements))
        #cm_measurements = cm_measurements.pop(2)
        key_to_remove = list(cm_measurements.keys())[2]
        cm_measurements.pop(key_to_remove)
        # --- Return results ---
        return jsonify({
            "status": "success", 
            "measurements_cm": cm_measurements,
            "recommended_sizing": sizing,
            "calibration_factors_used": calibration_factors,
            "pixel_measurements_raw": measurements_px,
            "estimated_y_coordinates": y_coords,
            "request_number": current_request_no # Optionally return the request number
        })

    except Exception as e:
        print(f"Error in /measure endpoint (Req #{current_request_no}): {e}"); traceback.print_exc()
        return jsonify({"error": f"Unexpected measurement error (Req #{current_request_no}): {e}"}), 500
# --- Root Route ---
# Remains the same
@app.route('/')
def index():
    try:
        return render_template('index.html', is_calibrated=is_calibrated, factors=calibration_factors)
    except Exception as e_template:
        print(f"Error rendering template: {e_template}")
        status_text = f"Calibrated: {is_calibrated} (Factors: {calibration_factors})" if is_calibrated else "Not Calibrated"
        return f"<html><body><h1>Measurement App V7 (Hybrid Y)</h1><p>Status: {status_text}</p><p>(Error rendering template)</p></body></html>"


# --- Main Execution ---
# Remains the same
if __name__ == '__main__':
    # *** UPDATED *** Check all 3 models
    models_ok = DEEPLAB_MODEL_LOADED and SEGFORMER_MODEL_LOADED and MP_POSE_LOADED
    if not models_ok:
        print("\nFATAL WARNING: One or more required models failed to load.")
        print(f"  DeepLab: {DEEPLAB_MODEL_LOADED}, SegFormer: {SEGFORMER_MODEL_LOADED}, MediaPipe: {MP_POSE_LOADED}")
        print("---> Application will likely FAIL. <---")
    app.run(host='0.0.0.0', port=5000, debug=False)