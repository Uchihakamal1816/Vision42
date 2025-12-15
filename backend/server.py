from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import json
import base64
import cv2
import numpy as np
import torch
import warnings
from ultralytics import YOLO
import requests
import re
import random
from io import BytesIO

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(
    title="GeoNLI API",
    version="2.8",
    description="Satellite Image Analysis API with Attribute-Based Grounding"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo_model = None

# ============================================================================
# YOLO CLASS MAPPING DICTIONARY
# ============================================================================

YOLO_CLASSES = {
    0: 'Expressway-Service-area',
    1: 'Expressway-toll-station',
    2: 'airplane',
    3: 'airport',
    4: 'baseballfield',
    5: 'basketballcourt',
    6: 'bridge',
    7: 'chimney',
    8: 'dam',
    9: 'golffield',
    10: 'groundtrackfield',
    11: 'harbor',
    12: 'overpass',
    13: 'ship',
    14: 'stadium',
    15: 'storagetank',
    16: 'tenniscourt',
    17: 'trainstation',
    18: 'vehicle',
    19: 'windmill',
    20: 'large-vehicle',
    21: 'small-vehicle',
    22: 'helicopter',
    23: 'roundabout',
    24: 'soccer-ball-field',
    25: 'swimming-pool',
    26: 'container-crane'
}

YOLO_CLASS_NAME_TO_ID = {v.lower(): k for k, v in YOLO_CLASSES.items()}

YOLO_SYNONYMS = {
    "expressway-service-area": [
        "expressway service area", "service area", "rest stop", "rest area", 
        "highway rest stop", "expressway rest", "service station area"
    ],
    "expressway-toll-station": [
        "expressway toll station", "toll station", "toll booth", "toll plaza", 
        "toll gate", "highway toll", "expressway toll"
    ],
    "airplane": [
        "airplane", "airplanes", "plane", "planes", "aircraft", "aircrafts", 
        "jet", "jets", "airliner", "airliners", "fighter", "fighters", 
        "aeroplane", "aeroplanes", "air plane", "air planes"
    ],
    "airport": [
        "airport", "airports", "airfield", "airfields", "aerodrome", 
        "aerodromes", "airstrip", "airstrips", "air port", "air field"
    ],
    "baseballfield": [
        "baseball field", "baseballfield", "baseball diamond", "baseball stadium", 
        "baseball pitch", "baseball ground", "base ball field"
    ],
    "basketballcourt": [
        "basketball court", "basketballcourt", "basketball field", "court", 
        "basketball pitch", "basket ball court", "bb court"
    ],
    "bridge": [
        "bridge", "bridges", "overpass", "viaduct", "flyover", "road bridge"
    ],
    "chimney": [
        "chimney", "chimneys", "smokestack", "smokestacks", "stack", "stacks", 
        "industrial chimney", "factory chimney"
    ],
    "dam": [
        "dam", "dams", "water dam", "reservoir dam", "hydroelectric dam"
    ],
    "golffield": [
        "golf field", "golffield", "golf course", "golf links", "golf ground", 
        "golfing area", "golf club"
    ],
    "groundtrackfield": [
        "ground track field", "groundtrackfield", "track field", "running track", 
        "athletics track", "athletic field", "track and field", "sports track"
    ],
    "harbor": [
        "harbor", "harbors", "harbour", "harbours", "port", "ports", 
        "seaport", "seaports", "marina", "marinas", "dock", "docks"
    ],
    "overpass": [
        "overpass", "overpasses", "flyover", "flyovers", "elevated road", 
        "elevated highway", "road overpass"
    ],
    "ship": [
        "ship", "ships", "boat", "boats", "vessel", "vessels", 
        "cargo ship", "tanker", "tankers", "ferry", "ferries"
    ],
    "stadium": [
        "stadium", "stadiums", "arena", "arenas", "sports stadium", 
        "sports venue", "venue", "athletic stadium"
    ],
    "storagetank": [
        "storage tank", "storagetank", "storage tanks", "storagetanks", 
        "tank", "tanks", "fuel tank", "fuel tanks", "water tank", "water tanks",
        "oil tank", "oil tanks", "tank storage", "stor", "storage"
    ],
    "tenniscourt": [
        "tennis court", "tenniscourt", "tennis field", "tennis pitch", 
        "tennis ground", "tennis courts"
    ],
    "trainstation": [
        "train station", "trainstation", "railway station", "rail station", 
        "railroad station", "train depot", "railway terminus"
    ],
    "vehicle": [
        "vehicle", "vehicles", "car", "cars", "automobile", "automobiles", 
        "auto", "autos", "van", "vans", "suv", "suvs"
    ],
    "windmill": [
        "windmill", "windmills", "wind turbine", "wind turbines", 
        "wind mill", "wind generator"
    ],
    "large-vehicle": [
        "large vehicle", "large vehicles", "truck", "trucks", "lorry", "lorries", 
        "heavy truck", "heavy vehicle", "semi truck", "trailer"
    ],
    "small-vehicle": [
        "small vehicle", "small vehicles", "small car", "small cars", 
        "compact car", "mini vehicle"
    ],
    "helicopter": [
        "helicopter", "helicopters", "chopper", "choppers", "heli", "helis", 
        "rotorcraft", "whirlybird"
    ],
    "roundabout": [
        "roundabout", "roundabouts", "traffic circle", "traffic circles", 
        "rotary", "rotaries", "circular intersection"
    ],
    "soccer-ball-field": [
        "soccer ball field", "soccer field", "football field", "soccer pitch", 
        "football pitch", "soccer ground", "football ground"
    ],
    "swimming-pool": [
        "swimming pool", "swimmingpool", "pool", "pools", "swim pool", 
        "swimming pools"
    ],
    "container-crane": [
        "container crane", "containercrane", "crane", "cranes", "port crane", 
        "shipping crane", "harbor crane", "cargo crane"
    ]
}

@app.on_event("startup")
async def startup_event():
    global yolo_model
    checkpoint_path = "/data1/MP3/best.pt"
    yolo_model = YOLO(checkpoint_path)
    
    print(f"✓ YOLO model loaded from {checkpoint_path}")
    print(f"✓ YOLO has {len(YOLO_CLASSES)} classes")

# ============================================================================
# CLASS MAPPING FUNCTIONS
# ============================================================================

def map_query_to_yolo_class(query: str) -> Optional[str]:
    """Map natural language query to YOLO class using synonym matching."""
    query_lower = query.lower()
    
    print(f"\n[CLASS MAPPING] Query: '{query}'")
    
    matches = []
    
    for yolo_class, synonyms in YOLO_SYNONYMS.items():
        for synonym in synonyms:
            synonym_lower = synonym.lower()
            
            if synonym_lower == query_lower:
                print(f"✓ [CLASS MAPPING] EXACT match '{synonym}' → '{yolo_class}'")
                return yolo_class
            
            if synonym_lower in query_lower:
                score = len(synonym_lower)
                matches.append((yolo_class, synonym, score))
    
    if matches:
        matches.sort(key=lambda x: x[2], reverse=True)
        best_class, best_synonym, best_score = matches[0]
        print(f"✓ [CLASS MAPPING] BEST match '{best_synonym}' → '{best_class}'")
        return best_class
    
    for class_name in YOLO_CLASS_NAME_TO_ID.keys():
        if class_name in query_lower or query_lower in class_name:
            print(f"✓ [CLASS MAPPING] Direct class match: '{class_name}'")
            return class_name
    
    print(f"⚠ [CLASS MAPPING] No match found")
    return None

def ask_qwen_for_class_mapping(query: str, image_b64: Optional[str] = None) -> Optional[str]:
    """Use Qwen to intelligently map query to YOLO class."""
    mapping_prompt = f"""You are a class mapper. Given a user query, identify which object class they're asking about.

User query: "{query}"

Available object classes (choose ONLY from this list):
{json.dumps(list(YOLO_CLASSES.values()), indent=2)}

MAPPING RULES:
1. "airplanes" or "aircrafts" → "airplane"
2. "storage tanks" or "tanks" or "stor" → "storagetank"  
3. Match plural to singular (e.g., "ships" → "ship")
4. Remove spaces/hyphens when matching

Respond with ONLY the exact class name from the list, or "NONE" if no match.

Answer:"""
    
    print(f"\n[QWEN MAPPING] Asking Qwen to map class...")
    
    try:
        if image_b64:
            payload = {
                "model": "/data1/MP3/qwen",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": mapping_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 30,
                "temperature": 0.0
            }
        else:
            payload = {
                "model": "/data1/MP3/qwen",
                "messages": [{"role": "user", "content": mapping_prompt}],
                "max_tokens": 30,
                "temperature": 0.0
            }
        
        response = requests.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"].strip()
            print(f"[QWEN MAPPING] Raw response: '{result}'")
            
            result_clean = result.lower().strip('.,!?\"\' ')
            
            if result_clean in YOLO_CLASS_NAME_TO_ID:
                print(f"✓ [QWEN MAPPING] Exact match: '{result_clean}'")
                return result_clean
            
            result_normalized = result_clean.replace(' ', '').replace('-', '')
            for class_name in YOLO_CLASS_NAME_TO_ID.keys():
                class_normalized = class_name.replace(' ', '').replace('-', '')
                if result_normalized == class_normalized:
                    print(f"✓ [QWEN MAPPING] Normalized match: '{class_name}'")
                    return class_name
            
            for class_name in YOLO_CLASS_NAME_TO_ID.keys():
                if class_name in result_clean or result_clean in class_name:
                    print(f"✓ [QWEN MAPPING] Partial match: '{class_name}'")
                    return class_name
        
    except Exception as e:
        print(f"✗ [QWEN MAPPING] Error: {e}")
    
    return None

def extract_target_class_from_query(query: str, image_b64: Optional[str] = None) -> Optional[str]:
    """Main function to extract target YOLO class from query."""
    yolo_class = map_query_to_yolo_class(query)
    
    if yolo_class:
        return yolo_class
    
    yolo_class = ask_qwen_for_class_mapping(query, image_b64)
    
    if yolo_class:
        return yolo_class
    
    return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def image_to_base64(img, max_size=512):
    """Convert image to base64 string."""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    _, buffer = cv2.imencode(".jpg", img, encode_param)
    return base64.b64encode(buffer).decode("utf-8")

def enforce_word_count(text: str, min_words: int = 40, max_words: int = 60) -> str:
    """
    Enforce word count to be between min_words and max_words.
    Truncates if too long, returns as-is if within range.
    """
    if not text:
        return text
    
    words = text.split()
    word_count = len(words)
    
    print(f"[WORD COUNT] Original: {word_count} words")
    
    if word_count < min_words:
        print(f"⚠ [WORD COUNT] Too short ({word_count} < {min_words}), returning as-is")
        return text
    elif word_count > max_words:
        truncated = ' '.join(words[:max_words])
        print(f"✓ [WORD COUNT] Truncated from {word_count} to {max_words} words")
        return truncated
    else:
        print(f"✓ [WORD COUNT] Within range ({min_words}-{max_words})")
        return text

def ask_qwen_caption(image_b64, question, min_words=40, max_words=60):
    """
    Query Qwen VLM specifically for captioning with word count enforcement.
    """
    caption_prompt = f"""{question}

IMPORTANT: Provide a detailed description that is between {min_words} and {max_words} words long, preferably 50 words. 
Include key features, objects, structures, terrain, and spatial relationships visible in the image."""
    
    payload = {
        "model": "/data1/MP3/qwen",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            }
        ],
        "max_tokens": 400,
        "temperature": 0.3
    }
    
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=60)
        if response.status_code == 200:
            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                raw_caption = response_json["choices"][0]["message"]["content"]
                return enforce_word_count(raw_caption, min_words, max_words)
        return None
    except Exception as e:
        print(f"Qwen caption error: {e}")
        return None

def ask_qwen(image_b64, question, max_tokens=512):
    """Query Qwen VLM for non-caption queries."""
    payload = {
        "model": "/data1/MP3/qwen",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", json=payload, timeout=60)
        if response.status_code == 200:
            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                return response_json["choices"][0]["message"]["content"]
        return None
    except Exception as e:
        print(f"Qwen error: {e}")
        return None

def load_image_from_url(url):
    """Load image from URL with proper headers and error handling."""
    try:
        # Add headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': url.rsplit('/', 1)[0] + '/',  # Use base URL as referer
        }
        
        print(f"\n[IMAGE DOWNLOAD] Attempting to download from: {url}")
        
        # Try with requests
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True, verify=True)
        
        print(f"[IMAGE DOWNLOAD] Status code: {resp.status_code}")
        print(f"[IMAGE DOWNLOAD] Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
        print(f"[IMAGE DOWNLOAD] Content-Length: {len(resp.content)} bytes")
        
        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code}: {resp.reason}")
        
        # Check if content is actually an image
        content_type = resp.headers.get('Content-Type', '').lower()
        if 'image' not in content_type and len(resp.content) < 1000:
            # Might be HTML error page
            print(f"[IMAGE DOWNLOAD] Response preview: {resp.content[:200]}")
            raise ValueError(f"Response does not appear to be an image (Content-Type: {content_type})")
        
        # Decode image
        data = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("OpenCV failed to decode image data")
        
        print(f"✓ [IMAGE DOWNLOAD] Successfully loaded image: {img.shape}")
        return img
        
    except requests.exceptions.Timeout:
        print(f"✗ [IMAGE DOWNLOAD] Request timeout after 30 seconds")
        raise ValueError("Image download timeout")
    except requests.exceptions.RequestException as e:
        print(f"✗ [IMAGE DOWNLOAD] Request failed: {e}")
        raise ValueError(f"Image download failed: {str(e)}")
    except Exception as e:
        print(f"✗ [IMAGE DOWNLOAD] Unexpected error: {e}")
        raise ValueError(f"Image processing failed: {str(e)}")

def yolo_detect_objects(img, target_class: Optional[str] = None):
    """Run YOLO OBB detection."""
    print(f"\n[YOLO DETECT] Starting detection...")
    print(f"[YOLO DETECT] Target class filter: {target_class if target_class else 'None'}")

    try:
        results = yolo_model(img, conf=0.25, verbose=False)

        if results is None or len(results) == 0:
            print("✗ [YOLO DETECT] No results returned")
            return []

        detections = []
        img_h, img_w = img.shape[:2]

        for result in results:
            if not hasattr(result, "obb") or result.obb is None:
                print("⚠ [YOLO DETECT] No OBB detections")
                continue

            obb_list = result.obb
            print(f"[YOLO DETECT] Found {len(obb_list)} OBB detections")

            for ob in obb_list:
                cls_id = int(ob.cls[0])
                cls_name = yolo_model.names[cls_id].lower().replace(" ", "").replace("-", "")
                conf = float(ob.conf[0])
                cx, cy, w, h, ang = ob.xywhr[0].tolist()

                # Filter
                if target_class:
                    tgt = target_class.lower().replace(" ", "").replace("-", "")
                    if cls_name != tgt:
                        continue

                detections.append({
                    "class": yolo_model.names[cls_id],
                    "confidence": round(conf, 3),
                    "obb_xywhr": {
                        "cx": round(cx / img_w, 4),
                        "cy": round(cy / img_h, 4),
                        "w": round(w / img_w, 4),
                        "h": round(h / img_h, 4),
                        "angle_deg": round(np.degrees(ang), 2)
                    }
                })

        print(f"[YOLO DETECT] Final detections: {len(detections)}")
        return detections

    except Exception as e:
        print(f"✗ [YOLO DETECT] Fatal error: {e}")
        return []

def draw_obb_on_image(img, detections):
    """Draw oriented bounding boxes on image WITHOUT labels."""
    img_annotated = img.copy()
    img_h, img_w = img.shape[:2]
    
    # Color palette for different classes
    np.random.seed(42)
    colors = {}
    
    for det in detections:
        class_name = det["class"]
        obb = det["obb_xywhr"]
        
        # Get color for this class
        if class_name not in colors:
            colors[class_name] = tuple(np.random.randint(0, 255, 3).tolist())
        color = colors[class_name]
        
        # Convert normalized OBB to pixel coordinates
        cx = obb["cx"] * img_w
        cy = obb["cy"] * img_h
        w = obb["w"] * img_w
        h = obb["h"] * img_h
        angle_rad = np.radians(obb["angle_deg"])
        
        # Calculate rotated rectangle corners
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        dw, dh = w / 2, h / 2
        
        corners = np.array([
            [-dw, -dh],
            [dw, -dh],
            [dw, dh],
            [-dw, dh]
        ])
        
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = (corners @ R.T) + np.array([cx, cy])
        
        # Draw polygon ONLY - NO LABELS
        pts = rotated.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_annotated, [pts], isClosed=True, color=color, thickness=3)
    
    return img_annotated

def obb_xywhr_to_8points(obb, img_w, img_h):
    """
    Convert YOLO OBB to 8-point normalized format: (x1, y1, x2, y2, x3, y3, x4, y4)
    All coordinates are normalized (divided by image dimensions).
    """
    try:
        cx = obb["cx"] * img_w
        cy = obb["cy"] * img_h
        w = obb["w"] * img_w
        h = obb["h"] * img_h
        ang = np.radians(obb["angle_deg"])

        cos_a = np.cos(ang)
        sin_a = np.sin(ang)

        dw, dh = w / 2, h / 2

        corners = np.array([
            [-dw, -dh],
            [dw, -dh],
            [dw, dh],
            [-dw, dh]
        ])

        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = (corners @ R.T) + np.array([cx, cy])

        # Normalize coordinates
        rotated_norm = rotated / np.array([img_w, img_h])

        # Flatten to 8-point format
        return rotated_norm.reshape(-1).round(4).tolist()

    except Exception as e:
        print("OBB conversion error:", e)
        return [0.0] * 8

def filter_detections_by_attributes(detections, query, image_b64, img_w, img_h):
    """
    Use Qwen to filter detections based on attribute specifications in the query.
    Handles positional (second from left), size (tallest), color, shape attributes.
    """
    if not detections:
        return []
    
    # Check if query has attribute specifications
    attribute_keywords = [
        'tallest', 'largest', 'biggest', 'smallest', 'shortest',
        'first', 'second', 'third', 'last',
        'left', 'right', 'top', 'bottom', 'center',
        'dome', 'open', 'closed', 'round', 'square',
        'color', 'red', 'blue', 'green', 'white', 'black'
    ]
    
    has_attributes = any(kw in query.lower() for kw in attribute_keywords)
    
    if not has_attributes or len(detections) == 1:
        return detections
    
    print(f"\n[ATTRIBUTE FILTER] Detected attribute-based query")
    print(f"[ATTRIBUTE FILTER] Total detections before filtering: {len(detections)}")
    
    # Build filtering prompt
    detection_descriptions = []
    for idx, det in enumerate(detections, 1):
        obb = det["obb_xywhr"]
        cx, cy = obb["cx"], obb["cy"]
        w, h = obb["w"], obb["h"]
        
        # Position description
        h_pos = "left" if cx < 0.33 else "center" if cx < 0.67 else "right"
        v_pos = "top" if cy < 0.33 else "middle" if cy < 0.67 else "bottom"
        
        detection_descriptions.append(
            f"Object {idx}: located in {h_pos}-{v_pos}, "
            f"size {w:.3f}x{h:.3f}, center at ({cx:.3f}, {cy:.3f})"
        )
    
    filter_prompt = f"""Given this query: "{query}"

You have detected {len(detections)} objects of the target class:
{chr(10).join(detection_descriptions)}

Which object(s) match the specification in the query? Consider:
- Positional cues (left, right, top, bottom, second from left, etc.)
- Size attributes (tallest, largest, smallest, etc.)
- Visual attributes mentioned in the query

Respond with ONLY the object number(s) that match, separated by commas (e.g., "2" or "1,3").
If ALL objects match or no specific filter is needed, respond with "ALL".

Answer:"""
    
    try:
        payload = {
            "model": "/data1/MP3/qwen",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": filter_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                }
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }
        
        response = requests.post("http://localhost:8000/v1/chat/completions", 
                                json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"].strip()
            print(f"[ATTRIBUTE FILTER] Qwen response: '{result}'")
            
            if "ALL" in result.upper():
                print(f"[ATTRIBUTE FILTER] Keeping all {len(detections)} detections")
                return detections
            
            # Parse object IDs
            selected_ids = []
            for part in result.split(','):
                try:
                    obj_id = int(part.strip())
                    if 1 <= obj_id <= len(detections):
                        selected_ids.append(obj_id - 1)  # Convert to 0-indexed
                except:
                    continue
            
            if selected_ids:
                filtered = [detections[i] for i in selected_ids]
                print(f"[ATTRIBUTE FILTER] Filtered to {len(filtered)} detection(s): {[i+1 for i in selected_ids]}")
                return filtered
        
    except Exception as e:
        print(f"[ATTRIBUTE FILTER] Error: {e}")
    
    # Fallback: return all detections
    print(f"[ATTRIBUTE FILTER] Fallback: returning all detections")
    return detections

def compute_area_from_obb(obb_xywhr, img_w, img_h, spatial_res):
    """Compute physical area in square meters from OBB."""
    try:
        w_norm = obb_xywhr["w"]
        h_norm = obb_xywhr["h"]

        w_px = w_norm * img_w
        h_px = h_norm * img_h

        w_m = w_px * spatial_res
        h_m = h_px * spatial_res

        area_m2 = w_m * h_m

        return round(area_m2, 2)

    except Exception as e:
        print("✗ AREA COMPUTATION ERROR:", e)
        return None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ImageMetadata(BaseModel):
    width: int
    height: int
    spatial_resolution_m: Optional[float] = None

class InputImage(BaseModel):
    image_id: str
    image_url: str
    metadata: ImageMetadata

class QueryInstruction(BaseModel):
    instruction: str

class AttributeQuery(BaseModel):
    binary: QueryInstruction
    numeric: QueryInstruction
    semantic: QueryInstruction

class Queries(BaseModel):
    caption_query: QueryInstruction
    grounding_query: QueryInstruction
    attribute_query: AttributeQuery

class GeoNLIRequest(BaseModel):
    input_image: InputImage
    queries: Queries

# ============================================================================
# EVALUATION ENDPOINT - /api/v1/evaluate
# ============================================================================

@app.post("/api/v1/evaluate")
async def evaluate_json(request: GeoNLIRequest):
    """Evaluation mode - returns structured JSON with 8-point normalized OBB format."""
    try:
        print(f"\n{'#'*70}")
        print(f"# EVALUATION START: {request.input_image.image_id}")
        print(f"{'#'*70}")
        
        img = load_image_from_url(request.input_image.image_url)
        img_h, img_w = img.shape[:2]
        print(f"✓ Image loaded: {img_w}x{img_h}")
        
        image_b64 = image_to_base64(img, max_size=512)
        
        # ================================================================
        # CAPTION
        # ================================================================
        caption_instruction = request.queries.caption_query.instruction
        caption_prompt = f"{caption_instruction}\n\nDescribe the image in detail."
        caption_response = ask_qwen_caption(
            image_b64, 
            caption_prompt,
            min_words=40,
            max_words=60
        )
        
        # ================================================================
        # GROUNDING - 8-point format with attribute-based filtering
        # ================================================================
        grounding_instruction = request.queries.grounding_query.instruction
        target_class = extract_target_class_from_query(grounding_instruction, image_b64)
        
        obb_list = []
        if target_class:
            detections = yolo_detect_objects(img, target_class=target_class)
            
            # Apply attribute-based filtering
            filtered_detections = filter_detections_by_attributes(
                detections, grounding_instruction, image_b64, img_w, img_h
            )
            
            for idx, det in enumerate(filtered_detections[:50], 1):
                obb_8pt = obb_xywhr_to_8points(det["obb_xywhr"], img_w, img_h)
                obb_list.append({
                    "object-id": str(idx),
                    "obbox": obb_8pt
                })
        
        # ================================================================
        # NUMERIC
        # ================================================================
        numeric_instruction = request.queries.attribute_query.numeric.instruction
        spatial_res = request.input_image.metadata.spatial_resolution_m
        target_class = extract_target_class_from_query(numeric_instruction, image_b64)
        
        numeric_response = None
        if target_class:
            detections = yolo_detect_objects(img, target_class=target_class)
            
            query_lower = numeric_instruction.lower()
            is_area_query = any(kw in query_lower for kw in ["area", "size", "m²", "largest", "smallest"])
            
            if is_area_query and spatial_res:
                areas = []
                for det in detections:
                    area = compute_area_from_obb(det["obb_xywhr"], img_w, img_h, spatial_res)
                    if area:
                        areas.append(area)
                
                if areas:
                    if "largest" in query_lower:
                        numeric_response = str(round(max(areas), 2))
                    elif "smallest" in query_lower:
                        numeric_response = str(round(min(areas), 2))
                    elif "total" in query_lower:
                        numeric_response = str(round(sum(areas), 2))
                    else:
                        numeric_response = str(round(max(areas), 2))
                else:
                    numeric_response = "0"
            else:
                numeric_response = str(len(detections))
        else:
            # Try Qwen for non-YOLO numeric queries
            response_text = ask_qwen(image_b64, numeric_instruction, max_tokens=50)
            if response_text:
                # Extract numeric value
                import re
                numbers = re.findall(r'\d+\.?\d*', response_text)
                if numbers:
                    numeric_response = numbers[0]
        
        # ================================================================
        # BINARY
        # ================================================================
        binary_instruction = request.queries.attribute_query.binary.instruction
        binary_response = ask_qwen(image_b64, f"{binary_instruction}\n\nAnswer ONLY 'yes' or 'no'.", max_tokens=10)
        
        if binary_response:
            binary_response = 'yes' if 'yes' in binary_response.lower() else 'no'
        else:
            binary_response = None
        
        # ================================================================
        # SEMANTIC
        # ================================================================
        semantic_instruction = request.queries.attribute_query.semantic.instruction
        semantic_response = ask_qwen(image_b64, f"{semantic_instruction}\n\nAnswer concisely.", max_tokens=50)
        
        # ================================================================
        # BUILD RESPONSE IN SPECIFIED FORMAT
        # ================================================================
        response_data = {
            "input_image": {
                "image_id": request.input_image.image_id,
                "image_url": request.input_image.image_url,
                "metadata": {
                    "width": img_w,
                    "height": img_h,
                    "spatial_resolution_m": request.input_image.metadata.spatial_resolution_m
                }
            },
            "queries": {
                "caption_query": {
                    "instruction": caption_instruction,
                    "response": caption_response
                },
                "grounding_query": {
                    "instruction": grounding_instruction,
                    "response": obb_list
                },
                "attribute_query": {
                    "binary": {
                        "instruction": binary_instruction,
                        "response": binary_response
                    },
                    "numeric": {
                        "instruction": numeric_instruction,
                        "response": numeric_response
                    },
                    "semantic": {
                        "instruction": semantic_instruction,
                        "response": semantic_response
                    }
                }
            }
        }
        
        print(f"\n{'#'*70}")
        print(f"# EVALUATION COMPLETE")
        print(f"{'#'*70}\n")
        
        return response_data
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
# ============================================================================
# INTERACTIVE ENDPOINT - /geoNLI/eval
# ============================================================================

@app.post("/geoNLI/eval")
async def interactive_eval(
    file: UploadFile = File(...),
    query: str = Form(...),
    query_type: str = Form(default="auto")
):
    """
    Interactive mode for Vercel frontend.
    Returns JSON response or annotated image based on query type.
    """
    try:
        print(f"\n{'='*60}")
        print(f"[INTERACTIVE] Query: '{query}'")
        print(f"[INTERACTIVE] Type: {query_type}")
        
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        img_h, img_w = img.shape[:2]
        image_b64 = image_to_base64(img, max_size=512)
        
        # Auto-detect query type if not specified
        if query_type == "auto":
            query_lower = query.lower()
            if any(kw in query_lower for kw in ["describe", "caption", "what is in"]):
                query_type = "caption"
            elif any(kw in query_lower for kw in ["locate", "where", "find", "draw", "bounding", "box"]):
                query_type = "grounding"
            elif any(kw in query_lower for kw in ["how many", "count", "number", "area", "size"]):
                query_type = "numeric"
            elif any(kw in query_lower for kw in ["is there", "are there", "does"]):
                query_type = "binary"
            else:
                query_type = "semantic"
        
        print(f"[INTERACTIVE] Detected type: {query_type}")
        
        # ================================================================
        # CAPTION QUERY - WITH WORD COUNT ENFORCEMENT (40-60 words)
        # ================================================================
        if query_type == "caption":
            caption_response = ask_qwen_caption(
                image_b64, 
                f"{query}\n\nDescribe the image in detail.",
                min_words=40,
                max_words=60
            )
            return JSONResponse({
                "query": query,
                "query_type": "caption",
                "response": caption_response or "Unable to generate caption"
            })
        
        # ================================================================
        # GROUNDING QUERY - Returns annotated image WITHOUT TEXT
        # ================================================================
        elif query_type == "grounding":
            target_class = extract_target_class_from_query(query, image_b64)
            
            if target_class:
                print(f"[INTERACTIVE] Using YOLO for '{target_class}'")
                detections = yolo_detect_objects(img, target_class=target_class)
                
                # Apply attribute-based filtering
                filtered_detections = filter_detections_by_attributes(
                    detections, query, image_b64, img_w, img_h
                )
                
                if filtered_detections:
                    # Draw bounding boxes on image
                    img_annotated = draw_obb_on_image(img, filtered_detections)
                    
                    # Encode to JPEG
                    _, buffer = cv2.imencode('.jpg', img_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    # Return ONLY image, no JSON
                    return StreamingResponse(
                        BytesIO(buffer.tobytes()),
                        media_type="image/jpeg",
                        headers={
                            "X-Detection-Count": str(len(filtered_detections)),
                            "X-Target-Class": target_class
                        }
                    )
                else:
                    # No detections - return original image with message overlay
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"No {target_class} detected"
                    text_size = cv2.getTextSize(text, font, 1, 2)[0]
                    text_x = (img.shape[1] - text_size[0]) // 2
                    text_y = (img.shape[0] + text_size[1]) // 2
                    
                    cv2.putText(img, text, (text_x, text_y), font, 1, (0, 0, 255), 2)
                    
                    _, buffer = cv2.imencode('.jpg', img)
                    
                    return StreamingResponse(
                        BytesIO(buffer.tobytes()),
                        media_type="image/jpeg",
                        headers={
                            "X-Detection-Count": "0",
                            "X-Target-Class": target_class
                        }
                    )
            else:
                # Fallback to Qwen
                response = ask_qwen(image_b64, f"{query}\n\nDescribe locations.", max_tokens=200)
                return JSONResponse({
                    "query": query,
                    "query_type": "grounding",
                    "response": response or "Unable to locate objects"
                })
        
        # ================================================================
        # NUMERIC QUERY (Count or Area)
        # ================================================================
        elif query_type == "numeric":
            target_class = extract_target_class_from_query(query, image_b64)
            
            if target_class:
                detections = yolo_detect_objects(img, target_class=target_class)
                
                query_lower = query.lower()
                is_area_query = any(kw in query_lower for kw in ["area", "size", "m²", "largest", "smallest"])
                
                if is_area_query:
                    # Area computation
                    spatial_res = 1.0
                    
                    areas = []
                    for det in detections:
                        area = compute_area_from_obb(det["obb_xywhr"], img_w, img_h, spatial_res)
                        if area:
                            areas.append(area)
                    
                    if areas:
                        if "largest" in query_lower or "biggest" in query_lower:
                            result = max(areas)
                        elif "smallest" in query_lower:
                            result = min(areas)
                        elif "total" in query_lower:
                            result = sum(areas)
                        else:
                            result = max(areas)
                        
                        return JSONResponse({
                            "query": query,
                            "query_type": "numeric",
                            "response": f"{result} m²",
                            "value": result,
                            "unit": "m²"
                        })
                else:
                    # Count
                    count = len(detections)
                    return JSONResponse({
                        "query": query,
                        "query_type": "numeric",
                        "response": str(count),
                        "value": count,
                        "target_class": target_class
                    })
            else:
                # Fallback to Qwen
                response = ask_qwen(image_b64, query, max_tokens=50)
                return JSONResponse({
                    "query": query,
                    "query_type": "numeric",
                    "response": response or "Unable to answer"
                })
        
        # ================================================================
        # BINARY QUERY
        # ================================================================
        elif query_type == "binary":
            response = ask_qwen(image_b64, f"{query}\n\nAnswer ONLY 'yes' or 'no'.", max_tokens=10)
            
            if response:
                response = 'yes' if 'yes' in response.lower() else 'no'
            
            return JSONResponse({
                "query": query,
                "query_type": "binary",
                "response": response or "Unable to answer"
            })
        
        # ================================================================
        # SEMANTIC QUERY
        # ================================================================
        else:  # semantic
            response = ask_qwen(image_b64, f"{query}\n\nAnswer concisely.", max_tokens=50)
            
            return JSONResponse({
                "query": query,
                "query_type": "semantic",
                "response": response or "Unable to answer"
            })
        
    except Exception as e:
        print(f"[INTERACTIVE] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# OTHER ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "GeoNLI API v2.8 - Attribute-Based Grounding with 8-Point OBB",
        "yolo_classes": list(YOLO_CLASSES.values()),
        "total_classes": len(YOLO_CLASSES),
        "caption_word_range": "40-60 words",
        "obb_format": "8-point normalized (x1,y1,x2,y2,x3,y3,x4,y4)",
        "endpoints": {
            "evaluation": "/api/v1/evaluate (POST JSON)",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "yolo": "loaded" if yolo_model else "not loaded",
        "yolo_classes": len(YOLO_CLASSES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=15200)
