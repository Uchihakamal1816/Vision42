import requests
import base64
import json
import cv2
import numpy as np
import sys
import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore")

def load_image_from_url(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img_array = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image from url: " + url)
    return img


def image_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


def normalize_name(s):
    if s is None:
        return None
    return s.lower().replace(" ", "").replace("-", "").replace("_", "")


def draw_obb_on_image(image, obb_list, color=(0, 255, 0), thickness=2):
    H, W = image.shape[:2]
    img = image.copy()

    for idx, ob in enumerate(obb_list, start=1):
        cx, cy, w, h, ang = ob

        cx *= W
        cy *= H
        w *= W
        h *= H

        ang_deg = np.degrees(ang)

        rect = ((cx, cy), (w, h), ang_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.drawContours(img, [box], 0, color, thickness)
        # cv2.putText(img, f"{idx}", (int(cx), int(cy)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img



def ask_qwen(image_b64, question, mode=None):

    if mode == "binary":
        question += "\nAnswer Yes or No."
    elif mode == "semantic":
        question += "\nAnswer in ONE WORD."

    payload = {
        "model": "/data1/MP3/qwen/",
        "messages": [
            {"role": "user",
             "content": [
                 {"type": "text", "text": question},
                 {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                  }
             ]}
        ],
        "max_tokens": 128
    }

    try:
        resp = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
        return resp.json()["choices"][0]["message"]["content"].strip()
    except:
        return None


def extract_grounding_keyword(question, image_b64):
    prompt = f"""
Extract the single object class mentioned in this grounding instruction.
Return one word (singular).

Instruction:
{question}

Examples:
Aircraft, Helicopter, Vehicle, Heavy-Truck, Airport, Ship,
StorageTank, Stadium, Bridge, Overpass, Dam, Harbor
"""
    cls = ask_qwen(image_b64, prompt, mode="semantic")
    if not cls:
        return None
    return cls.lower().split()[0]




def normalize_class_mapping(raw_class):
    if raw_class is None:
        return None

    raw = normalize_name(raw_class)

    mapping = {
        "aircraft": "airplane", "plane": "airplane", "jet": "airplane",
        "airplane": "airplane",

        "helicopter": "helicopter",

        "vehicle": "vehicle", "car": "vehicle", "van": "vehicle",

        "largevehicle": "large-vehicle", "truck": "large-vehicle",
        "smallvehicle": "small-vehicle",

        "storagetank": "storagetank", "tank": "storagetank",

        "airport": "airport",

        "baseballfield": "baseballfield",
        "basketballcourt": "basketballcourt",
        "tenniscourt": "tenniscourt",
        "golffield": "golffield",
        "groundtrackfield": "groundtrackfield",
        "soccerballfield": "soccer-ball-field",

        "bridge": "bridge", "dam": "dam", "harbor": "harbor",
        "overpass": "overpass", "chimney": "chimney",
        "windmill": "windmill", "roundabout": "roundabout",
        "containercrane": "container-crane",

        "expresswayservicearea": "expressway-service-area",
        "expresswaytollstation": "expressway-toll-station",

        "stadium": "stadium",

        "ship": "ship", "boat": "ship"
    }

    return mapping.get(raw, None)




def load_yolo(weights="/data1/MP3/best.pt"):
    try:
        model = YOLO(weights)
        print("Loaded YOLO OBB model:", weights)
        return model
    except Exception as e:
        print("YOLO load error:", e)
        return None


def run_yolo_obb(img, target_norm, model, conf=0.15):
    if target_norm is None:
        return []

    results = model(img, conf=conf, verbose=False)
    boxes = []

    for r in results:
        if not hasattr(r, "obb") or r.obb is None:
            continue

        names = r.names

        for obb in r.obb:
            cls_id = int(obb.cls[0])
            cls_name = names[cls_id]

            if normalize_name(cls_name) != target_norm:
                continue

            cx, cy, w, h, ang = obb.xywhr[0].tolist()

            H, W = img.shape[:2]
            cx /= W
            cy /= H
            w /= W
            h /= H

            boxes.append([cx, cy, w, h, ang])

    return boxes



def process_query(json_path, weights="/data1/MP3/best.pt"):
    with open(json_path, "r") as f:
        query = json.load(f)

    img_url = query["input_image"]["image_url"]
    img = load_image_from_url(img_url)
    img_b64 = image_to_base64(img)
    metadata = query.get("input_image", {}).get("metadata", {})
    spatial_resolution = metadata.get("spatial_resolution_m")

    caption_q = query["queries"]["caption_query"]["instruction"]
    binary_q = query["queries"]["attribute_query"]["binary"]["instruction"]
    numeric_q = query["queries"]["attribute_query"]["numeric"]["instruction"]
    semantic_q = query["queries"]["attribute_query"]["semantic"]["instruction"]
    grounding_q = query["queries"]["grounding_query"]["instruction"]

    caption_ans = ask_qwen(img_b64, caption_q)
    binary_ans = ask_qwen(img_b64, binary_q, mode="binary")
    semantic_ans = ask_qwen(img_b64, semantic_q, mode="semantic")

    
    target_raw = extract_grounding_keyword(grounding_q, img_b64)
    target_norm = normalize_class_mapping(target_raw)

    yolo = load_yolo(weights)
    grounding_boxes = run_yolo_obb(img, target_norm, yolo)

    H, W = img.shape[:2]
    grounding_response = []
    for i, obb in enumerate(grounding_boxes, 1):
        entry = {"object-id": str(i), "obbox": obb}

        if spatial_resolution and spatial_resolution > 0:
            width_px = obb[2] * W
            height_px = obb[3] * H
            width_m = width_px * spatial_resolution
            height_m = height_px * spatial_resolution
            area_m2 = width_m * height_m
            entry["metrics"] = {
                "width_m": round(width_m, 2),
                "height_m": round(height_m, 2),
                "area_m2": round(area_m2, 2)
            }

        grounding_response.append(entry)

    
    numeric_raw = extract_grounding_keyword(numeric_q, img_b64)
    numeric_norm = normalize_class_mapping(numeric_raw)

    if numeric_norm == target_norm:
        numeric_ans = str(len(grounding_boxes))
    else:
        num_boxes = run_yolo_obb(img, numeric_norm, yolo)
        numeric_ans = str(len(num_boxes))

    
    try:
        vis_img = draw_obb_on_image(img, grounding_boxes)
        vis_path = json_path.replace("_query.json", "_response_vis.png")
        cv2.imwrite(vis_path, vis_img)
        print("Saved visualization:", vis_path)
    except Exception as e:
        print("Visualization failed:", e)

    
    return {
        "input_image": query["input_image"],
        "queries": {
            "caption_query": {"instruction": caption_q, "response": caption_ans},
            "grounding_query": {"instruction": grounding_q, "response": grounding_response},
            "attribute_query": {
                "binary": {"instruction": binary_q, "response": binary_ans},
                "numeric": {"instruction": numeric_q, "response": numeric_ans},
                "semantic": {"instruction": semantic_q, "response": semantic_ans}
            }
        }
    }


if __name__ == "__main__":
    input_json = sys.argv[1]
    output_json = sys.argv[2]

    weights = "/data1/MP3/best.pt"
    if len(sys.argv) > 3:
        weights = sys.argv[3]

    out = process_query(input_json, weights)

    with open(output_json, "w") as f:
        json.dump(out, f, indent=4)

    print("Saved:", output_json)
