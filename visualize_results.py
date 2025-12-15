import json
import cv2
import numpy as np
import requests
import sys
import os

def load_image_from_url(url):
    resp = requests.get(url)
    img_array = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def draw_obb(img, obb_norm, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    cx, cy, bw, bh, angle = obb_norm
    
    # Denormalize
    cx *= w
    cy *= h
    bw *= w
    bh *= h
    
    # Create the rotated rectangle
    rect = ((cx, cy), (bw, bh), angle)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    cv2.drawContours(img, [box], 0, color, thickness)

def visualize(json_path, output_image_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Load image
    img_url = data['input_image']['image_url']
    print(f"Loading image from: {img_url}")
    img = load_image_from_url(img_url)
    
    if img is None:
        print("Failed to load image.")
        return

    # Draw Grounding Boxes
    grounding_response = data['queries']['grounding_query']['response']
    print(f"Found {len(grounding_response)} objects.")
    
    for obj in grounding_response:
        obb = obj['obbox']
        draw_obb(img, obb)
        
        # Draw ID
        # cx, cy = obb[0] * img.shape[1], obb[1] * img.shape[0]
        # cv2.putText(img, obj['object-id'], (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Save image
    cv2.imwrite(output_image_path, img)
    print(f"Saved visualization to: {output_image_path}")
    
    # Print other results
    print("\n--- Caption ---")
    print(data['queries']['caption_query']['response'])
    
    print("\n--- Binary Attribute ---")
    print(f"Instruction: {data['queries']['attribute_query']['binary']['instruction']}")
    print(f"Response: {data['queries']['attribute_query']['binary']['response']}")
    
    print("\n--- Numeric Attribute ---")
    print(f"Instruction: {data['queries']['attribute_query']['numeric']['instruction']}")
    print(f"Response: {data['queries']['attribute_query']['numeric']['response']}")
    
    print("\n--- Semantic Attribute ---")
    print(f"Instruction: {data['queries']['attribute_query']['semantic']['instruction']}")
    print(f"Response: {data['queries']['attribute_query']['semantic']['response']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <response_json_path>")
        sys.exit(1)
        
    response_json_path = sys.argv[1]
    output_image_path = response_json_path.replace(".json", "_vis.png")
    
    visualize(response_json_path, output_image_path)
