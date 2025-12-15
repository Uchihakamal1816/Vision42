import requests
import json
from PIL import Image
import io
import re
import base64

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return img

def crop_region(img, query_text):
    w, h = img.size
    q = query_text.lower()

    # Region logic
    if "bottom right" in q or "lower right" in q:
        return img.crop((w*0.5, h*0.5, w, h))

    if "bottom left" in q or "lower left" in q:
        return img.crop((0, h*0.5, w*0.5, h))

    if "top right" in q or "upper right" in q:
        return img.crop((w*0.5, 0, w, h*0.5))

    if "top left" in q or "upper left" in q:
        return img.crop((0, 0, w*0.5, h*0.5))

    if "center" in q or "middle" in q or "central" in q:
        return img.crop((w*0.25, h*0.25, w*0.75, h*0.75))

    if "right" in q:
        return img.crop((w*0.5, 0, w, h))

    if "left" in q:
        return img.crop((0, 0, w*0.5, h))

    if "top" in q:
        return img.crop((0, 0, w, h*0.5))

    if "bottom" in q:
        return img.crop((0, h*0.5, w, h))

    # Default: full image
    return img


def save_temp_image(img, path="temp_crop.png"):
    img.save(path)
    return path


def query_vllm(question, image_path):
    # Read and convert to base64
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "model": "/data1/MP3/qwen",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image",
                        "image": img_b64
                    }
                ]
            }
        ]
    }

    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        timeout=120
    )

    try:
        res = response.json()
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error parsing response: {e}\nRAW RESPONSE:\n{response.text}"


if __name__ == "__main__":
    # The question you want to test
    question = "Is there any digit present in the bottom right corner of the scene?"

    # Step 1: Load full image
    img = load_image_from_url("https://bit.ly/4ouV45l")

    # Step 2: Crop based on question
    cropped = crop_region(img, question)

    # Step 3: Save cropped region
    crop_path = save_temp_image(cropped)

    # Step 4: Query model
    answer = query_vllm(question, crop_path)

    print("\nQUESTION:", question)
    print("ANSWER:", answer)
