import os
import sys
from ultralytics import YOLO
import torch
from PIL import Image

print("Starting...")

detect = YOLO("recognizer/detect-model.pt")
classify = YOLO("recognizer/classify-model.pt")
print("Model loaded")


def recognize_image(image_path):
    print("Recognizing")
    result = classify(image_path)[0]
    predicted_class = result.names[result.probs.top1]
    print("Recognized: ", predicted_class, "other: ", result)

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_path = f"{data_path}/output"
    input_path = f"{data_path}/input.png"
    os.makedirs(output_path, exist_ok=True)
    results = detect(input_path)
    for r_idx, r in enumerate(results):
        if r.boxes is None:
            recognize_image(input_path)
        else:
            image = Image.open(input_path)
            for box_id, box in enumerate(r.boxes):
                print("Detected: ", r.names[int(box.cls)], " prob: ", int(box.cls))
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                cropped = image.crop((x1, y1, x2, y2))
                cropped_path = f"{output_path}/{r_idx}_{box_id}.webp"
                cropped.save(cropped_path)
                recognize_image(cropped_path)
    pass