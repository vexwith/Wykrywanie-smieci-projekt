import os
import sys
from ultralytics import YOLO
import torch
from PIL import Image

print("Starting...")

detect = YOLO("recognizer/detect-model.pt")
classify = YOLO("recognizer/classify-model.pt")
classify_no_detect = YOLO("recognizer/classify-without-detect.pt")
print("Model loaded")


def recognize_image(image_path):
    print("Recognizing")
    result = classify_no_detect(image_path)[0]
    predicted_class = result.names[result.probs.top1]
    print("Recognized: ", predicted_class)

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_path = f"./outputs"
    for file_id, file in enumerate(os.listdir(f"{data_path}")):
        file_name = os.fsdecode(file)
        input_path = f"{data_path}/{file_name}"
        os.makedirs(output_path, exist_ok=True)
        # results = classify_no_detect(input_path)
        results = detect(input_path)
        for r_idx, r in enumerate(results):
            # r.save(f"{output_path}/{file_name}")
            # if r is None:
            #     recognize_image(input_path)
            image = Image.open(input_path)
            for box_id, box in enumerate(r.boxes):
                print("Detected: ", r.names[int(box.cls)], " prob: ", int(box.cls))
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                cropped = image.crop((x1 - 50, y1 - 50, x2 + 50, y2 + 50))
                cropped_path = f"{output_path}/{file_id}_{r_idx}_{box_id}.jpg"
                cropped.save(cropped_path)
                recognize_image(cropped_path)
        recognize_image(input_path)