import os
import cv2
from ultralytics import YOLO
from PIL import Image
import concurrent

# === Configuration ===
input_dir = "fun_images"
output_dir = "output_objects"
model_path = "recognizer/detect-model.py"
skipped = ["paper"]

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_images(class_folder_path, class_output_path):
    print(f"Processing class {class_name}: {class_input_path} {class_output_path}")
    model = YOLO(model_path)
    for filename in os.listdir(class_folder_path):
        path = f"{class_folder_path}/{filename}"
        results = model(path)
        print(f"Got results {len(results)} for {filename}")
        image = Image.open(path).convert("RGB")
        for id, r in enumerate(results):
            boxes = r.boxes
            if boxes is None:
                print("No detection for {filename}")
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                cropped = image.crop((x1,y1,x2,y2))
                save_path = f"{class_output_path}/{id}_{i}_{filename}"
                cropped.save(save_path)


# Process all images
if __name__ == "__main__":
    executor = concurrent.futures.ProcessPoolExecutor()
    for class_name in os.listdir(input_dir):
        class_input_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_input_path) or class_name in skipped:
            continue
        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)
        executor.submit(process_images, class_input_path, class_output_path)
    
