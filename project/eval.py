import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

# === Your classifier model setup ===

from trainer import RasNetBackbone, RasNetWithExtraInput, load_data # <- update with your filename

det_class_names, det_class_to_idx = load_data()

# See what trainer.py gets
idx_to_class = {'battery': 0, 'biological': 1, 'brown-glass': 2, 'cardboard': 3, 'green-glass': 4, 'metal': 5, 'paper': 6, 'plastic': 7, 'trash': 8, 'white-glass': 9}
idx_to_class =  {v: k for k, v in idx_to_class.items()}
print(idx_to_class)

# === Init models ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = YOLO("yolo.pt")  # Path to your trained YOLO model

rasnet_backbone = RasNetBackbone()
model = RasNetWithExtraInput(
    num_classes=len(idx_to_class),
    num_detection_classes=len(det_class_to_idx),
    rasnet_backbone=rasnet_backbone
)
model.load_state_dict(torch.load("rasnet_best.pt", map_location=device))
model.to(device)
model.eval()

# === Image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load input image ===
image_path = "input.jpg"
orig_img = Image.open(image_path).convert("RGB")
orig_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)

# === Run YOLO detection ===
results = yolo(image_path)[0]
boxes = results.boxes  # XYXY + class_id

# === Inference loop ===
if boxes is None or len(boxes) == 0:
    print("⚠️ No detections — using full image with zeroed detection vector")

    crop = orig_img
    image_tensor = transform(crop).unsqueeze(0).to(device)

    det_one_hot = torch.zeros(len(det_class_to_idx), device=device)  # all zeros
    det_one_hot = det_one_hot.unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor, det_one_hot)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(1).item()
        pred_label = idx_to_class[pred_idx]
        confidence = probs[0, pred_idx].item()

    print(f"🖼️ Full-image prediction: {pred_label} ({confidence:.2f})")
else:
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        det_id = int(boxes.cls[i].item())
        det_label = yolo.model.names[det_id]

        print(f"Detected: {det_label} at {x1},{y1},{x2},{y2}")

        if det_label not in det_class_to_idx:
            print(f"⚠️ Skipping unknown detection label: {det_label}")
            continue

        crop = orig_img.crop((max(x1 - 50,0),max(y1 - 50, 0),x2 + 50,y2 + 50))
        image_tensor = transform(crop).unsqueeze(0).to(device)

        det_one_hot = torch.zeros(len(det_class_to_idx), device=device)
        det_one_hot[det_class_to_idx[det_label]] = 1
        det_one_hot = det_one_hot.unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor, det_one_hot)
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax(1).item()
            pred_label = idx_to_class[pred_idx]
            confidence = probs[0, pred_idx].item()

        print(f"🔍 Prediction: {pred_label} ({confidence:.2f})")

        # Optional: draw on image
        cv2.rectangle(orig_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_cv, f"{pred_label} ({confidence:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# === Save or display result image ===
cv2.imwrite("output_annotated.png", orig_cv)
print("Saved output_annotated.png")
