from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("classify-model-best.pt")
    results = model.train(data='data-regular', epochs=200, imgsz=640,device=[1])
