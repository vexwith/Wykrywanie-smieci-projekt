from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11l.pt")
    results = model.train(data='yolo/data.yaml', epochs=200, imgsz=640,device=[1])
