
from ultralytics import YOLO


if __name__ == "__main__":
    # cos()
    model1 = YOLO("./runs/detect/train5/weights/best.pt")
    # model2 = YOLO("yolo11n-cls.pt")

    for i in range(1, 15):
        results1 = model1(f"./fun_images/fota{i}.jpg")  # predict on an image
        # results2 = model1(f"./fun_images/fota_black{i}.jpg")  # predict on an image

        for idx, result in enumerate(results1):
            result.save(f"outputs/segmented_image_{i}_result_{idx}.jpg")
