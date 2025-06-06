from ultralytics import YOLO
from classification import WasteClassificationNN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import string


garbage_classes = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]
recycle_categories = {0: "plastic and metal", 1: "paper", 2: "glass", 3: "bio", 4: "mixed"}

def createDataFolderStructure():
    global garbage_classes

    for garbage_class in garbage_classes:
        os.makedirs(f"data_black/train/{garbage_class}", exist_ok=True)
        os.makedirs(f"data_black/test/{garbage_class}", exist_ok=True)


def treningModelu():
    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="data_black", epochs=2, imgsz=640)

def maskowanie(model, all_files, data_base_path, garbage_class, data_out_path):
    for file in all_files:
        file_name = os.fsdecode(file)
        results = model(f"./{data_base_path}/{garbage_class}/{file_name}", verbose=False)  # predict on an image

        if results[0] is None:
            continue

        if results[0].masks is None:
            continue

        img = np.copy(results[0].orig_img)
        b_mask = np.zeros(img.shape[:2], np.uint8)
        image = np.zeros_like(img)

        for mask in results[0].masks.xy:
            # Konwersja maski do typu int32 i odpowiedniego kształtu
            contour = mask.astype(np.int32).reshape(-1, 1, 2)

            # Tworzenie binarnej maski
            b_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(b_mask, [contour], -1, 255, cv2.FILLED)

            # Konwersja maski do formatu 3-kanałowego
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)

            # Izolacja obiektu na podstawie maski
            isolated = cv2.bitwise_and(mask3ch, img)

            # Dodanie izolowanego obiektu do końcowego obrazu
            image = cv2.bitwise_or(isolated, image)
        cv2.imwrite(f"{data_out_path}/{garbage_class}/{file_name}", image)

def segmentacja():
    model = YOLO("yolov8s-seg.pt")
    data_test_base_path = "data/test"
    data_train_base_path = "data/train"

    data_test_out_path = "data_black/test"
    data_train_out_path = "data_black/train"

    for garbage_class in garbage_classes:
        all_train_files = os.listdir(f"{data_train_base_path}/{garbage_class}")
        all_test_files = os.listdir(f"{data_test_base_path}/{garbage_class}")

        #train
        maskowanie(model, all_train_files, data_train_base_path, garbage_class, data_train_out_path)
        #test
        maskowanie(model, all_test_files, data_test_base_path, garbage_class, data_test_out_path)

def auto_mapping(yolo_class):
    yolo_class = yolo_class.lower()
    for plastic in ["can", "plastic", "metal", "pop"]:
        if plastic in yolo_class:
            return "plastic and metal"
    for paper in ["paper", "cardboard", "carton"]:
        if paper in yolo_class:
            return "paper"
    for bio in ["food", "fruit", "bio"]:
        if bio in yolo_class:
            return "bio"
    if "glass" in yolo_class:
        return "glass"
    else:
        return "mixed"  # default

def zapisz_do_pliku(tekst, nazwa_pliku):
    """
    Zapisuje podany tekst do pliku, rozpoczynając od nowej linii.

    :param tekst: String do zapisania
    :param nazwa_pliku: Nazwa pliku docelowego
    """
    with open(nazwa_pliku, 'a', encoding='utf-8') as plik:
        plik.write(tekst+'\n')

def znajdz_najczestsze_slowo(nazwa_pliku):
    """
    Czyta plik tekstowy i zwraca najczęściej występujące słowo wraz z liczbą wystąpień.
    W przypadku remisu zwraca pierwsze napotkane słowo o maksymalnej częstotliwości.

    :param nazwa_pliku: Nazwa pliku do analizy
    :return: Krotka (słowo, liczba_wystąpień)
    """
    licznik = {}

    with open(nazwa_pliku, 'r', encoding='utf-8') as plik:
        for linia in plik:
            # Usuwanie znaków interpunkcyjnych i podział na słowa
            slowa = linia.translate(str.maketrans('', '', string.punctuation)).lower().split()
            for slowo in slowa:
                licznik[slowo] = licznik.get(slowo, 0) + 1

    if not licznik:
        return None

    return {k: v for k, v in sorted(licznik.items(), key=lambda item: item[1], reverse=True)}

def wczytanie_labeli():
    model_data = torch.load("runs/classify/train/weights/best.pt", map_location='cpu', weights_only=False)
    # Sprawdzenie, czy etykiety są w modelu
    if 'model' in model_data and hasattr(model_data['model'], 'names'):
        return model_data['model'].names  # Nowe wersje YOLO
    elif 'names' in model_data:
        return model_data['names']  # Starsze wersje
    elif 'ema' in model_data and hasattr(model_data['ema'], 'names'):
        return model_data['ema'].names
    else:
        raise ValueError("Nie znaleziono etykiet w pliku modelu!")

def test_waste_classification(user_input, eval_model, max_word_length):
    input_tensor = torch.zeros(1, max_word_length)
    for i, char in enumerate(user_input[:max_word_length]):
        input_tensor[0, i] = ord(char)  # Simple ASCII encoding

    # Get prediction
    with torch.no_grad():
        prediction = eval_model.predict(input_tensor)
        predicted_class = torch.argmax(prediction).item()

    print(f"System proponuje kategorię {predicted_class}: {recycle_categories[predicted_class]}")
    return recycle_categories[predicted_class]

def main():
    # createDataFolderStructure()
    # segmentacja()
    # treningModelu()

    # model1 = YOLO("./runs/classify/train/weights/best.pt")
    model1 = YOLO("./runs/classify/train/weights/best.pt")

    labels = wczytanie_labeli()
    # Konwersja na listę (jeśli to słownik np. {0: 'class1', 1: 'class2'})
    if isinstance(labels, dict):
        label_list = list(labels.values())
    else:
        label_list = labels  # Jeśli już jest listą
    longest_word = max(label_list, key=len)
    max_word_length = len(longest_word)  # Najdłuższe słowo w zbiorze

    model_path = 'waste_classification_model.pth'
    eval_model = WasteClassificationNN(max_word_length=max_word_length)
    eval_model.load_state_dict(torch.load(model_path))
    eval_model.eval()

    recycle_map = dict()

    for file in os.listdir(f"./fun_images"):
        file_name = os.fsdecode(file)
        results1 = model1(f"./fun_images/{file_name}")  # predict on an image

        for idx, result in enumerate(results1):
            result.save(f"./outputs/{file_name}")
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
            for name in names:
                zapisz_do_pliku(name, "test.txt")
                rec_class = test_waste_classification(name, eval_model, max_word_length)
                recycle_map[name] = rec_class
    licznik = znajdz_najczestsze_slowo("test.txt")
    print(licznik)
    print(recycle_map)

if __name__ == "__main__":
    main()

# 15:00 EA530
