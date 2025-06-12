import os
import numpy as np
import shutil
import argparse
from tqdm import tqdm
from typing import NoReturn

#garbage_classes = ["battery", "biological", "cardboard", "metal", "paper", "plastic", "trash", "glass"]
garbage_classes = ["biological", "cardboard", "metal", "paper", "plastic", "trash", "glass"]


def createModelFolderStructure(model_name: str) -> NoReturn:
    result_dir = f"results/{model_name}"
    models_dir = os.path.join(result_dir, "models")
    plots_dir = os.path.join(result_dir, "plots")

    for dir_path in [result_dir, models_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)


def createDataFolderStructure() -> NoReturn:
    global garbage_classes

    for garbage_class in garbage_classes:
        os.makedirs(f"data-regular//train/{garbage_class}", exist_ok=True)
        os.makedirs(f"data-regular/test/{garbage_class}", exist_ok=True)


def split_data(source_dir, train_dir, test_dir, split_ratio=0.8, force_split=False) -> tuple[int, int]:
    """Zwraca wielkość zbiorów danych"""

    if not os.path.exists(source_dir):
        print(f"Błąd: folder źródłowy {source_dir} nie istnieje.")
        return 0, 0

    files = os.listdir(source_dir)
    train_files = os.listdir(train_dir) if os.path.exists(train_dir) else []
    test_files = os.listdir(test_dir) if os.path.exists(test_dir) else []

    if force_split:
        if os.path.exists(train_dir):
            for file in os.listdir(train_dir):
                os.remove(os.path.join(train_dir, file))
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, file))

    if len(train_files) > 0 and len(test_files) > 0 and not force_split:
        print(f"Foldery {train_dir} i {test_dir} już zawierają dane.")
        print(f"Liczba plików treningowych: {len(train_files)}")
        print(f"Liczba plików testowych: {len(test_files)}")
        return len(train_files), len(test_files)

    # Stały seed dla powtarzalności
    np.random.seed(42069)
    np.random.shuffle(files)
    train_size = int(len(files) * split_ratio)

    train_files = files[:train_size]
    test_files = files[train_size:]

    for file_name in tqdm(train_files, desc=f"Kopiowanie plików treningowych z {os.path.basename(source_dir)}"):
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(train_dir, file_name)
        shutil.copy(src, dst)

    for file_name in tqdm(test_files, desc=f"Kopiowanie plików testowych z {os.path.basename(source_dir)}"):
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(test_dir, file_name)
        shutil.copy(src, dst)

    return len(train_files), len(test_files)


def prepare_data(split_ratio=0.8, force_split=False, debug_info=False) -> bool:
    """Zwraca czy przygotowanie danych się udało"""
    global garbage_class

    number_of_classes = len(garbage_classes)

    source_dir = "garbage-dataset"
    base_dir = "data-regular"
    garbage_dirs = []
    train_dirs = []
    test_dirs = []

    train_count = [0] * number_of_classes
    test_count = [0] * number_of_classes

    for garbage_class in garbage_classes:
        garbage_dir = os.path.join(source_dir, garbage_class)
        garbage_dirs.append(garbage_dir)

        train_dir = os.path.join(f"{base_dir}/train", garbage_class)
        train_dirs.append(train_dir)

        test_dir = os.path.join(f"{base_dir}/test", garbage_class)
        test_dirs.append(test_dir)

    for directory in garbage_dirs:
        if not os.path.exists(directory):
            print(f"Błąd: katalog źródłowy {directory} nie istnieje.")
            print("Upewnij się, że struktura folderów jest poprawna i dane są na swoim miejscu.")
            return False

    print(f"Dzielenie danych na zbiory treningowy ({split_ratio * 100}%) i testowy ({round((1 - split_ratio) * 100)}%)...")

    for i in range(number_of_classes):
        train_count[i], test_count[i] = split_data(garbage_dirs[i], train_dirs[i], test_dirs[i], split_ratio, force_split)

    if debug_info:
        for i in range(number_of_classes):
            print(f"\nLiczba obrazów treningowych {garbage_classes[i]}: {train_count[i]}")
            print(f"Liczba obrazów testowych {garbage_classes[i]}: {test_count[i]}")
            print(f"Łącznie obrazów: {train_count[i] + test_count[i]}")

    print("\nPrzygotowanie danych zakończone!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Klasyfikacja śmieci")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Proporcja podziału na zbiór treningowy (domyślnie: 0.8)")
    parser.add_argument("--skip-data-prep", action="store_true", help="Pomiń przygotowanie danych (użyj, gdy dane są już podzielone)")
    parser.add_argument("--force-split", action="store_true", help="Wymuś ponowny podział danych (usunie istniejące pliki w folderach train/test)")

    args = parser.parse_args()

    if not args.skip_data_prep:
        createModelFolderStructure("Yolo")
        createDataFolderStructure()
        success = prepare_data(split_ratio=args.split_ratio, force_split=args.force_split, debug_info=True)
        if not success:
            print("Błąd podczas przygotowywania danych. Przerwanie programu.")
            return


main()
