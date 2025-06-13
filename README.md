## Backend do projektu wykrywania śmieci
Porównanie kilku sposobów detekcji i klasyfikacji śmieci ze zdjęć na modelach YOLO i Rasnet

## Startup
Do przetestowania skryptu potrzebna jest struktura projektu

```bash
project/
│
├── fun_images/ # Przykładowe obrazy testowe
├── outputs/ # Wygenerowane wyniki
├── recognizer/
│ ├── classify-model.pt # Model klasyfikacji
│ ├── classify-without-detect.pt # Model klasyfikacji bez detekcji
│ └── detect-model.pt # Model detekcji
│
├── recognize_oneshot.py # Końcowy backend użyty do detekcji i klasyfikacji
├── yolo.py # Pierwotny plik klasyfikujący i segregujący (wraz z treningiem)
├── detect_and_crop.py # Wielowątkowo używa YOLO detect i przycina śmieci
├── classification.py # Trenuje model segregujący śmieci na podstawie nazw
├── eval.py # Skrypt ewaluacyjny Rasnet
├── trainer.py # Skrypt treningowy Rasnet
│
├── splitter.py # Dzieli dane w folderach na treningowe i testowe
├── train_classify.py # Trening modelu klasyfikacji
├── segment_train.py # Trening modelu segmentacji
├── segment_test.py # Test modelu segmentacji
│
├── waste_classification_model.pth # Model klasyfikacji śmieci
└── rasnet_best.pt # Wytrenowany model Rasnet

```

## Główne skrypty

### recognize_oneshot.py
Porównuje:
- Klasyfikację na całych obrazkach
- Klasyfikację na przyciętych fragmentach (po detekcji)
Pozwala ocenić skuteczność obu podejść.

### yolo.py
Pierwotny plik klasyfikujący, który:
1. Wykorzystuje `classify-without-detect.pt` do klasyfikacji śmieci na obrazie
2. Używa `waste_classification_model.pth` do segregacji wykrytych śmieci
3. Generuje opatrzone adnotacjami wyniki w formie obrazów

### detect_and_crop.py
Wielowątkowy skrypt, który:
1. Wykorzystuje YOLO do detekcji śmieci
2. Przycina wykryte obiekty jako osobne obrazy
3. Zapisuje wyniki w określonym folderze

### classification.py
Służy do trenowania modelu segregującego śmieci na podstawie ich nazw i kategorii.

