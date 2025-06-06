import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

recycle_categories = {0: "plastic and metal", 1: "paper", 2: "glass", 3: "bio", 4: "mixed"}

class WasteClassificationNN(nn.Module):
    def __init__(self, max_word_length, hidden_size=128, num_classes=5):
        """
        Inicjalizacja sieci neuronowej do klasyfikacji śmieci

        Args:
            max_word_length (int): Długość najdłuższego słowa w zbiorze danych
            hidden_size (int): Rozmiar warstw ukrytych
            num_classes (int): Liczba klas (pojemników na śmieci)
        """
        super(WasteClassificationNN, self).__init__()

        # Warstwa wejściowa - zakładamy, że każdy znak jest reprezentowany jako wartość numeryczna (ASCII)
        self.input_size = max_word_length

        # Warstwy ukryte
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Warstwa wyjściowa
        self.fc_out = nn.Linear(hidden_size, num_classes)

        # Funkcje aktywacji
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Przepływ danych przez sieć
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)
        return x

    def predict(self, x):
        # Predykcja z użyciem softmax
        with torch.no_grad():
            outputs = self.forward(x)
            probs = self.softmax(outputs)
        return probs


class HumanReinforcementLearner:
    def __init__(self, model, max_word_length, memory_size=1000, batch_size=32, learning_rate=0.001):
        """
        Klasa do uczenia z reinforcement learning z udziałem człowieka

        Args:
            model (nn.Module): Model sieci neuronowej
            max_word_length (int): Maksymalna długość słowa
            memory_size (int): Rozmiar pamięci doświadczeń
            batch_size (int): Rozmiar batcha do uczenia
            learning_rate (float): Szybkość uczenia
        """
        self.model = model
        self.max_word_length = max_word_length
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def preprocess_input(self, word):
        """
        Przetwarzanie słowa na wektor numeryczny
        """
        # Konwersja znaków na wartości ASCII i wypełnienie zerami do max_word_length
        word_ascii = [ord(c) for c in word.lower()]

        # Padding jeśli słowo jest za krótkie
        if len(word_ascii) < self.max_word_length:
            word_ascii += [0] * (self.max_word_length - len(word_ascii))
        else:
            word_ascii = word_ascii[:self.max_word_length]

        return torch.FloatTensor(word_ascii).unsqueeze(0)  # Dodajemy wymiar batcha

    def remember(self, state, action, reward):
        """
        Zapamiętanie doświadczenia
        """
        self.memory.append((state, action, reward))

    def train_on_memory(self):
        """
        Uczenie na podstawie zgromadzonych doświadczeń
        """
        # print(f"Przykładowe dane z pamięci: {self.memory[0]}")
        if len(self.memory) < self.batch_size:
            print(f"Za mało próbek: {len(self.memory)} < {self.batch_size}")
            return  # Nie ma wystarczająco danych

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards = zip(*batch)

        states = torch.cat(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        # Forward pass
        outputs = self.model(states)

        # Debug: sprawdź kształty tensorów
        print(f"Outputs: {outputs.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}")

        # Obliczenie strat (ważone przez nagrody)
        loss = (self.criterion(outputs, actions) * rewards).mean()

        # Backward pass i optymalizacja
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

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

def auto_mapping(class_name):
    class_name = class_name.lower()
    for plastic in ["can", "plastic", "metal", "pop"]:
        if plastic in class_name:
            return 0
    for paper in ["paper", "cardboard", "carton"]:
        if paper in class_name:
            return 1
    for bio in ["food", "fruit", "bio"]:
        if bio in class_name:
            return 3
    if "glass" in class_name:
        return 2
    else:
        return 4  # default

# Przykład użycia
if __name__ == "__main__":
    # Przykładowe dane
    num_classes = 5  # 5 pojemników na śmieci

    labels = wczytanie_labeli()
    # Konwersja na listę (jeśli to słownik np. {0: 'class1', 1: 'class2'})
    if isinstance(labels, dict):
        label_list = list(labels.values())
    else:
        label_list = labels  # Jeśli już jest listą
    longest_word = max(label_list, key=len)
    max_word_length = len(longest_word)  # Najdłuższe słowo w zbiorze

    print(label_list)
    label_list = label_list + label_list.copy()

    # Inicjalizacja modelu
    model = WasteClassificationNN(max_word_length=max_word_length, hidden_size=128, num_classes=num_classes)
    learner = HumanReinforcementLearner(model, max_word_length=max_word_length)

    # Przykładowa interakcja z człowiekiem
    def human_feedback_loop():
        print("System klasyfikacji śmieci - tryb uczenia z udziałem człowieka")
        print("Podaj nazwę śmiecia (np. 'plastikowa butelka'), a system spróbuje je sklasyfikować.")
        print("Następnie podaj poprawną kategorię (0-4) i ocenę (1-10) jak dobrze system sobie poradził.")
        print("Wpisz 'exit' aby zakończyć.")

        counter = 0
        while True:
            # user_input = input("\nWpisz nazwę śmiecia: ").strip()
            user_input = label_list[counter]
            counter += 1
            print(user_input)

            if not user_input:
                continue

            # Przetwarzanie wejścia
            state = learner.preprocess_input(user_input)

            # Predykcja
            with torch.no_grad():
                probs = model.predict(state)
                predicted_class = torch.argmax(probs).item()

            print(f"System proponuje kategorię: {predicted_class}")
            print("Prawdopodobieństwa dla każdej kategorii:")
            for i, prob in enumerate(probs.squeeze().numpy()):
                print(f"{recycle_categories[i]} ({i}): {prob * 100:.1f}%")

            # Pobranie feedbacku od człowieka
            try:
                auto_category = auto_mapping(user_input)
                if auto_category == 4:
                    correct_class_str = (input("Podaj poprawną kategorię (0-4): "))
                    if correct_class_str.lower() == 'exit':
                        break
                    correct_class = int(correct_class_str)

                    if correct_class < 0 or correct_class >= num_classes:
                        print("Nieprawidłowa kategoria, pomijam...")
                        continue

                    rating = float(input("Oceń jakość predykcji (1-10, gdzie 10 to najlepsza): "))
                    if rating < 1 or rating > 10:
                        print("Ocena poza zakresem, ustawiam 5")
                        rating = 5

                    # Normalizacja nagrody do zakresu [0, 1]
                    reward = rating / 10.0

                    # Jeśli predykcja była poprawna, zwiększ nagrodę
                    if predicted_class == correct_class:
                        reward = min(1.0, reward + 0.2)
                #automatyczne uczenie
                else:
                    correct_class = auto_category
                    reward = 1.0 if predicted_class == correct_class else 0.0

                # Zapamiętanie doświadczenia
                learner.remember(state, correct_class, reward)

                # Uczenie na podstawie pamięci
                loss = learner.train_on_memory()
                print(f"Doświadczenie zapisane. Strata: {loss if loss is not None else 'N/A'}")

            except ValueError:
                print("Nieprawidłowe dane wejściowe, pomijam...")

            #end condition
            if counter == len(label_list):
                break

    def test_waste_classification(model_path='waste_classification_model.pth'):
        eval_model = WasteClassificationNN(max_word_length=max_word_length)
        eval_model.load_state_dict(torch.load(model_path))
        eval_model.eval()

        print("Waste Classification Tester")
        print("Enter waste item descriptions (or 'quit' to exit)")
        print("Example inputs: plastic bottle, cardboard box, food waste")

        while True:
            user_input = input("\nEnter waste item: ").lower()
            if user_input == 'quit':
                break

            if not user_input:
                continue

            # Preprocess input
            input_tensor = torch.zeros(1, max_word_length)
            for i, char in enumerate(user_input[:max_word_length]):
                input_tensor[0, i] = ord(char)  # Simple ASCII encoding

            # Get prediction
            with torch.no_grad():
                prediction = eval_model.predict(input_tensor)
                predicted_class = torch.argmax(prediction).item()

            print(f"System proponuje kategorię {predicted_class}: {recycle_categories[predicted_class]}")
            print("Prawdopodobieństwa dla każdej kategorii:")
            for i, prob in enumerate(prediction.squeeze().numpy()):
                print(f"{recycle_categories[i]} ({i}): {prob * 100:.1f}%")

    def test():
        # Przykładowe dane (5 próbek)
        for _ in range(33):
            state = torch.randn(1, max_word_length)  # Losowy stan
            action = random.randint(0, num_classes - 1)  # Losowa akcja
            reward = random.uniform(0.5, 1.0)  # Losowa nagroda
            learner.remember(state, action, reward)

        # Uczenie
        loss = learner.train_on_memory()
        print(f"Loss: {loss}")

    # Uruchomienie pętli feedbacku
    # human_feedback_loop()
    # test()
    test_waste_classification()

    # Zapis modelu po uczeniu
    # torch.save(model.state_dict(), "waste_classification_model.pth")
    # print("Model zapisany do waste_classification_model.pth")