import os
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 1. Load detection label names from YOLO data.yaml
def load_data():
    with open("../TACO-master/yolo/data.yaml", "r") as f:
        yolo_data = yaml.safe_load(f)
    det_class_names = yolo_data["names"]
    det_class_to_idx = {name: i for i, name in enumerate(det_class_names)}
    return (det_class_names, det_class_to_idx)


# 2. Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust if RasNet expects different size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. Dataset
class HybridClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, det_class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.det_class_to_idx = det_class_to_idx
        self.class_to_idx = {}

        for class_idx, class_folder in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_folder] = class_idx
            for fname in os.listdir(class_path):
                if fname.endswith(('.jpg', '.png')):
                    fpath = os.path.join(class_path, fname)
                    det_label = fname.split('-')[1].split('_')[0]
                    self.samples.append((fpath, det_label, class_folder))
        print(self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _, class_folder = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        fname = os.path.basename(img_path)
        name = os.path.splitext(fname)[0]

        if not name.startswith("det-"):
            raise ValueError(f"Filename '{fname}' does not start with 'det-'")

        parts = name[4:].split('_')  # skip "det-"
        if len(parts) < 4:
            raise ValueError(f"Filename '{fname}' does not contain enough parts")

        det_label = '_'.join(parts[:-3])  # drop last 3 values

        det_idx = self.det_class_to_idx[det_label]
        one_hot = torch.zeros(len(self.det_class_to_idx))
        one_hot[det_idx] = 1.0

        class_idx = self.class_to_idx[class_folder]

        return image, one_hot, class_idx

# 4. RasNet backbone and model
class RasNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18  # Replace with your RasNet
        base = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.out_features = base.fc.in_features  # 512 for resnet18

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)

class RasNetWithExtraInput(nn.Module):
    def __init__(self, num_classes, num_detection_classes, rasnet_backbone):
        super().__init__()
        self.rasnet = rasnet_backbone
        self.classifier = nn.Sequential(
            nn.Linear(self.rasnet.out_features + num_detection_classes, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, det_one_hot):
        image_features = self.rasnet(image)
        combined = torch.cat([image_features, det_one_hot], dim=1)
        return self.classifier(combined)

# 5. Setup
if __name__ == "__main__":
    det_class_names, det_class_to_idx = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = HybridClassificationDataset("../output_objects", det_class_to_idx, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    rasnet_backbone = RasNetBackbone()
    model = RasNetWithExtraInput(
        num_classes=len(train_dataset.class_to_idx),
        num_detection_classes=len(det_class_to_idx),
        rasnet_backbone=rasnet_backbone
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 6. Training Loop
    epochs = 10
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, det_one_hot, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, det_one_hot, labels = images.to(device), det_one_hot.to(device), labels.to(device)

            outputs = model(images, det_one_hot)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        torch.save(model.state_dict(), "rasnet_last.pt")
        acc = correct / total * 100
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), "rasnet_best.pt")

        avg_loss = total_loss / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2f}%")