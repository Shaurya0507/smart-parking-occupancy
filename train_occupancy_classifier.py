import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# These are the paths and runtime configurations.
TRAIN = "dataset/train"
VAL   = "dataset/val"
DEVICE = "cpu"
SIZE = 8
EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
MODEL = "occupancy_classifier_check.pt"
SEED = 42

# This function establishes fixed random seeds to ensure consistency.
def set_random_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seeds(SEED)

# I based these values on the average RGB color and the spread of RGB values across all ImageNet images.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Here, I transform the images in the train folder to match the needed training input format.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale = (0.8, 1.0)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness = 0.35, contrast = 0.35, saturation = 0.20, hue = 0.02)
    ], p = 0.8),
    transforms.RandomRotation(degrees = 8),
    transforms.ToTensor(),
    transforms.Normalize(mean = MEAN, std = STD),
])

# Here, I transform the images in the val folder to match the needed training input format.
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = MEAN, std = STD),
])

train_dataset = datasets.ImageFolder(TRAIN, transform = train_transforms)
val_dataset = datasets.ImageFolder(VAL, transform = val_transforms)
train_loader = DataLoader(train_dataset, batch_size = SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = SIZE, shuffle = False)

model = models.resnet18(weights = None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
# Using AdamW (Adaptive Moment Estimation), I optimized the model's weights and biases to minimize the cross-entropy loss.
optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

def eval_metrics():
    model.eval()
    correct = 0
    total = 0
    # # Here, I built a confusion matrix where rows = true labels and columns = predicted labels.
    matrix = torch.zeros((2, 2), dtype = torch.int64)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            predictions = outputs.argmax(dim = 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels.view(-1), predictions.view(-1)):
                matrix[int(t), int(p)] += 1

    accuracy = correct / max(total, 1)
    # Recall formula: TP / (TP + FN)
    recall0 = (matrix[0, 0].item() / max((matrix[0, 0] + matrix[0, 1]).item(), 1))
    recall1 = (matrix[1, 1].item() / max((matrix[1, 0] + matrix[1, 1]).item(), 1))
    balanced_accuracy = 0.5 * (recall0 + recall1)

    return accuracy, balanced_accuracy, (recall0, recall1), matrix

best_balanced_accuracy = -1.0
best_accuracy = -1.0

#This function trains the model.
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / max(len(train_loader), 1)

    val_accuracy, val_balanced_accuracy, (rec0, rec1), matrix = eval_metrics()
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Accuracy: {val_accuracy:.3f} "
        f"- Val Balanced Accuracy: {val_balanced_accuracy:.3f} - Recall(occupied/unoccupied): ({rec0:.3f}/{rec1:.3f})"
    )
    print(f"Confusion Matrix: rows = true, cols = pred: [[{matrix[0, 0].item()}, {matrix[0, 1].item()}], [{matrix[1, 0].item()}, {matrix[1, 1].item()}]]")

    if val_balanced_accuracy > best_balanced_accuracy:
        best_balanced_accuracy = val_balanced_accuracy
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), MODEL)
        print(f"Saved this model.")

print("25/25 epochs are complete. Best val balanced accuracy: ", best_balanced_accuracy, "(val accuracy:", best_accuracy, ")")