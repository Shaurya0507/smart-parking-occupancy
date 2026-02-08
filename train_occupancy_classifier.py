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
MODEL = "occupancy_classifier.pt"
SEED = 42
# These values are based on the average RGB color and spread of RGB values across the ImageNet dataset.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# This function establishes fixed random seeds to ensure reproducible training results.
def set_random_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seeds(SEED)

# Here, I transform and augment the images in the train folder to improve training performance.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale = (0.8, 1.0)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness = 0.35, contrast = 0.35, saturation = 0.20, hue = 0.02)
    ], p = 0.8),
    transforms.RandomRotation(degrees = 8),
    transforms.ToTensor(),
    transforms.Normalize(mean = MEAN, std = STD),
])

# Here, I transform the images in the val folder without augmentation to match the needed training input format.
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = MEAN, std = STD),
])

# In this section, the training and validation datasets are loaded and data loaders are created to feed images to the model.
train_dataset = datasets.ImageFolder(TRAIN, transform = train_transforms)
val_dataset = datasets.ImageFolder(VAL, transform = val_transforms)
train_images = DataLoader(train_dataset, batch_size = SIZE, shuffle = True)
val_images = DataLoader(val_dataset, batch_size = SIZE, shuffle = False)
# A ResNet-18 model is initialized for binary classification and set up training components.
model = models.resnet18(weights = None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

def eval_metrics():
    model.eval()
    correct = 0
    total = 0
    # Here, I built a confusion matrix where rows = true labels and columns = predicted labels.
    matrix = torch.zeros((2, 2), dtype = torch.int64)
    # The model is evaluated here on the validation set, tracking accuracy and a confusion matrix.
    with torch.no_grad():
        for imgs, labels in val_images:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            predictions = outputs.argmax(dim = 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels.view(-1), predictions.view(-1)):
                matrix[int(t), int(p)] += 1

    accuracy = correct / max(total, 1)
    # Recall per class: TP / (TP + FN)
    recall0 = (matrix[0, 0].item() / max((matrix[0, 0] + matrix[0, 1]).item(), 1))
    recall1 = (matrix[1, 1].item() / max((matrix[1, 0] + matrix[1, 1]).item(), 1))
    balanced_accuracy = 0.5 * (recall0 + recall1)

    return accuracy, balanced_accuracy, (recall0, recall1), matrix

# These variables store the top validation metrics so far.
best_balanced_accuracy = -1.0
best_accuracy = -1.0

# This loop trains the model for a fixed number of epochs and evaluates performance after each epoch.
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    # For each batch, the model runs forward, computes the loss, and updates the parameters.
    for images, labels in train_images:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / max(len(train_images), 1)
    # Here, I evaluate the model on the validation set and compute metrics.
    val_accuracy, val_balanced_accuracy, (rec0, rec1), matrix = eval_metrics()
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Accuracy: {val_accuracy:.3f} "
        f"- Val Balanced Accuracy: {val_balanced_accuracy:.3f} - Recall(occupied/unoccupied): ({rec0:.3f}/{rec1:.3f})"
    )
    print(f"Confusion Matrix: rows = true, cols = pred: [[{matrix[0, 0].item()}, {matrix[0, 1].item()}], [{matrix[1, 0].item()}, {matrix[1, 1].item()}]]")
    # This section saves the model whenever it achieves a new best balanced validation accuracy.
    if val_balanced_accuracy > best_balanced_accuracy:
        best_balanced_accuracy = val_balanced_accuracy
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), MODEL)
        print(f"Saved the new best model.")

print("25/25 epochs are complete. Best val balanced accuracy: ", best_balanced_accuracy, "(val accuracy:", best_accuracy, ")")