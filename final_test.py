import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# These are the paths and runtime configurations.
TEST = "dataset/test"
FILE = "occupancy_classifier.pt"
DEVICE = "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 16
# # These values are based on the average RGB color and spread of RGB values across the ImageNet dataset.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Here, I transform the images to match the input format used during training.
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = MEAN, std = STD)
])

# # In this function, I build the same neural network architecture used during training.
def pytorch_model() -> torch.nn.Module:
    model = models.resnet18(weights = None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if not os.path.exists(FILE):
        raise FileNotFoundError(f"Model weights not found in '{FILE}'.")

    state = torch.load(FILE, map_location = DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def main():
    if not os.path.exists(TEST):
        raise FileNotFoundError(f"Could not find test folder '{TEST}'. ")

    # The test dataset and map class indices are loaded back to class name.
    dataset = datasets.ImageFolder(TEST, transform = test_transforms)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found under '{TEST}'.")

    # Prepare the test data loader and class-label mappings needed for evaluation.
    print("Test class mapping:", dataset.class_to_idx)
    converted = {v: k for k, v in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)
    model = pytorch_model()
    # Here, I build a confusion matrix where rows = true labels and columns = predicted labels.
    matrix = torch.zeros((2, 2), dtype=torch.int64)
    # This list stores per-image results, including filename, true_label, predicted_label, and confidence, for error analysis.
    results = []

    # Here, the trained model is run on the test dataset to collect predictions for evaluation.
    with torch.no_grad():
        offset = 0
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            probabilities = torch.softmax(logits, dim = 1)
            predictions = probabilities.argmax(dim = 1)
            confidence = probabilities.max(dim = 1).values

            # The confusion matrix is updated here by comparing the true labels to the predicted labels.
            for t, p in zip(labels.view(-1), predictions.view(-1)):
                matrix[t.long(), p.long()] += 1

            # I record the accuracy of each image result here.
            batch_size = labels.size(0)
            for i in range(batch_size):
                path = dataset.samples[offset + i][0]
                true_idx = dataset.samples[offset + i][1]
                predicted_idx = int(predictions[i].item())
                conf = float(confidence[i].item())

                results.append((
                    os.path.basename(path),
                    converted[int(true_idx)],
                    converted[predicted_idx],
                    conf
                ))

            offset += batch_size

    # Here, the confusion matrix is used to determine accuracy.
    total = int(matrix.sum().item())
    correct = int((matrix[0, 0] + matrix[1, 1]).item())
    accuracy = correct / max(total, 1)

    # Recall formula: TP / (TP + FN)
    recall_occupied = matrix[0, 0].item() / max((matrix[0, 0] + matrix[0, 1]).item(), 1)
    recall_unoccupied = matrix[1, 1].item() / max((matrix[1, 0] + matrix[1, 1]).item(), 1)
    balanced_acc = 0.5 * (recall_occupied + recall_unoccupied)

    print("\nTest Results")
    print(f"Total images: {total}")
    print(f"Accuracy: {accuracy:.3f}  ({correct}/{total})")
    print(f"Balanced Acc: {balanced_acc:.3f}")
    print(f"Recall occupied: {recall_occupied:.3f}")
    print(f"Recall unoccupied: {recall_unoccupied:.3f}")
    print("Confusion matrix (rows = true cols = pred):")
    print(f"[[{matrix[0,0].item()}, {matrix[0,1].item()}],")
    print(f" [{matrix[1,0].item()}, {matrix[1,1].item()}]]")

    # Here, I identify the misclassified images.
    misses = [r for r in results if r[1] != r[2]]
    print(f"\nMisses ({len(misses)}/{total}):")
    if len(misses) == 0:
        print("None")
    else:
        for fn, t, p, conf in sorted(misses, key=lambda x: x[3]):
            print(f"{fn:30s} true = {t:10s} predicted = {p:10s} confidence = {conf:.3f}")

    # Additionally, the correctly classified images with the lowest confidence are identified.
    correct_rows = [r for r in results if r[1] == r[2]]
    correct_rows_sorted = sorted(correct_rows, key=lambda x: x[3])

    print("\nLowest-confidence correct predictions:")
    for fn, t, p, conf in correct_rows_sorted[:10]:
        print(f"{fn:30s} label={t:10s} conf={conf:.3f}")


if __name__ == "__main__":
    main()