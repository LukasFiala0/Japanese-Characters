import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# ===================
# Load and preprocess data
# ===================
def load_data(file_path):
    data = pd.read_csv(file_path, dtype=np.float32)
    features = data.iloc[:, 2:].values / 255
    labels = data.label.values
    return features, labels

features_train, targets_train = load_data("/content/drive/MyDrive/ComputerVision/k49_train_data.csv")
features_test, targets_test = load_data("/content/drive/MyDrive/ComputerVision/k49_test_data.csv")

# Split training data into training and validation sets
features_train, features_val, targets_train, targets_val = train_test_split(
    features_train, targets_train, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
def to_tensor(features, targets):
    return torch.from_numpy(features), torch.from_numpy(targets).type(torch.LongTensor)

features_train_tr, targets_train_tr = to_tensor(features_train, targets_train)
features_val_tr, targets_val_tr = to_tensor(features_val, targets_val)
features_test_tr, targets_test_tr = to_tensor(features_test, targets_test)

# ===================
# Define the CNN model
# ===================
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(16, 32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# Instantiate model, loss function, and optimizer
model = BasicCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===================
# Helper functions for training and evaluation
# ===================
def evaluate(loader):
    model.eval()
    total_loss, correct, y_true, y_pred = 0, 0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(-1, 1, 28, 28)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, accuracy, f1

# ===================
# Training and validation
# ===================
train_loader = DataLoader(TensorDataset(features_train_tr, targets_train_tr), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(features_val_tr, targets_val_tr), batch_size=128, shuffle=False)
num_epochs = 15

metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, y_true, y_pred = 0, 0, [], []
    for images, labels in train_loader:
        images = images.view(-1, 1, 28, 28)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, 1)
        correct += (preds == labels).sum().item()
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

    train_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    train_f1 = f1_score(y_true, y_pred, average="macro")

    val_loss, val_acc, val_f1 = evaluate(val_loader)

    metrics["train_loss"].append(train_loss)
    metrics["val_loss"].append(val_loss)
    metrics["train_acc"].append(train_acc)
    metrics["val_acc"].append(val_acc)
    metrics["train_f1"].append(train_f1)
    metrics["val_f1"].append(val_f1)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# ===================
# Visualize training progress
# ===================
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, metrics["train_loss"], label="Train Loss")
plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(epochs, metrics["train_acc"], label="Train Accuracy")
plt.plot(epochs, metrics["val_acc"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, metrics["train_f1"], marker="o", label="Train F1 Score")
plt.plot(epochs, metrics["val_f1"], marker="x", label="Validation F1 Score")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1 Score Progress")
plt.show()

# ===================
# Test Evaluation
# ===================
## 1.) Testing on data from same source - but those data are for testing - not the same as training
test_loader = DataLoader(TensorDataset(features_test_tr, targets_test_tr), batch_size=128, shuffle=False)
test_loss, test_acc, test_f1 = evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")
## 2.) Testing on data from different source
diff_features_test, diff_targets_test = load_data("diff_test_data")
diff_features_test_tr, diff_targets_test_tr = to_tensor(diff_features_test, diff_targets_test)
diff_test_loader = DataLoader(TensorDataset(diff_features_test_tr, diff_targets_test_tr), batch_size=128, shuffle=False)
diff_test_loss, diff_test_acc, diff_test_f1 = evaluate(diff_test_loader)
print(f"Test Loss: {diff_test_loss:.4f}, Test Accuracy: {diff_test_acc:.4f}, Test F1 Score: {diff_test_f1:.4f}")
