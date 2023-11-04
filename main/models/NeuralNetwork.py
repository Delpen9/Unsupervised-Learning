# Standard Libraries
import numpy as np
import pandas as pd

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Model Evaluation
from sklearn.metrics import roc_auc_score, accuracy_score


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_dimension_1=128, hidden_dimension_2=64):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_dimension_1)
        self.layer2 = nn.Linear(hidden_dimension_1, hidden_dimension_2)
        self.layer3 = nn.Linear(hidden_dimension_2, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dimension_1=128, hidden_dimension_2=64):
        super(MultiClassClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_dimension_1)
        self.layer2 = nn.Linear(hidden_dimension_1, hidden_dimension_2)
        self.layer3 = nn.Linear(hidden_dimension_2, num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def tune_neural_network(
    train_loader, val_loader, input_size, num_epochs=10, learning_rate=0.001, multiclass=False, num_classes=2,
    hidden_dimension_1=None, hidden_dimension_2=None
) -> tuple[any, list, list]:
    if multiclass == False:
        model = BinaryClassifier(input_size, hidden_dimension_1, hidden_dimension_2)
    else:
        model = MultiClassClassifier(input_size, num_classes, hidden_dimension_1, hidden_dimension_2)

    if multiclass == False:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    validation_loss_history = []
    training_loss_history = []
    for epoch in range(num_epochs):
        validation_loss_epoch = []
        training_loss_epoch = []

        # Training loop
        for inputs, labels in train_loader:
            outputs = model(inputs)
            if multiclass == False:
                loss = criterion(outputs, labels.unsqueeze(1))
            else:
                loss = criterion(outputs, labels.to(torch.long))

            training_loss_epoch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                if multiclass == False:
                    loss = criterion(outputs, labels.unsqueeze(1))
                else:
                    loss = criterion(outputs, labels.to(torch.long))

                validation_loss_epoch.append(loss.item())
                val_loss += loss.item()

        validation_loss_history.append(np.mean(np.array(validation_loss_epoch)))
        training_loss_history.append(np.mean(np.array(training_loss_epoch)))

        print(
            f"Epoch {epoch+1}/{num_epochs} - Training Loss: {loss.item():.4f}, Val Loss: {val_loss / len(val_loader):.4f}"
        )

    return (model, training_loss_history, validation_loss_history)


def evaluate_model(model, test_loader, num_classes=2):
    model.eval()
    true_labels = []
    predicted_probs = []
    predicted_labels = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)

            if num_classes == 2:
                predicted = torch.round(outputs)
                predicted_labels.extend(predicted.numpy())
                predicted_probs.extend(outputs.numpy())
                true_labels.extend(labels.numpy())
            else:
                probs = softmax(outputs)
                predicted = torch.argmax(outputs, dim=1)
                predicted_labels.extend(predicted.numpy())
                predicted_probs.extend(probs.numpy())
                true_labels.extend(labels.numpy())

    auc = roc_auc_score(true_labels, predicted_probs, multi_class='ovr', average='macro')
    accuracy = accuracy_score(true_labels, predicted_labels)

    return auc, accuracy


if __name__ == "__main__":
    data = pd.read_csv("../../data/auction_verification_dataset/data.csv")

    X = data.iloc[:, :-2].copy()
    y = data.iloc[:, -2].copy().astype(int)

    dataset = TensorDataset(
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32),
    )

    train_size = int(0.8 * len(dataset))
    temp_size = len(dataset) - train_size
    train_dataset, temp_dataset = random_split(dataset, [train_size, temp_size])

    val_size = int(0.5 * temp_size)
    test_size = temp_size - val_size
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

    input_size = X.shape[1]
    num_epochs = 200

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    best_model, _, _ = tune_neural_network(train_loader, val_loader, input_size, num_epochs)

    auc, accuracy = evaluate_model(best_model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
