import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Define the Artificial Neural Network (ANN) with ReLU activation
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)  # Increase units in the first hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)  # Add a second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)  # Add a third hidden layer
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)  # Output layer

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
    
if __name__ == '__main__':

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the pixel values
    ])

    # Download and load the CIFAR-10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Create a DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    # Download and load the CIFAR-10 test dataset
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)    

    # Instantiate the ANN model, loss function, and optimizer
    ann_model = ANN().to(device)
    criterion_ann = nn.CrossEntropyLoss()
    optimizer_ann = optim.SGD(ann_model.parameters(), lr=0.01, momentum=0.9)

    # Training loop for ANN with mini-batch gradient descent
    print("Training data...")
    for epoch in range(10):  # Adjust the number of epochs as needed
        ann_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_ann.zero_grad()
            outputs = ann_model(inputs)
            loss = criterion_ann(outputs, labels)
            loss.backward()
            optimizer_ann.step()

            running_loss += loss.item()
    
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    print("Training completed!")

    # Evaluate ANN model on the test set
    ann_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = ann_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the ANN on the test set: {100 * correct / total}%")


# # Results: 
# Training data...
# Epoch 1, Loss: 1.763975113554074
# Epoch 2, Loss: 1.4647862037734303
# Epoch 3, Loss: 1.3384879416669422
# Epoch 4, Loss: 1.2436119726551769
# Epoch 5, Loss: 1.168443648940157
# Epoch 6, Loss: 1.0930330592500583
# Epoch 7, Loss: 1.0273479946586481
# Epoch 8, Loss: 0.9616936314136476
# Epoch 9, Loss: 0.9015462623761438
# Epoch 10, Loss: 0.8384432461484314
# Training completed!
# Accuracy of the ANN on the test set: 54.74%