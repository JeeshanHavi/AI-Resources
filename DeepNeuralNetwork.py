import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DeepNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DeepNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.l5 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Weight initialization
        nn.init.kaiming_normal_(self.l1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.l2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.l3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.l4.weight, nonlinearity='relu')
        # For the output layer
        nn.init.xavier_normal_(self.l5.weight)  

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l5(out)
        return out  # Return raw logits

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 784  # Example for MNIST
    hidden_size = 128
    num_classes = 10
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Define test dataset and loader
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss function, optimizer
    model = DeepNeuralNet(input_size, hidden_size, num_classes).cuda()  # Move model to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()
