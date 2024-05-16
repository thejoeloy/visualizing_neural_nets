# MLP with Batch Norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(784, 30)
        self.bn1 = nn.BatchNorm1d(30)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(30, 30)
        self.bn2 = nn.BatchNorm1d(30)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(30, 10)
        self.bn3 = nn.BatchNorm1d(10)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

dnn_model = DNN()

# Print the model architecture
print(dnn_model)

total_params = sum(p.numel() for p in dnn_model.parameters())
print(f"Number of parameters in the model: {total_params}")

model = DNN()

model.load_state_dict(torch.load('dnn_model.pth'))

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define transformations to apply to the input image
transform = transforms.Compose([
    transforms.Resize((28, 28)),     # Resize image to match input size of the model
    transforms.Grayscale(),          # Convert image to grayscale
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize image pixels to range [-1, 1]
])

# Load and preprocess the image
img = Image.open('img.png')
img = transform(img).unsqueeze(0)  # Add batch dimension

# Move the input image to the appropriate device
img = img.to(device)

# Perform inference
with torch.no_grad():
    output = model(img)

# Get predicted class
_, predicted_class = torch.max(output, 1)

print("Predicted class:", predicted_class.item())  
