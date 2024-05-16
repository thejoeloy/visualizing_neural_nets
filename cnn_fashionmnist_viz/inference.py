import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class FashionCNNModel(nn.Module):
    def __init__(self):
        super(FashionCNNModel, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()  # Add ReLU layer
        
        # Max Pooling Layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()  # Add ReLU layer
        
        # Max Pooling Layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()  # Add ReLU layer
        
        # Output Layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional Layer 1
        x = self.pool1(self.relu1(self.batch_norm1(self.conv1(x))))
        
        # Convolutional Layer 2
        x = self.pool2(self.relu2(self.batch_norm2(self.conv2(x))))
        
        # Flatten Layer
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully Connected Layer 1
        x = self.relu3(self.batch_norm3(self.fc1(x)))
        
        # Output Layer
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# Instantiate model
model = FashionCNNModel()

# Load model state dictionary
model.load_state_dict(torch.load('fashion_model.pth'))

# Put the model in evaluation mode
model.eval()

# Move the model to the appropriate device (CPU or GPU)
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
img = Image.open('img2.png')
img = transform(img).unsqueeze(0)  # Add batch dimension

# Move the input image to the appropriate device
img = img.to(device)

# Perform inference
with torch.no_grad():
    output = model(img)

# Get predicted class
_, predicted_class = torch.max(output, 1)

print("Predicted class:", predicted_class.item())  
