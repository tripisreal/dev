import torch
from PIL import Image
from torch import nn, load
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

# Get data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to range [-1, 1]
])

train = datasets.CIFAR10(root="data", download=True, train=True, transform=transform)

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.model(x)

# Instance of the neural network
clf = ImageClassifier().to('cuda')

# Load trained model state
with open('model_state-CIFAR10.pt', 'rb') as f:
    clf.load_state_dict(load(f))

image_path = 'CIFAR10-images/img_2.jpg'
img = Image.open(image_path)

img_tensor = transform(img).unsqueeze(0).to('cuda')

output = clf(img_tensor)
probabilities = torch.softmax(output, dim=1)  # Apply softmax to convert logits to probabilities
predicted_prob, predicted_label = torch.max(probabilities, dim=1)
predicted_label -= 1  # Adjust label to match zero-indexed class_names list

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

predicted_class = class_names[predicted_label.item()]
confidence = predicted_prob.item() * 100

print(f"\nDEBUG\n ----------------------------- \n Predicted class: {predicted_class}\n Predicted label: {predicted_label.item()}\n Prediction confidence: {confidence:.2f}%\n ----------------------------- ")
print(f"\nThis is a {predicted_class}!\n")

    # for epoch in range(10):  # Train for 10 epochs
    #     for batch in dataset:
    #         X, y = batch
    #         X, y = X.to('cuda'), y.to('cuda')
    #         yhat = clf(X)
    #         loss = loss_fn(yhat, y)

    #         # Apply backpropagation
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #     print(f"Epoch:{epoch} loss is {loss.item()}")

    # with open('model_state-CIFAR10.pt', 'wb') as f:
    #     save(clf.state_dict(), f)
