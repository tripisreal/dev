# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
# 1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10),
        )

    def forward(self, x):
        return self.model(x)

# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    with open('model_state-MNIST.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    image_path = 'MNIST-images\img_1.jpg'
    img = Image.open(image_path)

    if img.mode == 'RGBA':
        img = img.convert('RGB')  # Discard alpha channel if present

    img_grayscale = img.convert('L')  # Convert to grayscale

    img_tensor = ToTensor()(img_grayscale).unsqueeze(0).to('cuda')

    output = clf(img_tensor)

    probabilities = torch.softmax(output, dim=1)  # Apply softmax to convert logits to probabilities
    predicted_prob, predicted_label = torch.max(probabilities, dim=1)
    predicted_label = torch.argmax(output)

    confidence = predicted_prob.item() * 100

    print(f"\nDEBUG\n ----------------------------- \n Predicted label: {predicted_label.item()}\n Prediction confidence: {confidence:.2f}%\n ----------------------------- ")
    print(f"\nThis is a {predicted_label}!\n")

    # for epoch in range(10): # Train for 10 epochs
    #     for batch in dataset:
    #         X,y = batch
    #         X, y = X.to('cuda'), y.to('cuda')
    #         yhat = clf(X)
    #         loss = loss_fn(yhat, y)

    #         # Apply backdrop
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #     print(f"Epoch:{epoch} loss is {loss.item()}")

    # with open('model_state-MNIST.pt', 'wb') as f:
    #     save(clf.state_dict(), f)
