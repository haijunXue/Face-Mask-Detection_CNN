import torch
from matplotlib import pyplot as plt
import numpy as np

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


"""
steps = 100
x = np.linspace(0, 2 * np.pi, steps)
data = np.sin(x)

# Plot the data using plot_curve
plot_curve(data)


import torch
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Get a batch of 6 images and their corresponding labels
images, labels = zip(*[mnist_data[i] for i in range(6)])

# Convert the images and labels to tensors
images = torch.stack(images)
labels = torch.tensor(labels)


# Example of using the plot_image function
plot_image(images, labels, "Digit")
"""
def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


import torch


def one_hot(label: torch.Tensor, depth: int = 10) -> torch.Tensor:
    # Create a tensor of zeros with shape (batch_size, depth)
    out = torch.zeros(label.size(0), depth)

    # Convert label to a LongTensor and reshape to (batch_size, 1)
    idx = label.long().view(-1, 1)

    # Scatter the value 1 at the appropriate indices
    out.scatter_(dim=1, index=idx, value=1)

    return out

"""
labels = torch.tensor([0, 2, 1, 4])
print(labels.size(0))
idx = labels.long().view(-1, 1)
print(idx)
one_hot_encoded = one_hot(labels, depth=5)
print(one_hot_encoded)
"""
