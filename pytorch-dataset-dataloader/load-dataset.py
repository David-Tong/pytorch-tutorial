from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# download MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# create DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for inputs, labels in train_loader:
    print(inputs.shape)
    print(labels.shape)

# display image
image_tensor = inputs[0]
to_pil = transforms.ToPILImage()
pil_image = to_pil(image_tensor)

pil_image.show()
