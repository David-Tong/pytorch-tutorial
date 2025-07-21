import torchvision.transforms as transforms
from PIL import Image

# transform 1
transform = transforms.Compose([
    transforms.Resize((892, 1280)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('./pytorch-dataset-dataloader/image.jpg')
image.show()

image_tensor = transform(image)
print(image_tensor.shape)

# display image
to_pil = transforms.ToPILImage()
pil_image = to_pil(image_tensor)

pil_image.show()

# transform 2
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(30), 
    transforms.RandomResizedCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_tensor = transform(image)

# display image
to_pil = transforms.ToPILImage()
pil_image = to_pil(image_tensor)

pil_image.show()