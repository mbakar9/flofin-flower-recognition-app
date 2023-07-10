
import os
from tkinter import Tk
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tk().withdraw()
data_dir = os.path.realpath(os.path.dirname(__file__))
model_path = os.path.join(data_dir, "model.pt")

def classify():
    model = torch.load(model_path)
    model.to(device)
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    return classify_image(model, train_dataset, device)


def classify_image(model, train_dataset, device):

    Tk().withdraw()
    image_path = data_dir + "/static/test.jpg"
    img_batch = load_image(image_path, transform=transform)

    with torch.no_grad():
        outputs = model(img_batch)
        _, predicted = torch.max(outputs.data, 1)

    class_names = train_dataset.classes
    return f"{class_names[predicted.item()]}"
    
def load_image(image_path, url=False, transform=None):
    img = Image.open(image_path).convert('L')
    img = img.convert("RGB")
    if transform:
        img_transformed = transform(img)
    else:
        img_transformed = transforms.ToTensor()(img)
    img_batch = img_transformed.unsqueeze(0)
    if device:
        img_batch = img_batch.to(device)
    return img_batch