from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ColorJitter
from wagglevision.datasets import CloudDataset
from PIL import Image


transform = Compose([
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    Resize(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


label_transform = Compose([
    # NEAREST is needed to preserve pixel classes
    Resize(224, interpolation=Image.NEAREST),
])


def transforms(image, label):
    return transform(image), label_transform(label)


train_data = CloudDataset(root='data',
                          image_set='train',
                          transforms=transforms,
                          download=True)

val_data = CloudDataset(root='data',
                        image_set='val',
                        transforms=transforms,
                        download=False)

train_loader = DataLoader(train_data,
                          batch_size=25,
                          shuffle=True,
                          pin_memory=True)

val_loader = DataLoader(val_data,
                        batch_size=25,
                        shuffle=False,
                        pin_memory=True)

for image, label in train_loader:
    print(image.shape, label.shape)
