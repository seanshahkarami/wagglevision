import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from wagglevision.datasets import CloudDataset
from wagglevision.models import fcn_resnet101
from PIL import Image


transform = Compose([
    Resize(112),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


label_transform = Compose([
    # NEAREST is needed to preserve pixel classes
    Resize(112, interpolation=Image.NEAREST),
])


def transforms(image, label):
    return transform(image), label_transform(label)


train_data = CloudDataset(root='data', image_set='train',
                          transforms=transforms, download=True)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

val_data = CloudDataset(root='data', image_set='val',
                        transforms=transforms, download=False)

val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

net = fcn_resnet101(pretrained=True, num_classes=2)

learner = optim.Adam(net.parameters(), lr=1e-2)
lossfunc = nn.CrossEntropyLoss()

# do single train interation
net.train()

for image, label in train_loader:
    learner.zero_grad()
    loss = lossfunc(net(image), label)
    loss.backward()
    learner.step()
    print('loss', loss)

# do single eval iteration
acc_sum = 0.0

with torch.no_grad():
    net.eval()

    for image, label in val_loader:
        predict = net(image).argmax(1)
        acc_sum += (predict == label).float().mean()

val_acc = acc_sum / len(val_loader)
print('val_acc', val_acc)
