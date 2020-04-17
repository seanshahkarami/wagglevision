from wagglevision.datasets import CloudDataset
from wagglevision.models import fcn_resnet101

# test datasets
train_data = CloudDataset('data', image_set='train')
val_data = CloudDataset('data', image_set='val')

print(len(train_data))
print(len(val_data))

# test models
net = fcn_resnet101(pretrained=True, num_classes=2)
