from wagglevision.datasets import CloudDataset
from wagglevision.models import fcn_resnet101

# test datasets
train_data = CloudDataset('data', image_set='train', download=True)
val_data = CloudDataset('data', image_set='val', download=False)

print(len(train_data))
print(train_data[0])

print(len(val_data))
print(val_data[0])
