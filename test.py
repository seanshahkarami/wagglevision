from wagglevision.datasets import CloudDataset

train_data = CloudDataset('data', image_set='train')
val_data = CloudDataset('data', image_set='val')

print(len(train_data))
print(len(val_data))
