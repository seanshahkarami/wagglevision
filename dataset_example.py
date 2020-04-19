from wagglevision.datasets import CloudDataset
import matplotlib.pyplot as plt

# test datasets
train_data = CloudDataset('data', image_set='train', download=True)
val_data = CloudDataset('data', image_set='val', download=False)

# show a few examples
fig, ax = plt.subplots(ncols=3, nrows=4, sharex=True, sharey=True)

for i in range(4):
    image, label = train_data[i]
    ax[i, 0].imshow(image)
    ax[i, 1].imshow(label, cmap='jet', interpolation='none')
    ax[i, 2].imshow(image)
    ax[i, 2].imshow(label, cmap='jet', interpolation='none', alpha=0.5)

plt.show()
