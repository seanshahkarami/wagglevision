from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive

dataset_url = 'https://web.lcrc.anl.gov/public/waggle/datasets/WaggleClouds.tar.gz'


class CloudDataset(Dataset):

    def __init__(self, root, image_set='train', transforms=None, download=True):
        if download:
            download_and_extract_archive(dataset_url, root, root)

        self.root = Path(root, 'WaggleClouds')
        self.files = Path(self.root, image_set + '.list').read_text().split()
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        name = self.files[index]
        image = Image.open(Path(self.root, 'images', name + '.jpg')).convert('RGB')
        label = Image.open(Path(self.root, 'labels', name + '.png'))
        if self.transforms is not None:
            image, label = self.transforms(image, label)
        return image, label
