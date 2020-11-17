import torch
from torch import Generator
from torchvision.transforms import Resize
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import os

class THINGSLoader():
    def __init__(self, data_dir="data/THINGS/", shape=None):
        IMAGES_DIR = data_dir+'images/'
        transform = None
        if shape:
            transform = Resize(shape)
            self.images = np.zeros((1854, 3, shape[0], shape[1]))
        else:
            self.images = np.zeros((1854, 3, 350, 350))
        for i in range(1, 1855):
            fname = f"{i:04d}.jpg"
            img = Image.open(f"{IMAGES_DIR}{fname}")
            if transform: 
                img = transform(img)
            self.images[i-1] = np.asarray(img).T / 255.0
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        del self.images
        
class THINGSTriplets(Dataset):
    def __init__(self, things_loader, csv_path="data/THINGS/triplets_to56.csv"):
        self.y = np.genfromtxt(csv_path, dtype=int)
        self.y -= 1
        self.things_loader = things_loader
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        y_ = self.y[idx]
        label = y_[-1]
        X1 = torch.tensor(self.things_loader[y_[0]], dtype=torch.float32)
        X2 = torch.tensor(self.things_loader[y_[1]], dtype=torch.float32)
        X3 = torch.tensor(self.things_loader[y_[2]], dtype=torch.float32)
        
        x = np.arange(3)*y_.shape[0]
        available_pos = np.random.permutation(np.arange(3)[np.arange(3) != label])
        positions = torch.tensor(np.hstack((available_pos, [label])), dtype=torch.long)
        return X1, X2, X3, positions
    

class FullTHINGS(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = Resize((128, 128))
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        img = torch.tensor(np.asarray(img).T / 255.0, dtype=torch.float32)
        return img

def get_things(batch_size=64, seed=0, things_dir="/nfs/THINGS/images_raw/"):
    """
    Returns train and test DSprites dataset.
    """
    things_loader = THINGSLoader(shape=(128, 128))
    generator = Generator().manual_seed(seed)
    data = THINGSTriplets(things_loader=things_loader)

    pretrain_data = DataLoader(FullTHINGS(things_dir), batch_size=64, shuffle=True)

    train_data, test_data = random_split(data, [1446680, 15000])# TODO, generator=generator)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)#, pin_memory=True, num_workers=16)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)#, pin_memory=True, num_workers=16)                    
    return pretrain_data, train_data, test_data

if __name__ == '__main__':
    things_loader = THINGSLoader(data_dir="THINGS/", shape=(128, 128))
    data = DataLoader(THINGSTriplets(things_loader=things_loader, csv_path="THINGS/triplets_to56.csv"),
         batch_size=5, shuffle=True)

    print(len(data.dataset))
    x1, x2, x3, pos = next(iter(data))
    print("pos", pos)
    print(x1.shape)
    x1 = x1.reshape(5, 3, 128, 128)
    x2 = x2.reshape(5, 3, 128, 128)
    x3 = x3.reshape(5, 3, 128, 128)
    fig, axes = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(7, 10))
    for i in range(5):
        axes[i][0].imshow(x1[i].T, cmap='Greys_r')
        axes[i][1].imshow(x2[i].T, cmap='Greys_r')
        axes[i][2].imshow(x3[i].T, cmap='Greys_r')
        [x.grid(False) for x in axes[i]]
        [x.set_xticks([]) for x in axes[i]]
        [x.set_yticks([]) for x in axes[i]]

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0.05, 
                        top=0.97, bottom=0.05, 
                        left=0, right=0.31)
    plt.show()