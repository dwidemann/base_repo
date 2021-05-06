import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

class MnistDataLoader(DataLoader):
    def __init__(self, data_dir, training=False, batch_size=32, shuffle=False,
                 num_workers=1, collate_fn=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.training = training
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.dataset = datasets.MNIST(self.data_dir, train=self.training, 
                                      download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, 
                         num_workers=self.num_workers,collate_fn=self.collate_fn)

class FashionMnistDataLoader(DataLoader):
    def __init__(self, data_dir, training=False, batch_size=32, shuffle=False,
                 num_workers=1, collate_fn=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.training = training
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.dataset = datasets.FashionMNIST(self.data_dir, train=self.training, 
                                      download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, 
                         num_workers=self.num_workers,collate_fn=self.collate_fn)