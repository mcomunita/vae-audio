from torch.utils.data import Dataset
from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset import transformers, CollectData


class CollectDataLoader(BaseDataLoader):
    """
    Spectrogram data loader; inheritate BaseDataLoader which inheritates PyTorch DataLoader,
    refer to DataLoader in PyTorch Doc. for further details.
    Additional transformations applied to spectrograms include:
        1. Load spectrograms that were preprocessed and stored in data_dir
        2. Perform SpecChunking to slice spectrograms into fixed-length, non-overlapping chunks
    TODO:
        [] Remove SpecChunking
        [] Prolly make self.transform as arguments in config file
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=1, **kwargs):
        self.transform = transforms.Compose([
            transformers.LoadNumpyAry(),
            # transformers.SpecChunking(duration=0.5, sr=22050, hop_size=735, reverse=False)
        ])

        self.data_dir = data_dir
        self.dataset = CollectData(self.data_dir, transform=self.transform, **kwargs)
        super(CollectDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        self.print_()
    
    def print_(self):
        super(self.__class__, self).print_()
        print('-- collect data loader --')
        print('transform: ', self.transform)
        print('data_dir: ', self.data_dir)
        print('dataset: ', self.dataset)
        print()


if __name__ == '__main__':
    # specify subset as train to load the training data only
    train_dl = CollectDataLoader(data_dir=['dataset/myAudioDataset/m64-5s'],
                                 batch_size=8, shuffle=True, validation_split=0.1, num_workers=1, subset='train')
    # specify subset as test to load the testing data only
    test_dl = CollectDataLoader(data_dir=['dataset/myAudioDataset/m64-5s'],
                                 batch_size=8, shuffle=True, validation_split=0.0, num_workers=1, subset='test')
    # specify subset as None to load all the data
    dl = CollectDataLoader(data_dir=['dataset/myAudioDataset/m64-5s'],
                                 batch_size=8, shuffle=True, validation_split=0.2, num_workers=1, subset=None)
    # refer to PyTorch Doc. for DataLoader for further detail
