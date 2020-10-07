import os
from pathlib import Path
from torch.utils.data import Dataset


class CollectData(Dataset):
    def __init__(self, path_to_dataset, extension=['wav', 'mp3', 'npy', 'pth'], subset=None, transform=None):
        """
        Assume directory structure as:
        - dataset (the level which path_to_dataset indicates)
            - trainingdata
                - class A
                - class B
                ...
            - testdata
                - class A
                - class B
                ...
        :param path_to_dataset: a list of path to dataset directory; thus allows multiple datasets
        :param subset: None|'train'|'test'; if None, both train and test sets are loaded
        :param transform: see https://pytorch.org/docs/stable/torchvision/transforms.html
        """
        assert isinstance(path_to_dataset, list), "The input path_to_dataset should be a list."
        assert subset in [None, 'train', 'test'], "subset should be in [None, 'train', 'test']"

        full_path_to_dataset = []
        for data_dir in path_to_dataset:
            if not subset:  # load both train and test folders
                full_path_to_dataset.append(os.path.join(data_dir, 'trainingdata'))
                full_path_to_dataset.append(os.path.join(data_dir, 'testdata'))
            elif subset == 'train':
                full_path_to_dataset.append(os.path.join(data_dir, 'trainingdata'))
            else:
                full_path_to_dataset.append(os.path.join(data_dir, 'testdata'))

        aggr_data_path = []
        aggr_label = []
        for data_dir in full_path_to_dataset:
            for subclass in sorted(os.listdir(data_dir)):
                if not(subclass.startswith(".")): # ignore hidden files/dirs
                    for f in sorted(os.listdir(os.path.join(data_dir, subclass))):
                        if not(f.startswith(".")) and any(ext in f for ext in extension): # ignore hidden files/dirs
                            aggr_data_path.append(os.path.join(data_dir, subclass, f))
                            aggr_label.append(subclass)

        self.path_to_dataset = path_to_dataset
        self.extension = extension
        self.subset = subset
        self.transform = transform
        self.path_to_data = aggr_data_path
        self.labels = aggr_label
        self.print_()

    def __len__(self):
        return len(self.path_to_data)

    def __getitem__(self, idx):
        if self.transform:
            return idx, self.labels[idx], self.transform(self.path_to_data[idx])
        return idx, self.labels[idx], self.path_to_data[idx]
    
    def print_(self):
        print('-- datasets.CollectData --')
        print('path_to_dataset: ', self.path_to_dataset)
        # for i in range(5):
        #     if i == 0:
        #         print('path_to_data: [\n\t', self.path_to_data[i])
        #     elif i == 4:
        #         print('\t', self.path_to_data[i], '\n]')
        #     else:
        #         print('\t', self.path_to_data[i])
        print('extensions: ', self.extension)
        print('subset: ', self.subset)
        print('transform: ', self.transform)
        print('labels: ', self.labels[:3])
        print('# of classes: ', len(set(self.labels)))
        print('length: ', len(self), '\n')


if __name__ == '__main__':
    path_to_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'myAudioDataset/audio')
    # path_to_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'myAudioDataset/m64-5s')
    
    dataset = CollectData(path_to_dataset=[path_to_data], extension=['wav', 'npy'], subset=None, transform=None)
    # print("the number of data: %d" % len(dataset))
    # try:
    #     print("the first five entries:")
    #     for n in range(5):
    #         print(dataset[n])
    # except:
    #     raise IndexError("There is none or fewer than 5 data in the input path")
