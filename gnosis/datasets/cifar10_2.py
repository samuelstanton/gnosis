import numpy as np
import torchvision
import os


class CIFAR102(torchvision.datasets.CIFAR10):
  
    def __init__(self, root, download, train, transform, cifar10_fraction):
        """Loads CIFAR10.2 as pytorch dataset.
        
        Note: test set coincides with the regular CIFAR10.
        
        Args:
          root, download, train, transforms: same as torchvision
          cifar10_fraction: fraction of CIFAR10 train set to add to the CIFAR10.2
        """
        super().__init__(root, train, transform, download=download)
        
        if download:
            files_exist = self.check_files_exist(root)
            print(files_exist)
            if not files_exist:
                self.download_c102(root)
        
        if train:
            original_data = self.data
            original_targets = self.targets
            num_retained = int(len(original_data) * cifar10_fraction)
            
            ds = np.load(os.path.join(root, "cifar102_train.npy"), allow_pickle=True).item()
            new_data, new_targets = ds["images"], ds["labels"]
            
            self.data = np.concatenate([original_data[:num_retained], new_data])
            self.targets = np.concatenate([original_targets[:num_retained], new_targets]).astype(np.int64)
    
    @staticmethod
    def check_files_exist(root):
        return all([os.path.isfile(os.path.join(root, "cifar102_{}.npy".format(split)))
                    for split in ["train", "test"]])
    
    @staticmethod
    def download_c102(root):
        for split in ["train", "test"]:
            file_path = os.path.join(root, "cifar102_{}.npy".format(split))
            os.system(
                "wget -O {} https://github.com/modestyachts/cifar-10.2/raw/"
                "master/cifar102_{}.npy".format(file_path, split))
