from abc import ABC, abstractmethod
import torchvision
import torchvision.transforms as transforms
import torch

# Abstract base class for data loaders
class AbstractDataLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass

# Concrete implementation of data loader for MNIST
class MNISTDataLoader(AbstractDataLoader):
    def __init__(self, batch_size=32, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )

    def load_data(self):
        # Download and load the training data
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=self.num_workers)

        # Download and load the testing data
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=self.num_workers)

        return self.trainloader, self.testloader

# Example usage
if __name__ == '__main__':
    mnist_loader = MNISTDataLoader()
    trainloader, testloader = mnist_loader.load_data()
