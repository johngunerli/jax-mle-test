import torch
from torchvision import datasets, transforms


def get_dataloader(batch_size, train=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("./data", train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train
    )
    return dataloader
