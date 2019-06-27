import torch
from torch.utils.data import DataLoader

from net import YOLO
from dataset import YOLODataset


def train():
    epochs = 100
    dataset = YOLODataset()
    dataloader = DataLoader(dataset,
                         batch_size=16,
                         shuffle=True,
                         num_workers=0)
    net = YOLO()

    for _ in range(epochs):
        for img, label in dataloader:
            net.train()
            out = net.forward(img)
            print(out)
            print(out.shape)
            print(label.keys())
            print(label["coord"].shape)
            assert()


if __name__ == "__main__":
    train()
