import torch
import torch.nn as nn
from torchvision.models import resnet18


class YOLO(nn.Module):
    def __init__(self, B=2, C=20):
        super(YOLO, self).__init__()
        self.features = list(resnet18(pretrained=True).children())[:-2]
        self.features = nn.Sequential(*self.features)
        self.regressor = nn.Conv2d(512, (5 * B + C), 1)
        print(self.features)

    def forward(self, X):
        X = self.features(X)
        X = self.regressor(X)
        return X


if __name__ == "__main__":
    dummy_image = torch.rand((1, 3, 448, 448))
    X = YOLO()(dummy_image)
    print(X.shape)
