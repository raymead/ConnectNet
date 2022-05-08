import torch
import torch.nn
import torch.nn.functional


class ConnectNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), stride=1, padding=0),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 4), stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten(),

            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 8)
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.fc(output)
        return torch.tanh(output[:, 0]), torch.nn.functional.softmax(output[:, 1:], dim=1)
