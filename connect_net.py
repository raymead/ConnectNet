from typing import Tuple

import torch
import torch.nn
import torch.nn.functional


class ConnectNet(torch.nn.Module):
    def __init__(self, channel_size: int = 128, hidden_size: int = 128):
        super().__init__()

        self.channel_size = channel_size
        self.hidden_size = hidden_size

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=self.channel_size,
                kernel_size=(4, 4), stride=1, padding=0,
            ),
            torch.nn.Tanh(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.channel_size, out_channels=self.hidden_size,
                kernel_size=(3, 4), stride=1, padding=0,
            ),
            torch.nn.Tanh(),
            torch.nn.Flatten(),

            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.Tanh(),

            torch.nn.Linear(self.hidden_size, 8)
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.fc(output)
        return torch.tanh(output[:, 0]), torch.nn.functional.softmax(output[:, 1:], dim=1)


class NetworkCache:
    def __init__(self, nnet: torch.nn.Module) -> None:
        self.nnet = nnet
        self.V = {}
        self.P = {}

    def network(self, state: torch.Tensor, rep: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if rep not in self.V:
            v, prob_a = self.nnet(state.view(1, 1, 6, 7))
            self.V[rep] = v
            self.P[rep] = prob_a[0]
        return self.V[rep], self.P[rep]


def save_model(nnet: torch.nn.Module, path: str) -> None:
    print(f"Saving nnet to path :: {path}")
    torch.save(nnet.state_dict(), f"{path}")


def load_model(path: str, log: bool = True) -> torch.nn.Module:
    model = ConnectNet()
    model.load_state_dict(torch.load(path))
    if log:
        print(model.eval())
    else:
        model.eval()
    return model
