from typing import Tuple, Type

import torch
import torch.nn.functional


class ConnectNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.channel_size = 128
        self.hidden_size = 128

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


class ConnectNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channel_size = 64
        hidden_size = 64
        dropout_p = 0.4

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=channel_size, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=channel_size, out_channels=channel_size, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=channel_size, out_channels=channel_size, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=channel_size, out_channels=channel_size, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=channel_size, out_channels=channel_size, kernel_size=(2, 2),
            stride=1, padding=0,
        )

        self.conv_layers = torch.nn.Sequential(
            self.conv1, torch.nn.Tanh(),
            self.conv2, torch.nn.Tanh(),
            self.conv3, torch.nn.Tanh(),
            self.conv4, torch.nn.Tanh(),
            self.conv5, torch.nn.Tanh(),
            torch.nn.Flatten(),
        )

        self.fc1 = torch.nn.Linear(1 * 2 * channel_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 8)

        self.dropout1 = torch.nn.Dropout(p=dropout_p)
        self.fc_layers = torch.nn.Sequential(
            self.dropout1,
            self.fc1, torch.nn.Tanh(),
            self.fc2, torch.nn.Tanh(),
            self.fc3,
        )

    def forward(self, x):
        output = self.conv_layers(x)
        output = self.fc_layers(output)
        return torch.tanh(output[:, 0]), torch.nn.functional.softmax(output[:, 1:], dim=1)


class ConnectNet3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        channel_size = 64
        hidden_size = 128
        dropout_p = 0.4

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=channel_size, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=channel_size, out_channels=channel_size, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=channel_size, out_channels=channel_size, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=channel_size, out_channels=channel_size, kernel_size=(2, 2),
            stride=1, padding=0,
        )

        self.conv_layers = torch.nn.Sequential(
            self.conv1, torch.nn.ReLU(),
            self.conv2, torch.nn.ReLU(),
            self.conv3, torch.nn.ReLU(),
            self.conv4, torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.fc1 = torch.nn.Linear(2 * 3 * channel_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 8)

        self.dropout1 = torch.nn.Dropout(p=dropout_p)
        self.fc_layers = torch.nn.Sequential(
            self.dropout1,
            self.fc1, torch.nn.ReLU(),
            self.fc2, torch.nn.ReLU(),
            self.fc3,
        )

    def forward(self, x):
        output = self.conv_layers(x)
        output = self.fc_layers(output)
        return torch.tanh(output[:, 0]), torch.nn.functional.softmax(output[:, 1:], dim=1)


class ConnectNet4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 256
        dropout_p = 0.5

        channel_size_1 = 64
        channel_size_2 = 96
        channel_size_3 = 144
        channel_size_4 = 288

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=channel_size_1, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=channel_size_1, out_channels=channel_size_2, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=channel_size_2, out_channels=channel_size_3, kernel_size=(2, 2),
            stride=1, padding=0,
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=channel_size_3, out_channels=channel_size_4, kernel_size=(2, 2),
            stride=1, padding=0,
        )

        self.conv_layers = torch.nn.Sequential(
            self.conv1, torch.nn.ReLU(),
            self.conv2, torch.nn.ReLU(),
            self.conv3, torch.nn.ReLU(),
            self.conv4, torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.fc1 = torch.nn.Linear(2 * 3 * channel_size_4, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 8)

        self.dropout1 = torch.nn.Dropout(p=dropout_p)
        self.dropout2 = torch.nn.Dropout(p=dropout_p)
        self.fc_layers = torch.nn.Sequential(
            self.dropout1,
            self.fc1, torch.nn.ReLU(),
            self.dropout2,
            self.fc2, torch.nn.ReLU(),
            self.fc3,
        )

    def forward(self, x):
        output = self.conv_layers(x)
        output = self.fc_layers(output)
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
    torch.save(nnet.state_dict(), path)


def load_model(path: str, klass: Type[torch.nn.Module], log: bool = True) -> torch.nn.Module:
    model = klass()
    model.load_state_dict(torch.load(path))
    if log:
        print(model.eval())
    else:
        model.eval()
    return model
