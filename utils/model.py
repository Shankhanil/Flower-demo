import torch
import torch.nn as nn
import torch.nn.functional as F


class MasterModel(nn.Module):
    """Master neural network that will be used by all clients. [Replace this with your model]

    Args:
        Module (torch.nn): Torch neural network module API
    """    
    def __init__(self, num_classes: int) -> None:
        super(MasterModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neural network forward function

        Args:
            x (torch.Tensor): Input tensor [Replace with your inputs]

        Returns:
            torch.Tensor: Output tensor [Replace with your labels]
        """        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x