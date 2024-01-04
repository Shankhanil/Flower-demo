from utils.model import MasterModel
import torch
from utils.prepare_dataset import get_mnist
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(MasterModel, trainloader, optimizer, epochs):
    """Train the MasterModelwork on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    MasterModel.train()
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            loss = criterion(MasterModel(images), labels)
            loss.backward()
            optimizer.step()
    return MasterModel


def test(MasterModel, testloader):
    """Validate the MasterModelwork on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    MasterModel.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = MasterModel(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def run_centralised(epochs: int, lr: float, momentum: float = 0.9):
    """A minimal (but complete) training loop"""

    # instantiate the model
    model = MasterModel(num_classes=10)

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # get dataset and construct a dataloaders
    trainset, testset = get_mnist()
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128)

    # train for the specified number of epochs
    trained_model = train(model, trainloader, optim, epochs)

    # training is completed, then evaluate model on the test set
    loss, accuracy = test(trained_model, testloader)
    print(f"{loss}")
    print(f"{accuracy}")